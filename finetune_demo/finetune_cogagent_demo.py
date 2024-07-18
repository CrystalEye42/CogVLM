import os
import torch
import argparse
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.helpers import print_rank0
from utils.models import FineTuneTrainCogAgentModel
from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor

def disable_untrainable_params(self):
    total_trainable = 0
    total = 0
    # enable = ['vit']
    enable = [] # ["encoder", "cross_attention", "linear_proj", 'mlp.vision', 'rotary.vision', 'eoi', 'boi', 'vit']
    if self.args.use_ptuning:
        enable.extend(['ptuning'])
    if self.args.use_lora or self.args.use_qlora:
        enable.extend(['matrix_A', 'matrix_B'])
    for n, p in self.named_parameters():
        flag = False
        total += p.numel()
        for e in enable:
            if type(e) is tuple:
                if e[0].lower() in n.lower() and e[1].lower() in n.lower() and 55 > int(n[:n.find('.mlp')].split('.')[-1]) > 45:
                    flag = True
                    break
            else:
                if e.lower() in n.lower():
                    flag = True
                    break
        if not flag:
            p.requires_grad_(False)
        else:
            total_trainable += p.numel()
            if 'encoder' in n or 'vit' in n:
                p.lr_scale = 0.1
            print_rank0(n)
    print_rank0("***** Total trainable parameters: "+str(total_trainable)+" *****")
    print_rank0("***** Total parameters: "+str(total)+" *****")

FineTuneTrainCogAgentModel.disable_untrainable_params = disable_untrainable_params

def data_collator(examples, cross_image_processor=None):
    def to_tensor(value):
        """Converts lists or numpy arrays to tensors."""
        if isinstance(value, list):
            return torch.tensor(value)
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return value
    
    def concatenate_tensors(attribute, key):
        """Concatenates tensors for a specific attribute and key."""
        if attribute is None:
            return torch.cat([ex[key] for ex in examples if isinstance(ex[key], torch.Tensor)])
        else:
            return torch.cat([ex[attribute][key] for ex in examples if isinstance(ex[attribute][key], torch.Tensor)])

    # Convert all lists and numpy arrays in examples to tensors
    for example in examples:
        for key, value in example.items():
            example[key] = to_tensor(value)

    # Extract and concatenate attributes from examples
    img_args = {}
    for attribute in ['vision', 'cross']:
        if attribute == 'cross' and cross_image_processor is None:
            continue

        if attribute in examples[-1]:  # Using the last example as reference
            for key in examples[-1][attribute]:
                tensor_key = f"{attribute}_{key}"
                tensors_to_concatenate = [ex[attribute][key] for ex in examples if isinstance(ex[attribute][key], torch.Tensor)]
                if tensors_to_concatenate:
                    img_args[tensor_key] = concatenate_tensors(attribute, key)
                else:
                    img_args[tensor_key] = examples[-1][attribute][key]

    # Remove 'vision' and 'cross' keys from examples
    for example in examples:
        example.pop('vision', None)
        example.pop('cross', None)

    # Create model_args by concatenating tensors and copying other attributes
    model_args = {key: concatenate_tensors(None, key) 
                  if isinstance(examples[-1][key], torch.Tensor) else examples[-1][key] 
                  for key in examples[-1]
                  }
    
    # Merge img_args into model_args
    model_args.update(img_args)
    return model_args


from collections import defaultdict

def broadcast_auto(data_dict):
    type2list = defaultdict(list)
    other = []
    for k in data_dict:
        if type(data_dict[k]) is torch.Tensor:
            type2list[data_dict[k].dtype].append(k)
        else:
            other.append(k)
    new_data = {}
    for k in type2list:
        new_data.update(mpu.broadcast_data(type2list[k], data_dict, k))
    for k in other:
        new_data[k] = data_dict[k]
    return new_data

def get_batch(data_iterator, args, timers):
    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = broadcast_auto(data)
    for k in data_b:
        if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
            if args.fp16:
                data_b[k] = data_b[k].half()
            elif args.bf16:
                data_b[k] = data_b[k].bfloat16()
    return data_b

from torch.nn import CrossEntropyLoss
import numpy as np

from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy


def chat(model, tokenizer, tokens,
         max_length: int = 1800, num_beams=5, top_p=0.95, top_k=0, temperature=0.8, **kwargs):
    inputs = tokens.to(model.parameters().__next__().device)[0]
    seq = torch.cat(
        [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=0.4, top_k=1, end_tokens=[tokenizer.eos_token_id])
    # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id],
    #                               num_beams=num_beams, consider_end=True)
    get_func = llama2_text_processor_inference.get_func(None, None, image_rope_mask=kwargs['image_rope_mask'])
    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy,
        get_masks_and_position_ids=get_func,
        **kwargs
    )[0]  # drop memory

    return output

def rxnscribe_eval(pred, label):
    pred = eval(pred)
    tar = eval(label)
    def get_iou(bb1, bb2):
        bb1 = bb1[0]
        bb2 = bb2[0]
        bb_intersect = [max(bb1[0], bb2[0]), max(bb1[1], bb2[1]), min(bb1[2], bb2[2]), min(bb1[3], bb2[3])]
        def get_area(bbox):
            return max(0, bbox[2] - bbox[0] + 1) * max(0, bbox[3] - bbox[1] + 1)
        inter_area = get_area(bb_intersect)
        return inter_area / (get_area(bb1) + get_area(bb2) - inter_area)

    def match_bbox(bbox, rxn, keys):
        max_iou = 0
        match_bb = None
        for k in keys:
            for bb in rxn[k]:
                iou = get_iou(bb, bbox)
                if iou > max_iou:
                    max_iou = iou
                    match_bb = bb
        return max_iou, match_bb
    
    def get_hit(rxn, comp_set, keys):
        for rxn1 in comp_set:
            flag = True
            for k, v in rxn.items():
                for bbox in v:
                    iou, bb = match_bbox(bbox, rxn1, keys[k])
                    if iou < 0.5:
                        flag = False
            for k, v in rxn1.items():
                for bbox in v:
                    iou, bb = match_bbox(bbox, rxn, keys[k])
                    if iou < 0.5:
                        flag = False
            if flag:
                return 1
        return 0

    soft_matched = {'prec': 0, 'rec': 0}
    hard_matched = {'prec': 0, 'rec': 0}
    soft_keys = {'reactants': ['reactants', 'conditions'], 'conditions': ['reactants', 'conditions'], 'products': ['products']}
    hard_keys = {'reactants': ['reactants'], 'conditions': ['conditions'], 'products': ['products']}
    for rxn in pred:
        soft_matched['prec'] += get_hit(rxn, tar, soft_keys)
        hard_matched['prec'] += get_hit(rxn, tar, hard_keys)
    for rxn in tar:
        soft_matched['rec'] += get_hit(rxn, pred, soft_keys)
        hard_matched['rec'] += get_hit(rxn, pred, hard_keys)
    prec_total = len(pred)
    rec_total = len(tar)
    return soft_matched, hard_matched, prec_total, rec_total


def forward_step_eval(data_iterator, model, args, timers):
    def compute_metrics(eval_preds):
        preds, labels, device = eval_preds
        preds = preds.unsqueeze(0)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "soft_pred_hits": 0,
            "soft_gold_hits": 0,
            "hard_pred_hits": 0,
            "hard_gold_hits": 0,
            "pred_total": 0,
            "gold_total": 0
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            if args.rank == 0:
                print('qid', qid, '\npred', pred, '\nlabel', label, flush=True)
                """
                print_rank0('----------------------')
                print_rank0(pred)
                print_rank0('-------- label:')
                print_rank0(label)
                print_rank0('----------------------')
                """
            try:
                soft_matched, hard_matched, prec_total, rec_total = rxnscribe_eval(pred, label)
            except:
                soft_matched, hard_matched = {'prec': 0, 'rec': 0}, {'prec': 0, 'rec': 0}
                rec_total = len(eval(label))
                prec_total = rec_total
            score_dict['soft_pred_hits'] += soft_matched['prec']
            score_dict['soft_gold_hits'] += soft_matched['rec']
            score_dict['hard_pred_hits'] += hard_matched['prec']
            score_dict['hard_gold_hits'] += hard_matched['rec']
            score_dict['pred_total'] += prec_total
            score_dict['gold_total'] += rec_total
            if args.rank == 0:
                print(score_dict, flush=True)

        return score_dict

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    context_len = int(data_b['context_length'][0])
    tokens = data_b['input_ids'][:, :context_len]
    data_b['vision_expert_mask'] = data_b['vision_expert_mask'][:, :context_len]
    data_b['image_embed_mask'] = data_b['image_embed_mask'][:, :context_len]
    data_b['image_rope_mask'] = data_b['image_rope_mask'][:, :context_len]

    data_b.pop('input_ids')
    data_b.pop('attention_mask')
    data_b.pop('position_ids')
    labels = data_b.pop('labels')
    qid = data_b.pop('question_id')

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    outputs = chat(model, tokenizer, tokens, **data_b)[0][context_len:]
    # print(outputs)
    model.del_mixin('auto-regressive')

    return torch.tensor(0, device=outputs.device), {k: torch.tensor(v, device=outputs.device) for k, v in
                                                    compute_metrics(
                                                        (outputs.cpu(), labels.cpu(), outputs.device)).items()}


def handle_metrics(metrics_total):
    metrics_total = {key: sum(value.split(1,0)) for key, value in metrics_total.items()}
    result = {'soft_precision': metrics_total['soft_pred_hits'] / metrics_total['pred_total'] , 
                'soft_recall': metrics_total['soft_gold_hits'] / metrics_total['gold_total'] , 
                'hard_precision': metrics_total['hard_pred_hits'] / metrics_total['pred_total'] , 
                'hard_recall': metrics_total['hard_gold_hits'] / metrics_total['gold_total'] , 
                }
    result['soft_f1'] = 2 * result['soft_precision'] * result['soft_recall'] / (result['soft_precision'] + result['soft_recall'] + 0.00001)
    result['hard_f1'] = 2 * result['hard_precision'] * result['hard_recall'] / (result['hard_precision'] + result['hard_recall'] + 0.00001)
    return result

from torch.nn import CrossEntropyLoss
def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
    labels = data_b.pop('labels') # shape: batch size x seq length
    """
    _labels = labels.cpu()
    _labels = np.where(_labels != -100, _labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(_labels, skip_special_tokens=True)
    print_rank0('-------')
    print_rank0(_labels[0])
    print_rank0(decoded_labels)
    """
    timers('batch generator').stop()
    logits = model(**data_b)[0]
    lm_logits = logits.to(torch.float32)
    # Shift so that tokens < n predict n
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.to(torch.float32)

    return loss, {'loss': loss}

from utils.utils import ItemDataset
def create_dataset_function(image_processor, text_processor, cross_image_processor, path, args):
    dataset = ItemDataset(image_processor, text_processor, args, path, cross_image_processor=cross_image_processor)
    return dataset

from sat.model.finetune.lora2 import LoraMixin
from sat.model.finetune.prompt_tuning import PTuningV2Mixin

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--max_length', type=int)
    py_parser.add_argument('--ignore_pad_token_for_loss', action='store_false')
    py_parser.add_argument("--version", type=str, default="chat", choices=["chat", "vqa"], help='version to interact with')
    py_parser.add_argument("--from_pretrained", type=str, default="cogagent-chat", help='pretrained ckpt')
    py_parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
    py_parser = FineTuneTrainCogAgentModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    #if args.use_qlora:
    #    args.device = 'cpu'

    model, args = FineTuneTrainCogAgentModel.from_pretrained(args.from_pretrained, args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    if args.use_ptuning: # TODO: wait for SAT updating
        model.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))

    if args.use_lora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range), reinit=True)
        model.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, layer_range=args.layer_range), reinit=True)
    elif args.use_qlora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
        model.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
        
    if args.use_qlora and torch.cuda.is_available():
        model = model.to('cuda')
    from utils.utils import llama2_tokenizer
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(args.cross_image_pix)
    text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

    model = training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=partial(create_dataset_function, image_processor, text_processor, cross_image_processor), collate_fn=partial(data_collator, cross_image_processor=cross_image_processor), forward_step_eval=forward_step_eval, handle_metrics_function=handle_metrics)
    if args.use_lora:
        model.get_mixin("lora").merge_lora()
        model.get_mixin("eva").vit_model.get_mixin("lora").merge_lora()
        args.use_lora = False
        args.save = os.path.join(args.save, "merged_lora_cogagent")
        from sat.training.model_io import save_checkpoint
        save_checkpoint(1, model, None, None, args)
