import os
import logging
import random
import logging
from io import BytesIO
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from sat.helpers import print_rank0
import glob
import json
import torch

def find_all_files(path, suffix=".jpg"):
    target_files = []
    for cur_dir, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith(suffix):
                target_files.append(os.path.join(cur_dir, f))
    print_rank0(f'find {len(target_files)} files...')
    return target_files

class ItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
        super().__init__()
        self.composite = 'train' in data_dirs  # hack
        self.data_dir = data_dirs
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
    
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict
    
    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)
    
    def load_data(self, data_dir):
        fname = glob.glob(os.path.join(data_dir, '*.json'))[0]
        with open(fname, "r") as f:
            all_files = json.load(f)['images']
        for sample in all_files:
            self.create_gt(sample)
        print_rank0(f"find {len(all_files)} samples in all...")
        return all_files

    def create_gt(self, data):
        def convert_bb(bbox):
            h, w = data['height'], data['width']
            temp = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
            temp = [int(t * 1000) for t in temp]
            return [temp[0], temp[1], temp[2] + temp[0], temp[3] + temp[1]]
        bboxes = []
        for bbox in data['bboxes']:
            if bbox['category_id'] == 1:
                bboxes.append(convert_bb(bbox['bbox']))
            else:
                bboxes.append(None)
        result = []
        for rxn in data['reactions']:
            curr = {}
            for k, v in rxn.items():
                curr[k] = [[bboxes[i]] for i in v if bboxes[i] is not None]
            result.append(curr)
        data['label'] = result
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # img
        try:
            img = Image.open(os.path.join(self.data_dir, data['file_name'])).convert('RGB')
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        if self.composite and len(str(data['label'])) < 1000 and random.random() < 0.7:
            # print_rank0('old')
            # print_rank0(data['label'])
            while True:
                data1 = self.data[int(random.random() * len(self.data))]
                if len(str(data1['label'])) < 1000:
                    break
            try:
                img1 = Image.open(os.path.join(self.data_dir, data1['file_name'])).convert('RGB')
            except Exception as e:
                print_rank0(e, level=logging.WARNING)
                return {}
            w, h = img.size
            w1, h1 = img1.size
            if (w + w1) * max(h, h1) > max(w, w1) * (h + h1):
                new_img = Image.new('RGB', (w + w1, max(h, h1)))
                if random.random() < 0.5:
                    offsets = ((0, (max(h, h1) - h) // 2), (w, (max(h, h1) - h1) // 2))
                else:
                    offsets = ((w1, (max(h, h1) - h) // 2), (0, (max(h, h1) - h1) // 2))
            else:
                new_img = Image.new('RGB', (max(w, w1), h + h1))
                if random.random() < 0.5:
                    offsets = (((max(w, w1) - w) // 2, 0), ((max(w, w1) - w1) // 2, h))
                else:
                    offsets = (((max(w, w1) - w) // 2, h1), ((max(w, w1) - w1) // 2, 0))
            new_img.paste(img, offsets[0])
            new_img.paste(img1, offsets[1])
            new_data = {'height': new_img.size[1], 'width': new_img.size[0], 'reactions': [], 'bboxes': []}
            for rxn in data['reactions']:
                new_data['reactions'].append(rxn)
            for bb in data['bboxes']:
                idt = bb['category_id']
                bb = bb['bbox']
                new_data['bboxes'].append({'bbox': [bb[0] + offsets[0][0], bb[1] + offsets[0][1], bb[2], bb[3]], 
                                           'category_id': idt})
            
            for rxn in data1['reactions']:
                new_rxn = {}
                for k, v in rxn.items():
                    new_rxn[k] = [i + len(new_data['bboxes']) for i in v]  # offset new indices by num original bboxes
                new_data['reactions'].append(new_rxn)
            for bb in data1['bboxes']:
                bb = bb['bbox']
                new_data['bboxes'].append({'bbox': [bb[0] + offsets[1][0], bb[1] + offsets[1][1], bb[2], bb[3]], 
                                           'category_id': idt})

            self.create_gt(new_data)
            img = new_img
            data = new_data
            """
            print_rank0(f'new {index}')
            print_rank0(data['label'])
            img1 = ImageDraw.Draw(img)
            for bb in data['bboxes']:
                bbox = bb['bbox']
                bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                img1.rectangle(bbox, outline='red')
            img.save(f"/scratch/wang7776/test_finetune/{index}.jpg")
            """

        img_dict = self.process_img(img)
        # text
        label = data['label']
        #random.shuffle(label)
        label = str(label)
        uni_key = label
        text_dict = self.process_text(label, 
            "Describe all reactions in the form [{'reactants': [[[x1,y1,x2,y2]], ... ], 'conditions': [[[x1,y1,x2,y2]], ... ]}, 'products': [[[x1,y1,x2,y2]], ... ]}, ... ]")
        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        return ret