#! /bin/bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CXX=g++

NUM_GPUS_PER_WORKER=2
MP_SIZE=2

eval "$(conda shell.bash hook)"
conda activate vlm

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="cogagent-chat"
VERSION="chat"
MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_length 400 \
    --lora_rank 40 \
    --use_lora \
    --local_tokenizer lmsys/vicuna-7b-v1.5 \
    --version $VERSION"
# TIPS: max_length include low-resolution image sequence (which has 256 tokens) 

OPTIONS_SAT="SAT_HOME=/scratch/wang7776/.sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

train_data="./archive_split/train"
valid_data="./archive_split/valid"

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 500 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${valid_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --vit_checkpoint_activations \
       --save-interval 200 \
       --eval-interval 200 \
       --save "/scratch/wang7776/test_finetune/checkpoints" \
       --eval-iters 10 \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config test_config_bf16.json \
       --skip-init \
       --seed 2023
"

              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_cogagent_demo.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
