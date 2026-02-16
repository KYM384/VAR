#!/bin/bash

#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=2:00:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j


module load gcc/8.3.1
module load cuda/12.2
module load singularity/3.9.5


singularity exec --nv --bind ${PWD}:/workspace --bind /work/gb20/share/imagenet-1k/:/data .docker/var.sif bash -c \
    "cd /workspace && \
    export WANDB_API_KEY=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 && \
    torchrun --nproc_per_node=8 --nnodes=1 train.py \
        --depth=16 --bs=768 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --pn 16 --patch_size 16 \
        --vfast=1 \
        --data_path='/data'"
