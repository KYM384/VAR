#!/bin/bash

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j


module load gcc/8.3.1
module load cuda/12.2
module load singularity/3.9.5


# WANDB_API_KEY must be exported in the submitting shell (e.g. `export WANDB_API_KEY=...`)
# BEFORE running this script. Do NOT hardcode the key here — it gets committed to git history.
# singularity forwards host env vars by default, so the value below is read from the host env.
singularity exec --nv --bind ${PWD}:/workspace --bind /work/gb20/share/imagenet-1k/:/data .docker/var.sif bash -c \
    "cd /workspace && \
    export WANDB_API_KEY=${WANDB_API_KEY:?set WANDB_API_KEY in your shell before submitting} && \
    torchrun --nproc_per_node=8 --nnodes=1 train.py \
        --depth=16 --bs=768 --ep=200 --fp16=2 --alng=1e-3 --wpe=0.1 --workers 8 --pn 16 --patch_size 16 \
        --vfast 3 --tfast 3 --afuse 0 \
        --data_path='/data'"
