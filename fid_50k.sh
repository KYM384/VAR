#!/bin/bash

#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=20:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j


module load gcc/8.3.1
module load cuda/12.2
module load singularity/3.9.5


# Standard FID-50k: 50,000 generated images (exactly 50 per class) against
# reference statistics over the FULL ImageNet training set. Generated images are
# never written to disk; the reference statistics are cached to
# fid_stats/imagenet256_train_all.npz after the first run.

singularity exec --nv --bind ${PWD}:/workspace --bind /work/gb20/share/imagenet-1k/:/data .docker/var.sif bash -c \
    "cd /workspace && \
    torchrun --nproc_per_node=8 --nnodes=1 compute_fid.py \
        --num-gen 50000 --gen-batch 16 \
        --num-steps 11 --dp timestep_metrics/metrics.csv --dp-metric l2_distance \
        --ref-split train --num-ref -1 --ref-batch 64 --ref-workers 8"
