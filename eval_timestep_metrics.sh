#!/bin/bash

#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=3:00:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j


module load gcc/8.3.1
module load cuda/12.2
module load singularity/3.9.5


# Per-timestep teacher-forcing metrics (accuracy / cross-entropy / codebook L2)
# over the full 50k validation set, sharded across 8 GPUs.
# Outputs: timestep_metrics/{metrics.csv, config.txt}. Render the plots
# afterwards with: python plot_timestep_metrics.py timestep_metrics/metrics.csv
#
# Knobs if the job runs long: raise --t-stride (fewer timesteps) or lower
# --num-val (fewer images). --t-list "0,50,100,150,199" evaluates specific steps.
singularity exec --nv --bind ${PWD}:/workspace --bind /work/gb20/share/imagenet-1k/:/data .docker/var.sif bash -c \
    "cd /workspace && \
    torchrun --nproc_per_node=8 --nnodes=1 eval_timestep_metrics.py \
        --num-val -1 --batch 32 --workers 8 \
        --t-stride 1 \
        --out-dir timestep_metrics"
