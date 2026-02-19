#!/bin/bash

#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=5:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j


module load gcc/8.3.1
module load cuda/12.2
module load singularity/3.9.5


singularity exec --nv --bind ${PWD}:/workspace .docker/var.sif bash -c \
    "cd /workspace && python compute_fid.py"
