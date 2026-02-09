#!/bin/bash

#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=15:00
#PJM -g gb20
#PJM -j

module load gcc/8.3.1
module load cuda/12.6
module load singularity/3.9.5

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

pip3 install spython
spython recipe .docker/Dockerfile .docker/var.def
deactivate
rm -rf .venv

rm .docker/var.sif

module load singularity/3.9.5
singularity build --fakeroot .docker/var.sif .docker/var.def

