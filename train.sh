
#!/bin/bash

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 \
  train.py --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --pn 32 --patch_size 2 \
    --data_path='cifar_images'
