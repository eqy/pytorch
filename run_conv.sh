#!/bin/bash
CUDA_LAUNCH_BLOCKING=0 torchrun --nproc-per-node $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) --nnodes 1 fuzz_conv.py
