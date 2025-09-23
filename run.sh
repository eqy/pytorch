#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc-per-node $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) --nnodes 1 fuzz.py
