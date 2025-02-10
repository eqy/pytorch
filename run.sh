#!/bin/bash
torchrun --nproc-per-node $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) --nnodes 1
