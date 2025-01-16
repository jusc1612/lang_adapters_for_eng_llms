#!/bin/bash

ARGS="$@"
shift $#

echo "activate conda environment: $CONDA_ENV"
source /opt/conda/bin/activate 
conda activate $CONDA_ENV

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    scripts/eval.py $ARGS
