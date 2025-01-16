#!/bin/bash

ALL_ARGS="$@"
shift $#

echo "activate conda environment: $CONDA_ENV"
source /opt/conda/bin/activate 
conda activate $CONDA_ENV

echo $ALL_ARGS

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    scripts/train_ta.py $ALL_ARGS
