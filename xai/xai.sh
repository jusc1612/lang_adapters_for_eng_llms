#!/bin/bash

ALL_ARGS="$@"
shift $#

echo "activate conda environment: $CONDA_ENV"
source /opt/conda/bin/activate 
conda activate $CONDA_ENV

echo $ALL_ARGS

TARGET_VAR=$(echo "$ALL_ARGS" | grep -oP '(?<=--xai_method )[^ ]+')

if [[ "$TARGET_VAR" == "logit_lens" ]]; then
  SCRIPT_NAME="logit_lens.py"
else
  SCRIPT_NAME="pca.py"
fi

echo $SCRIPT_NAME

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_GPUS \
    $SCRIPT_NAME $ALL_ARGS