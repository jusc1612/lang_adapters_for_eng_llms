#!/bin/bash
#SBATCH --array=0-35%36
#SBATCH --time=1-0
#SBATCH --job-name=eval_icl
#SBATCH --partition=H100,H200,L40S,A100-80GB,A100-40GB
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=120GB
#SBATCH --gpu-bind=none
#SBATCH --output /path/to/ta_logs/output_%A_%a.out  
#SBATCH --error /path/to/ta_logs/error_%A_%a.err    

# replace with your path to the log directory if desired. Also modify the respective SBATCH variables above
mkdir -p /path/to/log/dir

export HF_TOKEN="your_hf_token_here"  # Replace with your actual HF token

# define cache dir
export CACHE_DIR="/path/to/your/cache/dir"

# replace with directoy where all LAs are saved
export ADAPTER_PATH="/path/to/your/trained/adapters"

# replace with directory to save all evaluation scores
export EVAL_DIR="/path/to/eval/dir"

# ICL configs for MLQA and SIB-200
CONFIG_FILE="configs/icl_eval_configs.yaml.j2"

# ICL configs for ablations
#CONFIG_FILE="configs/icl_eval_configs_ablations.yaml.j2"

ARGS=$(python configs/process_yaml.py $CONFIG_FILE $SLURM_ARRAY_TASK_ID)

echo $ARGS

TARGET_VAR=$(echo "$ARGS" | grep -oP '(?<=--adapter_method )[^ ]+')

if [[ "$TARGET_VAR" == "lora" || "$TARGET_VAR" == "pt" ]]; then
  export CONDA_ENV="peft"
else
  export CONDA_ENV="adapters"
fi

export NUM_GPUS=$SLURM_GPUS

CONTAINER_WORKDIR=`pwd`
HOST_WORKDIR=`pwd`
HOST_CACHEDIR=/netscratch/$USER/.cache_slurm
HOST_CONDA_ENVS_DIR=/netscratch/$USER/miniconda3
CONTAINER_CONDA_ENVS_DIR=/opt/conda
CONTAINER_MOUNTS=/netscratch:/netscratch,/ds:/ds:ro,"$HOST_WORKDIR":"$CONTAINER_WORKDIR","$HOST_CONDA_ENVS_DIR":"$CONTAINER_CONDA_ENVS_DIR","$HOST_CACHEDIR":"/home/$USER/.cache","$HOST_CACHEDIR":/root/.cache,"$HOST_CACHEDIR":/home/root/.cache

srun -k \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.09-py3.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=$CONTAINER_MOUNTS \
  scripts/eval_all.sh $ARGS
