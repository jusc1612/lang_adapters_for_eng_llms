#!/bin/bash
#SBATCH --array=0-35%36
#SBATCH --job-name=ta_training
#SBATCH --partition=H100,H200,A100-80GB
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=120GB
#SBATCH --gpu-bind=none
#SBATCH --output /path/to/ta_logs/output_%A_%a.out  
#SBATCH --error /path/to/ta_logs/error_%A_%a.err    

# replace with your path to the log directory if desired. Also modify the respective SBATCH variables above
mkdir -p /path/to/log/dir

# set cache dir and output dir where to save TAs 
export CACHE_DIR="/path/to/your/cache/dir"
export OUTPUT_DIR="/path/to/your/output/dir"

# modify if needed
declare -a seeds=(42 43 44 45 46)
 
export HF_TOKEN="your_hf_token_here"  # Replace with your actual HF token

# trains TAs for MLQA and SIB-200 dataset with default configs
CONFIG_FILE_NAME="configs/ta_training_configs.yaml.j2"

# trains TAs for SIB-200 with modified configs (reduction factor=32, dropout=0.1)
#CONFIG_FILE_NAME="configs/sib200-drop_ta_training_configs.yaml.j2"

ARGS=$(python configs/process_yaml.py $CONFIG_FILE_NAME $SLURM_ARRAY_TASK_ID)

echo $ARGS

TA_NAME=$(echo "$ARGS" | grep -oP '(?<=--ta_name )[^ ]+')

if [[ "$TA_NAME" =~ "lora" ]]; then
  export CONDA_ENV="peft"
else
  export CONDA_ENV="adapters"
fi

echo $CONDA_ENV
export NUM_GPUS=$SLURM_GPUS

CONTAINER_WORKDIR=`pwd`
HOST_WORKDIR=`pwd`
HOST_CACHEDIR=/netscratch/$USER/.cache_slurm
HOST_CONDA_ENVS_DIR=/netscratch/$USER/miniconda3
CONTAINER_CONDA_ENVS_DIR=/opt/conda
CONTAINER_MOUNTS=/netscratch:/netscratch,/ds:/ds:ro,"$HOST_WORKDIR":"$CONTAINER_WORKDIR","$HOST_CONDA_ENVS_DIR":"$CONTAINER_CONDA_ENVS_DIR","$HOST_CACHEDIR":"/home/$USER/.cache","$HOST_CACHEDIR":/root/.cache,"$HOST_CACHEDIR":/home/root/.cache

# optional: define wandb arguments 
export WANDB_API_KEY="your_wandb_key_here"
export WANDB_PROJECT="name_of_wandb_project"
export WANDB_DIR="path/to/wandb_logs"

for seed in "${seeds[@]}"; do
    echo "Running config $SLURM_ARRAY_TASK_ID with seed $seed"

    srun -k \
      --container-image=/enroot/nvcr.io_nvidia_pytorch_24.09-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=$CONTAINER_MOUNTS \
      scripts/ta_training.sh $ARGS --seed $seed
done
  
