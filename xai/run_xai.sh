#!/bin/bash
#SBATCH --array 0-23%24
#SBATCH --job-name xai
#SBATCH --partition H100,H200,A100-80GB,L40S,A100-40GB
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 160GB
#SBATCH --gpu-bind none
#SBATCH --output /path/to/ta_logs/output_%A_%a.out  
#SBATCH --error /path/to/ta_logs/error_%A_%a.err    

# replace with your path to the log directory if desired. Also modify the respective SBATCH variables above
mkdir -p /path/to/log/dir

export NUM_GPUS=$SLURM_GPUS
export NUM_CPUS_PER_GPU=$SLURM_CPUS_PER_GPU
export HF_TOKEN="your_hf_token_here"  # Replace with your actual HF token

# define path to directory where to save the plots
export SAVE_DIR="path/to/save/dir"

CONFIG_FILE="xai_configs.yaml.j2"
ARGS=$(python ../configs/process_yaml.py $CONFIG_FILE $SLURM_ARRAY_TASK_ID)
echo $ARGS

CONTAINER_WORKDIR=`pwd`
HOST_WORKDIR=`pwd`
HOST_CACHEDIR=/netscratch/$USER/.cache_slurm
HOST_CONDA_ENVS_DIR=/netscratch/$USER/miniconda3
CONTAINER_CONDA_ENVS_DIR=/opt/conda
CONTAINER_MOUNTS=/netscratch:/netscratch,/ds:/ds:ro,"$HOST_WORKDIR":"$CONTAINER_WORKDIR","$HOST_CONDA_ENVS_DIR":"$CONTAINER_CONDA_ENVS_DIR","$HOST_CACHEDIR":"/home/$USER/.cache","$HOST_CACHEDIR":/root/.cache,"$HOST_CACHEDIR":/home/root/.cache

export CONDA_ENV="adapters"

#srun -k \
#  --container-image=/enroot/nvcr.io_nvidia_pytorch_24.09-py3.sqsh \
#  --container-workdir="`pwd`" \
#  --container-mounts=$CONTAINER_MOUNTS \
#  ./xai.sh $ARGS
