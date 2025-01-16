#!/bin/bash
#SBATCH --array 0%1
#SBATCH --job-name la_training
#SBATCH --partition H100,H200
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 200GB
#SBATCH --gpu-bind none
#SBATCH --output /path/to/la_logs/output_%A_%a.out  
#SBATCH --error /path/to/la_logs/error_%A_%a.err    

# replace with your path to the log directory if desired. Also modify the respective SBATCH variables above
mkdir -p /path/to/log/dir   

export NUM_GPUS=$SLURM_GPUS
export NUM_CPUS_PER_GPU=$SLURM_CPUS_PER_GPU
export HF_TOKEN="your_hf_token_here"  # Replace with your actual HF token

# single monolingual language adapters 
declare -a langs=("af")
declare -a lang_ratios=(1.0)

# all monolingual adapters
#declare -a langs=("de" "sv" "is" "fi" "nl" "es" "ca" "en" "da" "af" "hu" "pt" "gl")
#declare -a lang_ratios=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)

# multilingual adapters
#declare -a langs=("de sv da is" "de nl af" "es pt ca gl" "en fi hu")
#declare -a lang_ratios=("0.4 0.2 0.2 0.2" "0.4 0.3 0.3" "0.4 0.2 0.2 0.2" "0.4 0.3 0.3")

export LANG=${langs[$SLURM_ARRAY_TASK_ID]}
export LANG_RATIO=${lang_ratios[$SLURM_ARRAY_TASK_ID]}

# modify base LLM and model path + cache dir
MODEL_NAME="Llama-2-7b-hf"
export MODEL_PATH="meta-llama/$MODEL_NAME"
export CACHE_DIR="/path/to/your/cache/dir"

if [[ "$MODEL_NAME" == *.* ]]; then
    MODEL_NAME="${MODEL_NAME//./}"
else
    MODEL_NAME="$MODEL_NAME"
fi

# set method, dataset and name of adapter to be trained
export ADAPTER_METHOD="seq_bn" #LORA PROMPT_TUNING seq_bn seq_bn_inv
export DS="cc100" # check with the index of the config file further below
export SUFFIX="your_desired_suffix"
export ADAPTER_NAME="${MODEL_NAME}_${DS}_${LANG}_${ADAPTER_METHOD}_${SUFFIX}"
echo "Adapter Name:" $ADAPTER_NAME

# replace with your actual output directory where to save the adapter
export OUTPUT="/path/to/your/output/dir/${ADAPTER_NAME}"

# replace wiht your actual values
export STEPS=25000
export BATCH_SIZE=4
export NUM_TRAIN_SAMPLES=$((STEPS*BATCH_SIZE*NUM_GPUS))
export NUM_EVAL_SAMPLES=$(($NUM_TRAIN_SAMPLES / 100))
export LOG_EVAL_STEPS=$(($STEPS / 10))
export SAVE_STEPS=$(($STEPS / 5))

# set the path to the LA training file, should be a .txt file
#export LA_TRAIN_FILES_PATH="/ds/text/cc100/${LANG}.txt"
export LA_TRAIN_FILES_PATH="path/to/la/training/file"

# optional: set additional variables if required. Check respective config file for available variables.
export LAYERS_RANGES="24 31"

# set the index of the config file to use for LA training. Available indices: 0: CulturaX, 1: CC100, 2: Inv, 3: LoRA, 4: prompt tuning
CONFIG_INDEX=1
CONFIG_FILE="configs/la_training_configs.yaml.j2"
ARGS=$(python configs/process_yaml.py $CONFIG_FILE $CONFIG_INDEX)

echo $ARGS

CONTAINER_WORKDIR=`pwd`
HOST_WORKDIR=`pwd`
HOST_CACHEDIR=/netscratch/$USER/.cache_slurm
HOST_CONDA_ENVS_DIR=/netscratch/$USER/miniconda3
CONTAINER_CONDA_ENVS_DIR=/opt/conda
CONTAINER_MOUNTS=/netscratch:/netscratch,/ds:/ds:ro,"$HOST_WORKDIR":"$CONTAINER_WORKDIR","$HOST_CONDA_ENVS_DIR":"$CONTAINER_CONDA_ENVS_DIR","$HOST_CACHEDIR":"/home/$USER/.cache","$HOST_CACHEDIR":/root/.cache,"$HOST_CACHEDIR":/home/root/.cache

# defaults to currently activated conda env; modify if required
export CONDA_ENV=$CONDA_DEFAULT_ENV

# optional: define wandb arguments 
export WANDB_API_KEY="your_wandb_key_here"
export WANDB_PROJECT="name_of_wandb_project"
export WANDB_DIR="path/to/wandb_logs"

srun -k \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=$CONTAINER_MOUNTS \
  scripts/la_training.sh $ARGS
