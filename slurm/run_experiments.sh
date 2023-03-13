#!/bin/bash

# NOTE: This requires GNU getopt.
TEMP=$(getopt -o: --long dataset:,arch:,workers:,epochs:,batchsize:,lr:,momentum:,wd:,seed:,verbose:,nodes:,mainpath:,gpus:,folder:,stage:,transparent: \
              -n 'experiment_starter' -- "$@")

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

# Note the quotes around '$TEMP': they are essential!
eval set -- "$TEMP"

# NOTE: These default values might converge from the ones in parser.py
DATASET="imagenet"
ARCH="resnet50"
WORKERS=0
EPOCHS=90
BATCH_SIZE=32
LR=0.1
MOMENTUM=0.9
WD=0.0001
SEED=42
VERBOSE=true
NODES=1
MAIN_PATH="/pfs/work7/workspace/scratch/tz6121-shuffling/distributed_shuffling/resnet_imagenet/main.py"
GPUs_PER_NODE=8
FOLDER="experiments"
STAGE=true
TRANSPARENT=false
while true; do
  case "$1" in
    --dataset ) DATASET="$2"; shift 2 ;;
    --arch ) ARCH="$2"; shift 2 ;;
    --workers ) WORKERS="$2"; shift 2 ;;
    --epochs ) EPOCHS="$2"; shift 2 ;;
    --batchsize ) BATCH_SIZE="$2"; shift 2 ;;
    --lr ) LR="$2"; shift 2 ;;
    --momentum ) MOMENTUM="$2"; shift 2 ;;
    --wd ) WD="$2"; shift 2 ;;
    --seed ) SEED="$2"; shift 2 ;;
    --verbose ) VERBOSE="$2"; shift 2 ;;
    --nodes ) NODES="$2"; shift 2 ;;
    --mainpath ) MAIN_PATH="$2"; shift 2 ;;
    --gpus ) GPUs_PER_NODE="$2"; shift 2 ;;
    --folder ) FOLDER="$2"; shift 2 ;;
    --stage ) STAGE="$2"; shift 2 ;;
    --transparent ) TRANSPARENT="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

NTASKS=$(($NODES * $GPUs_PER_NODE))
EFFECTIVE_BATCH_SIZE=$(($BATCH_SIZE * $NTASKS))
JOB_NAME="${ARCH}_${DATASET}_${NTASKS}gpus"

# String variable
BASE_SCRIPT="#!/bin/bash

#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${JOB_NAME}.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_8
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
#SBATCH --gres=gpu:$GPUs_PER_NODE
#SBATCH --reservation=BA-Arbeit

module load compiler/gnu/10.2
module load devel/cuda/11.6
module load mpi/openmpi/4.0

source /pfs/work7/workspace/scratch/tz6121-shuffling/venvs/venv3.9/bin/activate


srun python -u $MAIN_PATH \
--output-name output \
--dataset-name $DATASET \
--arch $ARCH \
--epochs $EPOCHS \
--batch-size $BATCH_SIZE \
--momentum $MOMENTUM \
--wd $WD \
--lr $LR \
--print-freq 5 \
--seed $SEED \
--workers $WORKERS"

# If verbose, add --verbose to base script
if [ "$VERBOSE" = true ] ; then
  BASE_SCRIPT="${BASE_SCRIPT} --verbose"
fi

# If stage, add --stage to base script
if [ "$STAGE" = true ] ; then
  BASE_SCRIPT="${BASE_SCRIPT} --stage"
fi

# If transparent, add --transparent to base script
if [ "$TRANSPARENT" = true ] ; then
  BASE_SCRIPT="${BASE_SCRIPT} --transparent"
fi

if [ -d "$FOLDER" ]; then
  echo "$FOLDER folder already exists."
  exit 1
fi

PARAMETER_SUMMARY="seed: ${SEED}
global_batch_size: ${EFFECTIVE_BATCH_SIZE}
batchsize: ${BATCH_SIZE}
nodes: ${NODES}
lr: ${LR}
momentum: ${MOMENTUM}
wd: ${WD}
workers: ${WORKERS}
verbose: ${VERBOSE}
epochs: ${EPOCHS}
arch: ${ARCH}
dataset: ${DATASET}"

#shellcheck disable=SC2164
mkdir -p "$FOLDER/pre/step/local"
mkdir -p "$FOLDER/pre/step/noshuffle"
mkdir -p "$FOLDER/pre/seq/local"
mkdir -p "$FOLDER/pre/seq/noshuffle"
mkdir -p "$FOLDER/asis/step/local"
mkdir -p "$FOLDER/asis/step/noshuffle"
mkdir -p "$FOLDER/asis/seq/local"
mkdir -p "$FOLDER/asis/seq/noshuffle"
mkdir -p "$FOLDER/global"

echo "$PARAMETER_SUMMARY" > "$FOLDER/parameters.txt"

# Create sh files for each folder
PRE_SCRIPT="$BASE_SCRIPT --initial-shuffle"
echo "$PRE_SCRIPT --chunker \"step\" --shuffle" > "$FOLDER/pre/step/local/imres_pre_step_local.sh"
echo "$PRE_SCRIPT --chunker \"step\"" > "$FOLDER/pre/step/noshuffle/imres_pre_step_noshuffle.sh"

echo "$PRE_SCRIPT --chunker \"seq\" --shuffle" > "$FOLDER/pre/seq/local/imres_pre_seq_local.sh"
echo "$PRE_SCRIPT --chunker \"seq\"" > "$FOLDER/pre/seq/noshuffle/imres_pre_seq_noshuffle.sh"

ASIS_SCRIPT="$BASE_SCRIPT"
echo "$ASIS_SCRIPT --chunker \"step\" --shuffle" > "$FOLDER/asis/step/local/imres_asis_step_local.sh"
echo "$ASIS_SCRIPT --chunker \"step\"" > "$FOLDER/asis/step/noshuffle/imres_asis_step_noshuffle.sh"

echo "$ASIS_SCRIPT --chunker \"seq\" --shuffle" > "$FOLDER/asis/seq/local/imres_asis_seq_local.sh"
echo "$ASIS_SCRIPT --chunker \"seq\"" > "$FOLDER/asis/seq/noshuffle/imres_asis_seq_noshuffle.sh"

echo "$PRE_SCRIPT --chunker \"step\" --shuffle --global-shuffle" > "$FOLDER/global/imres_pre_step_global.sh"

# Go into all folders and submit jobs
cd "$FOLDER/global" || exit 1 && sbatch imres_pre_step_global.sh && cd ../
cd pre/step/local || exit 1 && sbatch imres_pre_step_local.sh && cd ../../../
cd pre/step/noshuffle || exit 1 && sbatch imres_pre_step_noshuffle.sh && cd ../../../

cd pre/seq/local || exit 1 && sbatch imres_pre_seq_local.sh && cd ../../../
cd pre/seq/noshuffle || exit 1 && sbatch imres_pre_seq_noshuffle.sh && cd ../../../

cd asis/step/local || exit 1 && sbatch imres_asis_step_local.sh && cd ../../../
cd asis/step/noshuffle || exit 1 && sbatch imres_asis_step_noshuffle.sh && cd ../../../

cd asis/seq/local || exit 1 && sbatch imres_asis_seq_local.sh && cd ../../../
cd asis/seq/noshuffle || exit 1 && sbatch imres_asis_seq_noshuffle.sh && cd ../../../
