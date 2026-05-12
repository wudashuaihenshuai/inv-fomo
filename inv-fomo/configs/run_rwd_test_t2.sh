#!/bin/bash

set -e
export TMPDIR=/home/data/wcy123/tmp/
mkdir -p $TMPDIR  ##############################################
cur_fname="$(basename $0 .sh)"
script_name=$(basename $0)

# Initialize TCP port and counter
TCP_INIT=29500
counter=0

# Declare arrays for different configurations
#declare -a DATASETS=(  )
declare -a DATASETS=( "Aquatic" "Surgical" "Medical"  "Game" "Aerial")
# declare -a DATASETS=("Aerial")
#declare -a MODELS=("/home/data/wcy123/FOMO-main/models/owlvit-base-patch16" "/home/data/wcy123/FOMO-main/models/owlvit-large-patch14")
declare -a MODELS=("/root/autodl-tmp/inv-fomo/models/owlvit-large-patch14")
# declare -a MODELS=("/root/autodl-tmp/inv-fomo/models/owlvit-base-patch16")

# declare -a NUM_SHOTS=(1 10 100)
declare -a NUM_SHOTS=(100)



# Declare associative array for CUR_INTRODUCED_CLS per dataset
declare -A PREV_INTRODUCED_CLS
PREV_INTRODUCED_CLS["Aerial"]=10
PREV_INTRODUCED_CLS["Surgical"]=6
PREV_INTRODUCED_CLS["Medical"]=6
PREV_INTRODUCED_CLS["Aquatic"]=4
PREV_INTRODUCED_CLS["Game"]=30


declare -A CUR_INTRODUCED_CLS
CUR_INTRODUCED_CLS["Aerial"]=10
CUR_INTRODUCED_CLS["Surgical"]=7
CUR_INTRODUCED_CLS["Medical"]=6
CUR_INTRODUCED_CLS["Aquatic"]=3
CUR_INTRODUCED_CLS["Game"]=29

declare -A BATCH_SIZEs
BATCH_SIZEs["google/owlvit-base-patch16"]=10
BATCH_SIZEs["google/owlvit-large-patch14"]=2

declare -A IMAGE_SIZEs
IMAGE_SIZEs["google/owlvit-base-patch16"]=768
IMAGE_SIZEs["google/owlvit-large-patch14"]=840

# Loop through each configuration
for num_shot in "${NUM_SHOTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    cur_cls=${CUR_INTRODUCED_CLS[$dataset]}
    prev_cls=${PREV_INTRODUCED_CLS[$dataset]}

    for model in "${MODELS[@]}"; do
      BATCH_SIZE=${BATCH_SIZEs[$model]}
      IMAGE_SIZE=${IMAGE_SIZEs[$model]}
      tcp=$((TCP_INIT + counter))
      out_dir="./out/${dataset}/few_shot_${num_shot}"


    # Construct the command to run
        # cmd="CUDA_VISIBLE_DEVICES=0 python main_copy_.py --model_name \"$model\" --num_few_shot $num_shot --batch_size 2 \
        # --PREV_INTRODUCED_CLS $prev_cls --CUR_INTRODUCED_CLS $cur_cls --TCP $tcp --dataset $dataset \
        # --image_conditioned --image_resize 840 --classnames_file 'classnames.txt'\
        # --output_dir "${out_dir}"  --prev_output_file "results_t2.csv" --output_file "results_t2.csv" \
        # --att_refinement --att_adapt --att_selection --use_attributes"
        cmd="CUDA_VISIBLE_DEVICES=0 python main_copy_.py --model_name \"$model\" --num_few_shot $num_shot --batch_size 2 \
        --PREV_INTRODUCED_CLS $prev_cls --CUR_INTRODUCED_CLS $cur_cls --TCP $tcp --dataset $dataset \
        --image_conditioned --image_resize 840 --classnames_file 'classnames.txt'\
        --output_dir "${out_dir}"  --prev_output_file "results_t2_sinv.csv" --output_file "results_t2_sinv.csv" \
        --att_refinement --att_adapt --att_selection --use_attributes"

        echo "Constructed Command: $cmd"
        eval "$cmd"

        counter=$((counter + 1))
      done
    done
done
