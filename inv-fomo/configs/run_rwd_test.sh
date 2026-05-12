##!/bin/bash
#
#set -e
#cur_fname="$(basename $0 .sh)"
#script_name=$(basename $0)
#
## Cluster parameters
#partition=""
#account=""
#
## Initialize TCP port and counter
#TCP_INIT=28500
#counter=0
#
## Declare arrays for different configurations
#declare -a DATASETS=("Aerial" "Surgical" "Medical" "Aquatic" "Game")
#declare -a MODELS=("google/owlvit-base-patch16" "google/owlvit-large-patch14")
#declare -a NUM_SHOTS=(1 10 100)
#
## Declare associative array for CUR_INTRODUCED_CLS per dataset
#declare -A CUR_INTRODUCED_CLS
#CUR_INTRODUCED_CLS["Aerial"]=10
#CUR_INTRODUCED_CLS["Surgical"]=6
#CUR_INTRODUCED_CLS["Medical"]=6
#CUR_INTRODUCED_CLS["Aquatic"]=4
#CUR_INTRODUCED_CLS["Game"]=30
#
#declare -A BATCH_SIZEs
#BATCH_SIZEs["google/owlvit-base-patch16"]=10
#BATCH_SIZEs["google/owlvit-large-patch14"]=5
#
#declare -A IMAGE_SIZEs
#IMAGE_SIZEs["google/owlvit-base-patch16"]=768
#IMAGE_SIZEs["google/owlvit-large-patch14"]=840
#
## Loop through each configuration
#for num_shot in "${NUM_SHOTS[@]}"; do
#  for dataset in "${DATASETS[@]}"; do
#    cur_cls=${CUR_INTRODUCED_CLS[$dataset]}
#    for model in "${MODELS[@]}"; do
#      BATCH_SIZE=${BATCH_SIZEs[$model]}
#      IMAGE_SIZE=${IMAGE_SIZEs[$model]}
#      tcp=$((TCP_INIT + counter))
#
#      # Construct the command to run --output_file res/$dataset-$num_shot-$model.csv \
#      cmd="python main.py --model_name \"$model\" --num_few_shot $num_shot --batch_size $BATCH_SIZE \
#      --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS $cur_cls --TCP $tcp --dataset $dataset \
#      --image_conditioned --image_resize $IMAGE_SIZE \
#      --att_refinement --att_adapt --att_selection --use_attributes"
#
#      echo "Constructed Command:\n$cmd"
#
#      # Uncomment below to submit the job
#      sbatch <<< \
#"#!/bin/bash
##SBATCH --job-name=${dataset}-${num_shot}-${cur_fname}
##SBATCH --output=slurm_logs/${dataset}/-${num_shot}-${cur_fname}-%j-out.txt
##SBATCH --error=slurm_logs/${dataset}/-${num_shot}-${cur_fname}-%j-err.txt
##SBATCH --mem=32gb
##SBATCH -c 2
##SBATCH --gres=gpu:a6000
##SBATCH -p $partition
##SBATCH -A $account
##SBATCH --time=48:00:00
##SBATCH --ntasks=1
#echo \"$cmd\"
## Uncomment below to actually run the command
#eval \"$cmd\"
#"
#
#      counter=$((counter + 1))
#    done
#  done
#done
#!/bin/bash
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
# declare -a DATASETS=( "Aquatic" "Surgical" "Medical"  "Game" "Aerial")
# declare -a DATASETS=("Surgical" "Medical"  "Game" "Aerial")
declare -a DATASETS=("Surgical")
#declare -a MODELS=("/home/data/wcy123/FOMO-main/models/owlvit-base-patch16" "/home/data/wcy123/FOMO-main/models/owlvit-large-patch14")
declare -a MODELS=("/root/autodl-tmp/inv-fomo/models/owlvit-large-patch14")
# declare -a MODELS=("/root/autodl-tmp/inv-fomo/models/owlvit-base-patch16")
# declare -a NUM_SHOTS=(1 10 100)
declare -a NUM_SHOTS=( 1 )




# Declare associative array for CUR_INTRODUCED_CLS per dataset
declare -A CUR_INTRODUCED_CLS
CUR_INTRODUCED_CLS["Aerial"]=10
CUR_INTRODUCED_CLS["Surgical"]=6
CUR_INTRODUCED_CLS["Medical"]=6
CUR_INTRODUCED_CLS["Aquatic"]=4
CUR_INTRODUCED_CLS["Game"]=30

declare -A BATCH_SIZEs
BATCH_SIZEs["/home/data/wcy123/FOMO-main/models/owlvit-base-patch16"]=10
BATCH_SIZEs["/home/data/wcy123/FOMO-main/models/owlvit-large-patch14"]=5

declare -A IMAGE_SIZEs
IMAGE_SIZEs["/home/data/wcy123/FOMO-main/models/owlvit-base-patch16"]=768
IMAGE_SIZEs["/home/data/wcy123/FOMO-main/models/owlvit-large-patch14"]=840

# Loop through each configuration
for num_shot in "${NUM_SHOTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    cur_cls=${CUR_INTRODUCED_CLS[$dataset]}
    for model in "${MODELS[@]}"; do
      BATCH_SIZE=${BATCH_SIZEs[$model]}
      IMAGE_SIZE=${IMAGE_SIZEs[$model]}
      tcp=$((TCP_INIT + counter))
      # Create directory structure for saving files
      loss_dir="./loss/${dataset}/few_shot_${num_shot}"
      out_dir="./out/${dataset}/few_shot_${num_shot}"

      mkdir -p "$loss_dir"
      mkdir -p "$out_dir"
      # Construct the command to run with torchrun for 2 GPUs
#      cmd="CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=18500 main.py --model_name \"$model\" --num_few_shot $num_shot \
#      --batch_size $BATCH_SIZE --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS $cur_cls \
#      --TCP $tcp --dataset $dataset --image_conditioned --image_resize $IMAGE_SIZE \
#      --att_refinement --att_adapt --att_selection --use_attributes"
      # cmd="CUDA_VISIBLE_DEVICES=0 python main_copy_.py --model_name \"$model\"  --num_few_shot $num_shot \
      # --batch_size 5 --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS $cur_cls \
      # --TCP $tcp --dataset $dataset --image_conditioned --image_resize 840   \
      # --output_dir "${out_dir}"  --prev_output_file "results.csv" --output_file "results.csv" \
      # --att_refinement --att_adapt --att_selection --use_attributes "
      cmd="CUDA_VISIBLE_DEVICES=0 python main_copy_.py --model_name \"$model\"  --num_few_shot $num_shot \
      --batch_size 5 --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS $cur_cls \
      --TCP $tcp --dataset $dataset --image_conditioned --image_resize 840  \
      --output_dir "${out_dir}"  --prev_output_file "results_qves.csv" --output_file "results_qves.csv" \
      --att_refinement --att_adapt --att_selection --use_attributes "


      echo -e "Running Command:\n$cmd"

      # Directly run the command locally
      eval "$cmd"

      counter=$((counter + 1))
    done
  done
done

