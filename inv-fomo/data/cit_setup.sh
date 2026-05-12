#!/bin/bash

datasets=("Medical" "Aquatic" "Aerial" "Game" "Surgical")
# datasets=( "Surgical")

for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    python "../models/causal_intervention_transformation.py" --dataset "$dataset"


done

echo "All datasets finished."

