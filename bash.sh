#!/bin/bash
# A simple script for training the model on all the dataset

data_dir_list=  "./data/skill_inflow_outflow/" "./data/skill_inflow_outflow/it" "./data/skill_inflow_outflow/fin" "./data/skill_inflow_outflow/con"
model = "static" "hier" "adaptive" "multi-view" "semantic"
for dir in $data_dir_list; do
    for m in model; do
        python train.py --n "${m} ${dir:28}" --config config.json --data_dir $dir --m $m --multi-view &
    done
done