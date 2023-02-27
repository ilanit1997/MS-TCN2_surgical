##!/bin/bash
# multiple models can be added at weight type. Please enter them in order of most global to most local
python predict_eval_visualize.py --videos="P040_tissue2"
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_types="none, learned_uniform" \
                --predict=1 \
                --visualize=1