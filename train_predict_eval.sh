#!/bin/bash


python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="none" \
                --experimental=0


python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="smooth" \
                --experimental=1

python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="framewise" \
                --experimental=1

python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="learned" \
                --experimental=1


python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="uniform" \
                --experimental=1

 python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="learned_smooth" \
                --experimental=1

python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="learned_framewise" \
                --experimental=1
#
python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="learned_uniform" \
                --experimental=1

python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="learned_framewise_exp" \
                --experimental=1

python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="learned_smooth_exp" \
                --experimental=1


python train_predict_eval.py --dataset=valid --fold="0,1,2,3,4" \
                --num_epochs=15 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --weight_type="learned_poly" \
                --experimental=1

