#!/usr/bin/env bash 

python3.6 load_data.py 
python3.6 train.py --epoch 30 --batch 128 --model_name \
    1 --which_gpu 0 --gpu_fraction 1
