#!/usr/bin/env bash 

#python3.6 load_data.py 
python3.6 train.py --epoch 30 --batch 256 --model_name \
    1 --which_gpu 0 --gpu_fraction 1

python3.6 train.py --epoch 30 --batch 256 --model_name \
    0 --which_gpu 0 --gpu_fraction 1 --which_model 1
