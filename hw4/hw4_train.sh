#!/usr/bin/env bash 

#python3.6 load_data.py 
python3.6 train.py --epoch 30 --batch 256 --model_name \
    4 --which_gpu 0 --gpu_fraction 1 --which_model 2

python3.6 train.py --epoch 30 --batch 512 --model_name \
    5 --which_gpu 0 --gpu_fraction 1 --which_model 2
