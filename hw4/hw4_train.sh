#!/usr/bin/env bash 

#python3.6 load_data.py 
python3.6 train.py --epoch 5 --batch 1024 --model_name \
    6 --which_gpu 0 --gpu_fraction 0.9 --which_model 2

python3.6 train.py --epoch 5 --batch 1024 --model_name \
    7 --which_gpu 0 --gpu_fraction 0.9 --which_model 1
