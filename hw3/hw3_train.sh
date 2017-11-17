echo "loading data..."

python3.6 load_data.py ${1}

echo "training..."
python3.6 train.py  --epoch 100 --batch 1024 --which_model 0 --model_name model_0_1024_100_4


