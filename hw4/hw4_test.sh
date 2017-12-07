python3.6 train.py testing test --load_model GRU_5 --test_path ${1} \
    --result_path result/GRU_5.csv

python3.6 train.py testing test --load_model GRU_5_semi --test_path ${1} \
    --result_path result/GRU_5_semi.csv

python3.6 train.py testing test --load_model LSTM_5_semi --test_path ${1} \
    --result_path result/LSTM_5_semi.csv


python3.6 train.py testing test --load_model LSTM_5 --test_path ${1} \
    --result_path ${2} 
