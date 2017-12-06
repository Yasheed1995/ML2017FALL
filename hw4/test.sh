
python3.6 train.py LSTM_5 train --cell LSTM
python3.6 train.py GRU_5 train --cell GRU
timeout 2h python3.6 train.py LSTM_5_semi semi --load_model LSTM_5
timeout 2h python3.6 train.py GRU_5_semi semi --load_model GRU_5
