#!/usr/bin/env bash 
echo -e "testing generative model..."
./hw2_generative.sh 1 1 feature/X_train feature/Y_train feature/X_test
echo -e "testing logistic model..."
./hw2_logistic.sh 1 1 feature/X_train feature/Y_train feature/X_test
echo -e "testing best model..."
./hw2_best.sh 1 1 feature/X_train feature/Y_train feature/X_test best_out/