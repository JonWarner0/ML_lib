#!/bin/bash
echo --- Bagging Bias and Variance ----
python3 Bagging.py bank/train.csv bank/test.csv 500 -num 1
echo ---- Random Forest Bias and Variance ----
python3 RandomForest.py bank/train.csv bank/test.csv 500 -num 1
exit 0
