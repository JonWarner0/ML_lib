#!/bin/bash 
echo ------ Batch Gradient Descent ------
python3 BatchGD.py concrete/train.csv concrete/test.csv 
echo ------ Stochastic Gradient Descent ------
python3 StochasticGD.py concrete/train.csv concrete/test.csv
exit 0

