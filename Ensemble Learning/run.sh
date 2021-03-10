#!/bin/bash
echo -------- AdaBoosting ----------
python3 AdaBoost.py bank/train.csv bank/test.csv 1 -num
for ((i=50;i<=500;i+=50))
do
    python3 AdaBoost.py bank/train.csv bank/test.csv $i -num
done

echo -------- Bagging ---------
python3 Bagging.py bank/train.csv bank/test.csv 1 -num
for ((i=50;i<=500;i+=50))
do
    python3 Bagging.py bank/train.csv bank/test.csv $i -num
done

echo ------- RandomForest --------
python3 RandomForest.py bank/train.csv bank/test.csv 1 -num
for ((i=50;i<=500;i+=50))
do
    python3 RandomForest.py bank/train.csv bank/test.csv $i -num
done

echo ------ Batch Gradient Descent ------
python3 BatchGD.py concrete/train.csv concrete/test.csv

echo ------ Stochastic Gradient Descent ------
python3 StochasticGD.py concrete/train.csv concrete/test.csv

