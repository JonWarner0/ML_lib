#!/bin/bash
python3 AdaBoost.py bank/train.csv bank/train.csv 1 -num
for((i=50;i<=500;i+=50))
do
	python3 AdaBoost.py bank/train.csv bank/train.csv $i -num
done;
exit 0
