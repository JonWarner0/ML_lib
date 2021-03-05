#!/bin/bash

python3 RandomForest.py bank/train.csv bank/test.csv 1 -num &
for((i=50;i<=500;i+=50))
do
	python3 RandomForest.py bank/train.csv bank/test.csv $i -num &
done

exit 0
