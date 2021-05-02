#!/bin/bash
python3 logRegression.py bank-note/train.csv bank-note/test.csv --map
echo 
python3 logRegression.py bank-note/train.csv bank-note/test.csv --mle
exit 0

