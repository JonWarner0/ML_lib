#!/bin/bash
python3 perceptron.py bank-note/train.csv bank-note/test.csv --standard
echo   
python3 perceptron.py bank-note/train.csv bank-note/test.csv --voted
echo   
python3 perceptron.py bank-note/train.csv bank-note/test.csv --avg
exit 0
