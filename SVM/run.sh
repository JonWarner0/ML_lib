#!/bin/bash
python3 svmPrimal.py bank-note/train.csv bank-note/test.csv 0
python3 svmPrimal.py bank-note/train.csv bank-note/test.csv 1
python3 svmPrimal.py bank-note/train.csv bank-note/test.csv 2

python3 svmDual.py bank-note/train.csv bank-note/test.csv 0
python3 svmDual.py bank-note/train.csv bank-note/test.csv 1
python3 svmDual.py bank-note/train.csv bank-note/test.csv 2

python3 svmGauss.py bank-note/train.csv bank-note/test.csv 0 0
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 0 1
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 0 2
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 0 3
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 0 4
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 1 0
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 1 1
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 1 2
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 1 3
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 1 4
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 2 0
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 2 1
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 2 2
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 2 3
python3 svmGauss.py bank-note/train.csv bank-note/test.csv 2 4

exit 0

