#!/bin/bash
python3 nn.py bank-note/train.csv bank-note/test.csv 5 4
python3 nn.py bank-note/train.csv bank-note/test.csv 10 4
python3 nn.py bank-note/train.csv bank-note/test.csv 25 4
python3 nn.py bank-note/train.csv bank-note/test.csv 50 4
python3 nn.py bank-note/train.csv bank-note/test.csv 100 4
exit 0
