#!/bin/bash
echo =============================================
echo Car Data Set 
echo =============================================
for ((i=1; i<=6; i++))
do
    echo Running depth=$i
    python3 ID3.py car/train.csv car/test.csv $i
    echo 
    python3 ID3.py car/train.csv car/train.csv $i
    echo -------------
done

echo ============================================
echo Bank Data Set, No Replacement
echo ============================================
for ((i=1; i<=16; i++))
do
    echo Running depth=$i
    python3 ID3.py bank/train.csv bank/test.csv $i -num
    echo 
    python3 ID3.py bank/train.csv bank/train.csv $i -num
    echo -------------
done

echo ===========================================
echo Bank Data Set, With Replacement
echo ===========================================
for ((i=1; i<=16; i++))
do
	echo Running depth $i
	python3 ID3.py bank/train.csv bank/test.csv $i -unkn
	echo
	python3 ID3.py bank/train.csv bank/train.csv $i -unkn	
	echo ----------- 
done

