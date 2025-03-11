#!/bin/bash
for i in $(seq 1 10);
do
    for j in $(seq 1000 1100);
    do
        python ../train_bc_ac_gym.py --Tx $i --Rx $j --train_ac_only True
    done
done
