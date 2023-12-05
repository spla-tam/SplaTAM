#!/bin/bash

for seed in 0 1 2
do
    SEED=${seed}
    export SEED
    for scene in 0 1 2 3 4 5 6 7
    do
        SCENE_NUM=${scene}
        export SCENE_NUM
        echo "Running scene number ${SCENE_NUM} with seed ${SEED}"
        python3 -u scripts/splatam.py configs/replica/replica_eval.py
    done
done