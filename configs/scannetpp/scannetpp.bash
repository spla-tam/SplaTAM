#!/bin/bash

for scene in 0 1
do
    SCENE=${scene}
    export SCENE
    echo "Running scene number ${SCENE} with seed 0"
    python3 -u scripts/splatam.py configs/scannetpp/scannetpp_eval.py
done