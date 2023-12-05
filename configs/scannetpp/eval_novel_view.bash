#!/bin/bash

SCENE=$1
export SCENE

echo "Evaluating scene number ${SCENE} with seed 0"
python3 -u scripts/eval_novel_view.py configs/scannetpp/eval_novel_view.py