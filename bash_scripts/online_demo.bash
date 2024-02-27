#!/bin/bash

# check rmem_max and wmem_max, and increase size if necessary
if [ "$#" -ne 1 ]; then
    echo "Usage: bash_scripts/online_demo.bash <config_file>"
    exit
fi

if [ ! -f $1 ]; then
    echo "Config file not found!"
    exit
fi

if sysctl -a | grep -q "net.core.rmem_max = 2147483647"; then
    echo "rmem_max already set to 2147483647"
else
    echo "Setting rmem_max to 2147483647"
    sudo sysctl -w net.core.rmem_max=2147483647
fi

if sysctl -a | grep -q "net.core.wmem_max = 2147483647"; then
    echo "wmem_max already set to 2147483647"
else
    echo "Setting wmem_max to 2147483647"
    sudo sysctl -w net.core.wmem_max=2147483647
fi

# Online Dataset Capture & SplaTAM
python3 scripts/iphone_demo.py --config $1

# Visualize SplaTAM Output
python3 viz_scripts/final_recon.py $1
