sudo sysctl -w net.core.rmem_max=2147483647
sudo sysctl -w net.core.wmem_max=2147483647

# Online Dataset Capture & SplaTAM
python3 scripts/iphone_demo.py --config $1

# Visualize SplaTAM Output
python3 viz_scripts/final_recon.py $1