sudo sysctl -w net.core.rmem_max=2147483647
sudo sysctl -w net.core.wmem_max=2147483647

# Capture Dataset
python3 scripts/nerfcapture2dataset.py --config $1

# Run SplaTAM
python3 scripts/splatam.py $1

# Visualize SplaTAM Output
python3 viz_scripts/final_recon.py $1