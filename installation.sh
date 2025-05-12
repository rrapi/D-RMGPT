#!/bin/bash

# ======== To run the script execute:
# > chmod +x installation.sh
# > ./intallation.sh <USER> (in this case `robot`)




echo "Installing required libraries..."

sudo apt-get update && apt-get install -y --no-install-recommends pip sox
pip install -r requirements.txt
pip install matplotlib pyrealsense2

export PATH=$HOME/.local/bin:$PATH >> ~/.bashrc
echo $PATH

sudo usermod -aG audio "$1"

echo "Done!"
