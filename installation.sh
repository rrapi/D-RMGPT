#!/bin/bash

echo "Installing required libraries..."

sudo apt-get install -y pip
pip install -r requirements.txt
pip install matplotlib pyrealsense2

export PATH=$HOME/.local/bin:$PATH >> ~/.bashrc
echo $PATH

echo "Done!"
