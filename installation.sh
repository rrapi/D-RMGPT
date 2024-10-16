#!/bin/bash

echo "Installing required libraries..."

sudo apt-get install -y pip
pip install numpy
pip install -r requirements.txt
export PATH=$HOME/.local/bin:$PATH >> ~/.bashrc
echo $PATH

echo "Done!"
