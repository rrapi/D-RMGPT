#!/bin/bash

echo "Installing required libraries..."

apt-get install -y pip
pip install -r requirements.txt
export PATH=$USER/.local/bin:$PATH >> ~/.bashrc



echo "Done!"
