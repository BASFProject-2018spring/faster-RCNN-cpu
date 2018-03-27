#!/bin/bash
echo "Starting, might take several minutes"
cd ~/nematodes/app/tools
echo "CD"
source /home/ac297/anaconda3/bin/activate py2 
echo "Source"
python run.py
