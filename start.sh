#!/bin/bash

while true
do

echo -e start human detector

source ./venv/bin/activate
python3 ./main.py

deactivate

echo -e restarting human detector in 5 seconds

sleep 5

done
