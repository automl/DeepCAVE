#!/bin/bash

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python record_openml.py --seed=0 &
python record_openml.py --seed=1 &
python record_openml.py --seed=2 &
python record_openml.py --seed=3 &
python record_openml.py --seed=4 &
python record_openml.py --seed=5 &