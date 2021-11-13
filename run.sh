#!/bin/bash

trap "exit" INT TERM ERR
trap "kill 0" EXIT

make clean

redis-server &
python worker.py &
python server.py