#!/bin/bash

trap "exit" INT TERM ERR
trap "kill 0" EXIT

if [ $1 == "--clean" ] || [ $1 == "-c" ];
then
    # Useful for debugging
    make clean
fi

REDIS_PORT=$(deepcave --get_config REDIS_PORT)

redis-server --port $REDIS_PORT &
python worker.py &
python server.py