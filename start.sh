#!/bin/bash

# Abort script if any error occurs
trap "exit" INT TERM ERR
trap "kill 0" EXIT

# Check if redis-server is installed
if ! [ -x "$(command -v redis-server)" ]; then
    echo "Error: redis-server is not installed." >&2
    exit 1
fi

REDIS_PORT=$(deepcave --get_config REDIS_PORT)

# We ping the redis server to see if it is running
RUNNING=true
redis-cli -p $REDIS_PORT PING || RUNNING=false

# Check if redis-server is already running
if [ "$RUNNING" == false ]; then
    echo "Redis server is not running. Starting..."

    # We don't want to show the redis-server output
    redis-server --port $REDIS_PORT > /dev/null 2>&1 &
    echo "Redis server successfully started."
else
    echo "Redis server already running. Skipping..."
fi

# Start worker in background
python worker.py &
python server.py