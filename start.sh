#!/bin/bash

# Abort script if any error occurs
trap "exit" INT TERM ERR
trap "kill 0" EXIT

INSTALLED="$(which deepcave)"

if ! [ $INSTALLED ]; then
    echo "Error: deepcave is not installed." >&2
    exit 1
fi

ROOT_PATH=`
python << END
import deepcave
from pathlib import Path
print(str(Path(deepcave.__file__).parent.parent))
END
`

# We remove dump.rdb to avoid issues from previous runs
file="dump.rdb"
if [ -f "$file" ] ; then
    rm "$file"
fi

OPEN=$1

# Save config value
CONFIG=$2
if ! [ $CONFIG ]; then
    CONFIG="default"
fi
echo "Using config '$CONFIG'"

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
python "$ROOT_PATH/worker.py" --config $CONFIG &

# Start the webserver
if [ $OPEN == "true" ]; then
    python "$ROOT_PATH/server.py" --config $CONFIG --open
else
    python "$ROOT_PATH/server.py" --config $CONFIG
fi
