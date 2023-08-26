"""
# Worker.

This module can start an RQ worker.
"""

from rq import Connection, Worker

from deepcave import queue

# https://github.com/tcbegley/dash-rq-demo
if __name__ == "__main__":
    print("\n-------------STARTING WORKER-------------")
    with Connection(queue._connection):
        w = Worker([queue._queue])
        w.work()
