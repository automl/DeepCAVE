from deepcave import queue
from rq import Connection, Worker


# https://github.com/tcbegley/dash-rq-demo
if __name__ == "__main__":
    print("\n-------------STARTING WORKER-------------")
    with Connection(queue._connection):
        w = Worker([queue._queue])
        w.work()
