
from deep_cave import queue
from rq import Connection, Worker


# https://github.com/tcbegley/dash-rq-demo
if __name__ == "__main__":
    with Connection(queue._connection):
        w = Worker([queue._queue])
        w.work()
