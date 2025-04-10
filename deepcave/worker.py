# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Worker.

This module can create a redis queue worker.
"""

from rq import Connection, Worker  # type: ignore

from deepcave import queue

# https://github.com/tcbegley/dash-rq-demo
if __name__ == "__main__":
    print("\n-------------STARTING WORKER-------------")
    with Connection(queue._connection):
        w = Worker([queue._queue])
        w.work()
