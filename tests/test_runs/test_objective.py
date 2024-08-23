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

import json
import unittest

from deepcave.runs.exceptions import NotMergeableError
from deepcave.runs.objective import Objective
from deepcave.utils.compression import Encoder


class TestRun(unittest.TestCase):
    def test(self) -> None:
        o1 = Objective("test", lower=0, upper=1, optimize="lower")
        assert o1.lock_lower
        assert o1.lock_upper

        o2 = Objective("test2", optimize="upper")
        assert not o2.lock_lower
        assert not o2.lock_upper

        with self.assertRaises(NotMergeableError):
            o1.merge(o2)

        # If objective changes upper and not locked, then they should be mergeable
        o3 = Objective("test3", lower=0, optimize="lower")
        o3.upper = 40
        o4 = Objective("test3", lower=0, optimize="lower")
        o4.upper = 50
        o3.merge(o4)

        # That should be successfull because o1 and o5 are the same
        o5 = Objective("test", lower=0, upper=1, optimize="lower")
        o1.merge(o5)

        # This should fail now because upper is locked but is changed
        o5.upper = 50
        with self.assertRaises(NotMergeableError):
            o1.merge(o5)

    def test_serializable(self) -> None:
        o1 = Objective("test", lower=0, upper=1, optimize="lower")
        o1_json = o1.to_json()
        json.dumps(o1_json, cls=Encoder)
        o2 = Objective.from_json(o1_json)
        o1.merge(o2)


if __name__ == "__main__":
    unittest.main()
