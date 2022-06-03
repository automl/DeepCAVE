import json
import unittest

from deepcave.runs.exceptions import NotMergeableError
from deepcave.runs.objective import Objective


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
        json.dumps(o1_json)
        o2 = Objective.from_json(o1_json)
        o1.merge(o2)


if __name__ == "__main__":
    unittest.main()
