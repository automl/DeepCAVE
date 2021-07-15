import os
import json
import glob
import pandas as pd
from typing import Dict, Type, Any

from deep_cave.converter.converter import Converter
from deep_cave.util.run import Run


class Test(Converter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def name():
        return 'Test'

    def retrieve_meta(self, run_name):
        return {}

    def retrieve_trials(self, run_name):
        return {}

