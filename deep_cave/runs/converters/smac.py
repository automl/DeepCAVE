import os
import json
import glob
import pandas as pd
from typing import Dict, Type, Any

from deep_cave.runs.converters.converter import Converter
from deep_cave.runs.run import Run
from smac.runhistory.runhistory import RunHistory
from ConfigSpace.read_and_write import json as cs_json


class SMAC(Converter):
    def __init__(self):
        super().__init__()

    @staticmethod
    def name():
        return 'SMAC'

    def retrieve_meta(self):
        #meta = self._get_json_content("meta")

        return {

        }

    def retrieve_runhistory(self):
        rh_filename = self._get_json_filename("runhistory")

        rh = RunHistory()
        rh.load_json(rh_filename, cs=self.retrieve_configspace())

        return rh

    def retrieve_configspace(self):
        cs_filename = self._get_json_filename("configspace")

        with open(cs_filename, 'r') as f:
            json_string = f.read()
            config = cs_json.read(json_string)

        return config

