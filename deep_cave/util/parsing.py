import json
from typing import Dict, Union, Optional
import datetime

import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write.json import read, write

from .logs import get_logger


logger = get_logger(__name__)


def deep_cave_data_encoder(data: Dict, **parser_kwargs) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=True, cls=JsonEncoder, **parser_kwargs)


def parse_datetime(s: str) -> Union[datetime.datetime, datetime.date, datetime.time]:
    return datetime.datetime.fromisoformat(s)


def serialize_datetime(datetime_object: Optional[Union[datetime.datetime, datetime.date, datetime.time]]) -> str:
    return datetime_object.isoformat()


def serialize_timedelta(timedelta_object: datetime.timedelta) -> float:
    return timedelta_object.total_seconds()


def parse_timedelta(v: float) -> datetime.timedelta:
    return datetime.timedelta(milliseconds=v*1000)


def parse_configspace(s: str) -> ConfigurationSpace:
    return read(s)


def serialize_configspace(configuration_space: ConfigurationSpace) -> str:
    return write(configuration_space)


def deep_cave_data_decoder(str_data: str) -> Dict:
    return json.loads(str_data, object_hook=deep_cave_hook)


def deep_cave_hook(dct):
    for k, v in dct.items():
        if k in ['start_time', 'end_time']:
            if v is None:
                logger.warning(f'Field {k} in .json file doesn\'t contain a valid value.'
                               f' Expecting "datetime.datetime.fromisoformat". Got {v} of type {type(v)}.'
                               f'Using "None" as default value.')
                dct[k] = None
                continue
            dct[k] = datetime.datetime.fromisoformat(v)
        if k == 'duration':
            if v is None:
                logger.warning(f'Field {k} in .json file doesn\'t contain a valid value.'
                               f' Expecting "int" as datetime.timedelta in seconds. Got {v} of type {type(v)}.'
                               f'Using "None" as default value.')
                dct[k] = None
                continue
            dct[k] = datetime.timedelta(milliseconds=v*1000)
        if k == 'search_space':
            if v is None:
                continue
            dct[k] = read(v)
    return dct


class JsonEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        elif isinstance(obj, datetime.timedelta):
            return obj.total_seconds()

        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None
        elif isinstance(obj, ConfigurationSpace):
            return write(obj)

        return json.JSONEncoder.default(self, obj)
