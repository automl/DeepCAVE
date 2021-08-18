import os

root = os.getcwd()
cache_dir = os.path.join(root, "cache")
working_dir = os.path.join(root, "data", "smac3-output")


CONFIG = {
    'DIR': cache_dir,
    'NAME': "sEjia302pyWqs",
    'REDIS_URL': "redis://localhost:6379"
}


REQUIRED_DATA = {
    'working_dir': working_dir,
    'converter_name': 'SMAC',
    'run_id': None,
    'matplotlib-mode': False,
}


__all__ = [CONFIG, REQUIRED_DATA]
