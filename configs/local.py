from deepcave.config import Config as C


class Config(C):
    DEBUG = True
    DEV_TOOLS = False
    REFRESH_RATE: int = 2000
