from deepcave.config import Config as C


class Config(C):
    DEBUG = False

    REDIS_PORT = 6379
    REDIS_ADDRESS = "redis://localhost"

    DASH_PORT = 8050
    DASH_ADDRESS = "re"  # If you are connected to a remote server sass@se, the address is "re".
