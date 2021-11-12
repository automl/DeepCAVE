import os


def make_dirs(filename):
    # Remove last entry
    filename_array = filename.split("/")[:-1]
    path = "/".join(filename_array)

    os.makedirs(path, exist_ok=True)
