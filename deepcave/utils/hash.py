import hashlib
from pathlib import Path


def string_to_hash(string: str) -> str:
    hash_object = hashlib.md5(string.encode())
    return hash_object.hexdigest()


def file_to_hash(filename: Path) -> str:
    hash = hashlib.md5()
    with Path(filename).open("rb") as f:
        while chunk := f.read(4082):
            hash.update(chunk)

    return hash.hexdigest()
