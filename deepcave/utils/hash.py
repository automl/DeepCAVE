import hashlib


def string_to_hash(string: str) -> str:
    hash_object = hashlib.md5(string.encode())
    return hash_object.hexdigest()


def file_to_hash(filename: str) -> str:
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        while chunk := f.read(4082):
            hash.update(chunk)

    return hash.hexdigest()
