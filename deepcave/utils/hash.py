import hashlib


def string_to_hash(string):
    hash_object = hashlib.md5(string.encode())
    return hash_object.hexdigest()


def file_to_hash(filename):
    with open(filename, "rb") as f:
        hash = hashlib.md5()
        while chunk := f.read(4082):
            hash.update(chunk)

        return hash.hexdigest()
