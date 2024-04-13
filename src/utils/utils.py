import json
from os import path, makedirs


def json_to_str(v):
    try:
        json.loads(v)
        return json.dumps(v)
    except Exception:
        return v


def get_root_dir():
    return path.dirname(path.abspath(path.join(__file__, "..", "..")))


def init_dirs(*paths):
    for p in paths:
        makedirs(p, exist_ok=True)
