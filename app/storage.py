import shutil
from pathlib import Path

from app.config import settings

# ponytail: local disk behind key-based API; S3 client drops in behind these four functions


def path_for(key: str) -> Path:
    return Path(settings.media_dir) / key


def save_bytes(key: str, data: bytes) -> str:
    p = path_for(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return key


def read_bytes(key: str) -> bytes:
    return path_for(key).read_bytes()


def save_upload(key: str, fileobj) -> str:
    p = path_for(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as out:
        shutil.copyfileobj(fileobj, out)
    return key
