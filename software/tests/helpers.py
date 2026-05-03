from pathlib import Path
from contextlib import contextmanager
from collections.abc import Iterator
from uuid import uuid4


@contextmanager
def temporary_directory() -> Iterator[str]:
    root = Path(__file__).resolve().parents[1] / ".test_tmp"
    root.mkdir(exist_ok=True)
    directory = root / f"tmp-{uuid4().hex}"
    directory.mkdir()
    yield str(directory)
