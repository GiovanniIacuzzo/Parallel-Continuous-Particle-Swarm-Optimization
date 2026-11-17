"""Compatibility entry point. The maintained workflow is publication.py."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import json


def pin(path,settings):
    from publication_tools.common import freeze
    return freeze(Path(path),settings)


def read(path):return json.loads(Path(path).read_text())


@contextmanager
def campaign_lock(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.campaign.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(f'Another campaign is active in {folder}')
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':
    from publication import main
    main()
