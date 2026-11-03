"""Receive only the two release artifacts through the restricted delivery key."""

import fcntl
import re
import sys
import tarfile
import tempfile
from pathlib import Path

BASE = Path("/opt/atrag")


def receive(sha, stream):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected a full commit SHA")
    with (BASE / "deploy.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=BASE / "incoming") as directory:
            temporary = Path(directory)
            names = set()
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    if (
                        not member.isfile()
                        or member.name not in {"release.tar.gz", "release.sha256"}
                        or member.name in names
                        or not 0 < member.size <= 100_000_000
                    ):
                        raise ValueError("Unexpected delivery member")
                    content = archive.extractfile(member)
                    assert content is not None
                    (temporary / member.name).write_bytes(content.read())
                    names.add(member.name)
            if names != {"release.tar.gz", "release.sha256"}:
                raise ValueError("Incomplete delivery")
            destination = BASE / "incoming" / sha
            destination.mkdir(exist_ok=True)
            for name in names:
                (temporary / name).replace(destination / name)


if __name__ == "__main__":
    receive(sys.argv[1], sys.stdin.buffer)
