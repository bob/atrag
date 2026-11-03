"""Build and verify a small, secret-free release from an exact Git commit."""

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

PYTHON_VERSION = "3.14.6"
UV_VERSION = "0.12.1"
SHA = re.compile(r"[0-9a-f]{40}")
PATHS = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
    "streamlit_app.py",
    "src",
    "data/sources",
    "scripts/deploy",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def schema_fingerprint(files):
    # Dependency changes may also change LangGraph checkpoint migrations.
    return digest(
        b"".join(
            name.encode() + b"\0" + files[name]
            for name in sorted(files)
            if name.startswith("src/atrag/migrations/") or name == "uv.lock"
        )
    )


def read_archive(path, expected_sha):
    """Validate every member before extracting anything (including links)."""
    if not SHA.fullmatch(expected_sha):
        raise ValueError("Expected a full lowercase commit SHA")
    with tarfile.open(path, "r:gz") as archive:
        files = {}
        for member in archive:
            name = member.name
            if (
                not member.isfile()
                or name in files
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or name.startswith("./")
                or not (
                    name == "release.json"
                    or any(name == p or name.startswith(p + "/") for p in PATHS)
                )
            ):
                raise ValueError("Unsafe or unexpected release member")
            stream = archive.extractfile(member)
            assert stream is not None
            files[name] = stream.read()
    metadata = json.loads(files.pop("release.json"))
    if metadata["sha"] != expected_sha:
        raise ValueError("Release commit does not match requested commit")
    if metadata["files"] != {name: digest(data) for name, data in files.items()}:
        raise ValueError("Release file checksums do not match")
    if metadata["schema"] != schema_fingerprint(files):
        raise ValueError("Release schema fingerprint does not match")
    for required in ("pyproject.toml", "uv.lock", "streamlit_app.py"):
        if required not in files:
            raise ValueError("Incomplete release")
    if metadata["python"] != PYTHON_VERSION or metadata["uv"] != UV_VERSION:
        raise ValueError("Unexpected release runtime")
    return metadata, files


def build(root, output, sha):
    if not SHA.fullmatch(sha):
        raise ValueError("Expected a full commit SHA")
    raw = subprocess.check_output(["git", "archive", sha, "--", *PATHS], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        files = {}
        for member in archive:
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                files[member.name] = stream.read()
    policy = json.loads(files["scripts/deploy/compatibility.json"])
    metadata = {
        "sha": sha,
        "python": PYTHON_VERSION,
        "uv": UV_VERSION,
        "schema": schema_fingerprint(files),
        "rollback_safe_from": policy["rollback_safe_from"],
        "files": {name: digest(data) for name, data in files.items()},
    }
    files["release.json"] = json.dumps(metadata, sort_keys=True).encode()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "release.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
    (output / "release.sha256").write_text(digest(path.read_bytes()) + "\n")
    read_archive(path, sha)
    print(json.dumps({"sha": sha, "schema": metadata["schema"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(Path.cwd(), args.output, args.sha)
