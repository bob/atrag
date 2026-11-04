"""Unprivileged release delivery; privileged operations go through atrag-control."""

import argparse
import fcntl
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from release import SHA, UV_VERSION, digest, read_archive

BASE = Path("/opt/atrag")
CONTROL = "/usr/local/sbin/atrag-control"
UV = "/usr/local/bin/uv"


def run(*args, cwd=None, timeout=180):
    return subprocess.check_output(args, cwd=cwd, timeout=timeout, text=True).strip()


def control(action, *args):
    return run("sudo", "-n", CONTROL, action, *args, timeout=600)


def atomic_link(target, link):
    temporary = link.with_name(link.name + ".next")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    temporary.replace(link)


def compatible(previous, candidate):
    return previous["schema"] == candidate["schema"] or (
        previous["schema"] in candidate["rollback_safe_from"]
    )


def check_master(sha, token):
    request = Request(
        "https://api.github.com/repos/bob/atrag/git/ref/heads/master",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)["object"]["sha"] == sha


def prepare(incoming, sha):
    path = incoming / "release.tar.gz"
    if digest(path.read_bytes()) != (incoming / "release.sha256").read_text().strip():
        raise ValueError("Archive checksum does not match")
    metadata, files = read_archive(path, sha)
    release = BASE / "releases" / sha
    marker = release / ".ready.json"
    if marker.exists():
        if json.loads(marker.read_text()) != metadata:
            raise ValueError("Existing release differs from delivered artifact")
        # A retry must not replace files or the live virtual environment.
        for name, value in metadata["files"].items():
            if digest((release / name).read_bytes()) != value:
                raise ValueError("Existing release was modified")
        control("run", sha, "config")
        return release, metadata
    if release.exists():
        if (BASE / "current").resolve() == release:
            raise ValueError("Refusing to overwrite an active incomplete release")
        shutil.rmtree(release)
    release.mkdir()
    for name, data in files.items():
        destination = release / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (release / "release.json").write_text(json.dumps(metadata))
    (release / ".env").symlink_to(BASE / "shared/.env")
    (release / ".streamlit").mkdir(exist_ok=True)
    (release / ".streamlit/secrets.toml").symlink_to(BASE / "shared/secrets.toml")
    if run(UV, "--version").split()[1] != UV_VERSION:
        raise ValueError("Server uv version differs from the build runtime")
    run(
        "env",
        "UV_PYTHON_INSTALL_DIR=/opt/atrag/python",
        UV,
        "sync",
        "--locked",
        "--no-dev",
        "--python",
        metadata["python"],
        cwd=release,
        timeout=600,
    )
    run(
        str(release / ".venv/bin/python"),
        "-c",
        "import atrag.cli, atrag.interface.views, streamlit",
        cwd=release,
    )
    control("run", sha, "config")
    control("run", sha, "warm")
    marker.write_text(json.dumps(metadata))
    return release, metadata


def health(release, *, external=True, schema_release=None):
    for url in (
        "http://127.0.0.1:8501/_stcore/health",
        *(["https://tax.example.com/_stcore/health"] if external else []),
    ):
        request = Request(
            url, headers={"User-Agent": "ATRAG-Health/1.0", "Accept": "text/plain"}
        )
        failure = "unexpected response"
        for attempt in range(20):
            try:
                with urlopen(request, timeout=5) as response:
                    if response.status == 200 and response.read().strip() == b"ok":
                        break
                    failure = f"HTTP {response.status}; expected body 'ok'"
            except OSError as error:
                failure = str(error)
            if attempt == 19:
                raise RuntimeError(f"Streamlit health check failed: {url} ({failure})")
            time.sleep(2)
    control("run", (schema_release or release).name, "smoke")
    if (BASE / "current").resolve() != release:
        raise RuntimeError("Active release pointer changed")
    # /proc/<pid>/cwd of the atrag user is not readable by deploy.
    control("verify-sha", release.name)
    control("status")


def prune(current, previous):
    keep = {current, previous}
    successes = sorted(
        (BASE / "releases").glob("*/.successful"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    keep.update(p.parent for p in successes[:5])
    for marker in successes:
        if marker.parent not in keep:
            shutil.rmtree(marker.parent)


def activate(release, metadata, token, bootstrap=False, external=True):
    link = BASE / "current"
    previous = link.resolve() if link.is_symlink() else None
    if bootstrap and previous is not None:
        raise ValueError("Corpus bootstrap is restricted to the first deployment")
    if previous and previous != release:
        old = json.loads((previous / "release.json").read_text())
        if not compatible(old, metadata):
            raise ValueError(
                "Schema/checkpoint change needs a reviewed rollback_safe_from entry or separate maintenance"
            )
    if not check_master(release.name, token):
        print("Superseded commit; production unchanged")
        return
    if previous == release and (release / ".successful").exists():
        health(release, external=external)
        return
    if previous is None and not bootstrap:
        raise ValueError("First deployment requires --bootstrap")
    stopped = False
    migration_started = False
    backup = False
    try:
        control("stop")
        stopped = True
        control("backup", release.name)
        backup = True
        migration_started = True
        control("run", release.name, "migrate")
        if bootstrap:
            control("run", release.name, "bootstrap")
        atomic_link(release, link)
        control("start")
        health(release, external=external)
        (release / ".successful").touch()
        if previous and previous != release:
            atomic_link(previous, BASE / "previous")
    except BaseException:
        # The compatibility gate runs before downtime and covers partial checkpoint setup.
        if stopped:
            control("stop")
            if previous and (
                not migration_started
                or compatible(
                    json.loads((previous / "release.json").read_text()), metadata
                )
            ):
                atomic_link(previous, link)
                control("start")
                health(previous, external=external, schema_release=release)
                print("Previous release restored; deployment failed", file=sys.stderr)
            elif previous is None:
                link.unlink(missing_ok=True)
        if backup:
            print(
                "Temporary recovery dump retained in /var/backups/atrag",
                file=sys.stderr,
            )
        raise
    # Cleanup errors must not roll back a release that already passed acceptance.
    control("discard-backup", release.name)
    prune(release, previous)
    print("Deployed " + release.name)


def rollback(sha, external=True):
    current = (BASE / "current").resolve()
    target = BASE / "releases" / sha
    if not (target / ".successful").exists():
        raise ValueError("Rollback target was never successfully deployed")
    if not compatible(
        json.loads((target / "release.json").read_text()),
        json.loads((current / "release.json").read_text()),
    ):
        raise ValueError("Rollback schema compatibility has not been reviewed")
    try:
        control("stop")
        atomic_link(target, BASE / "current")
        control("start")
        health(target, external=external, schema_release=current)
    except BaseException:
        control("stop")
        atomic_link(current, BASE / "current")
        control("start")
        raise
    atomic_link(current, BASE / "previous")
    print("Rolled back to " + sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--incoming", type=Path)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument(
        "--local-health-only",
        action="store_true",
        help="Initial setup before DNS; not accepted in CI",
    )
    args = parser.parse_args()
    if not SHA.fullmatch(args.sha):
        parser.error("A full commit SHA is required")

    def interrupted(signum, frame):
        raise InterruptedError("Deployment interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    with (BASE / "deploy.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.rollback:
            rollback(args.sha, external=not args.local_health_only)
            return
        if args.incoming is None:
            parser.error("--incoming is required")
        token = json.load(sys.stdin)["token"]
        release, metadata = prepare(args.incoming, args.sha)
        activate(release, metadata, token, args.bootstrap, not args.local_health_only)


if __name__ == "__main__":
    main()
