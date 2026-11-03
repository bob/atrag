"""Deployment failures must preserve the active release and application data."""

import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest


@pytest.fixture
def deployment(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/deploy"
    monkeypatch.syspath_prepend(str(source))
    spec = importlib.util.spec_from_file_location(
        "atrag_deploy_server", source / "server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BASE", tmp_path)
    return module


def release(deployment, sha, schema="a", safe=None):
    directory = deployment.BASE / "releases" / (sha * 40)
    directory.mkdir(parents=True)
    metadata = {
        "sha": directory.name,
        "schema": schema,
        "rollback_safe_from": safe or [],
    }
    (directory / "release.json").write_text(json.dumps(metadata))
    return directory, metadata


def active_release(deployment):
    old, _ = release(deployment, "a")
    (old / ".successful").touch()
    (deployment.BASE / "current").symlink_to(old)
    return old


@pytest.mark.parametrize("failure", ["backup", "migrate", "start", "health"])
def test_failed_activation_restores_previous_release(deployment, monkeypatch, failure):
    old = active_release(deployment)
    new, metadata = release(deployment, "b")
    calls, failed = [], False

    def control(action, *args):
        nonlocal failed
        calls.append((action, *args))
        if not failed and (
            action == failure or (failure == "migrate" and args[-1:] == ("migrate",))
        ):
            failed = True
            raise RuntimeError("injected failure")

    def health(directory, **kwargs):
        if failure == "health" and directory == new:
            raise RuntimeError("injected health failure")

    monkeypatch.setattr(deployment, "control", control)
    monkeypatch.setattr(deployment, "health", health)
    monkeypatch.setattr(deployment, "check_master", lambda *args: True)
    with pytest.raises(RuntimeError):
        deployment.activate(new, metadata, "token")
    assert (deployment.BASE / "current").resolve() == old
    assert ("start",) in calls
    assert not (new / ".successful").exists()
    assert not any(c[0] == "discard-backup" for c in calls)


@pytest.mark.parametrize("reason", ["stale", "incompatible", "bootstrap"])
def test_rejected_deploy_never_stops_service(deployment, monkeypatch, reason):
    old = active_release(deployment)
    new, metadata = release(
        deployment, "b", schema="different" if reason == "incompatible" else "a"
    )
    calls = []
    monkeypatch.setattr(deployment, "control", lambda *args: calls.append(args))
    monkeypatch.setattr(deployment, "check_master", lambda *args: reason != "stale")
    if reason == "stale":
        deployment.activate(new, metadata, "token")
    else:
        with pytest.raises(ValueError):
            deployment.activate(new, metadata, "token", bootstrap=reason == "bootstrap")
    assert calls == []
    assert (deployment.BASE / "current").resolve() == old


def test_success_records_previous_and_removes_temporary_dump(deployment, monkeypatch):
    old = active_release(deployment)
    new, metadata = release(deployment, "b")
    calls = []
    monkeypatch.setattr(deployment, "control", lambda *args: calls.append(args))
    monkeypatch.setattr(deployment, "check_master", lambda *args: True)
    monkeypatch.setattr(deployment, "health", lambda *args, **kwargs: None)
    deployment.activate(new, metadata, "token")
    assert (deployment.BASE / "current").resolve() == new
    assert (deployment.BASE / "previous").resolve() == old
    assert (new / ".successful").exists()
    assert calls.index(("backup", new.name)) < calls.index(("run", new.name, "migrate"))
    assert calls[-1] == ("discard-backup", new.name)


def test_rollback_failure_restarts_original_release(deployment, monkeypatch):
    old = active_release(deployment)
    target, _ = release(deployment, "b")
    (target / ".successful").touch()
    calls = []
    monkeypatch.setattr(deployment, "control", lambda *args: calls.append(args))

    def fail(*args, **kwargs):
        raise RuntimeError("bad rollback target")

    monkeypatch.setattr(deployment, "health", fail)
    with pytest.raises(RuntimeError):
        deployment.rollback(target.name)
    assert (deployment.BASE / "current").resolve() == old
    assert calls[-1] == ("start",)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../secret", "file"),
        (".env", "file"),
        ("src/link", "symlink"),
        ("src/link", "hardlink"),
    ],
)
def test_archive_rejects_unsafe_members(deployment, tmp_path, name, kind):
    from release import read_archive

    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo(name)
        member.type = {
            "file": tarfile.REGTYPE,
            "symlink": tarfile.SYMTYPE,
            "hardlink": tarfile.LNKTYPE,
        }[kind]
        member.linkname = "/etc/passwd" if kind != "file" else ""
        output.addfile(member, io.BytesIO())
    with pytest.raises(ValueError, match="Unsafe"):
        read_archive(archive, "a" * 40)


def test_artifact_uses_commit_and_excludes_secrets(deployment, tmp_path):
    from release import build, read_archive

    root = tmp_path / "repository"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    git("init", "-q")
    files = {
        "pyproject.toml": "original",
        "uv.lock": "lock",
        ".python-version": "3.14.6",
        "README.md": "readme",
        "streamlit_app.py": "app",
        "src/atrag/migrations/__init__.py": "migrate",
        "data/sources/cirs.json": "{}",
        "scripts/deploy/compatibility.json": '{"rollback_safe_from": []}',
        ".env": "secret",
        ".streamlit/secrets.toml": "secret",
        "data/private.pdf": "private document",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "initial",
    )
    sha = git("rev-parse", "HEAD")
    (root / "pyproject.toml").write_text("dirty working tree")
    output = tmp_path / "artifact"
    build(root, output, sha)
    metadata, contents = read_archive(output / "release.tar.gz", sha)
    assert metadata["sha"] == sha
    assert contents["pyproject.toml"] == b"original"
    assert ".env" not in contents and ".streamlit/secrets.toml" not in contents
    assert "data/private.pdf" not in contents
    assert "data/sources/cirs.json" in contents


def test_checksum_failure_does_not_touch_current(deployment, tmp_path):
    old = active_release(deployment)
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "release.tar.gz").write_bytes(b"corrupt")
    (incoming / "release.sha256").write_text("wrong")
    with pytest.raises(ValueError, match="checksum"):
        deployment.prepare(incoming, "b" * 40)
    assert (deployment.BASE / "current").resolve() == old
    assert not (deployment.BASE / "releases" / ("b" * 40)).exists()


@pytest.mark.parametrize(
    "members",
    [
        ["release.tar.gz", "release.sha256"],
        ["../escape"],
        ["server.py"],
        ["release.tar.gz"],
    ],
)
def test_restricted_receiver(deployment, monkeypatch, tmp_path, members):
    import receive

    monkeypatch.setattr(receive, "BASE", tmp_path)
    (tmp_path / "incoming").mkdir()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in members:
            member = tarfile.TarInfo(name)
            member.size = 4
            archive.addfile(member, io.BytesIO(b"test"))
    stream.seek(0)
    sha = "a" * 40
    if len(members) == 2:
        receive.receive(sha, stream)
        assert (tmp_path / "incoming" / sha / "release.tar.gz").read_bytes() == b"test"
    else:
        with pytest.raises(ValueError):
            receive.receive(sha, stream)
        assert not (tmp_path / "incoming" / sha).exists()


def test_forced_ssh_command_rejects_shell_injection(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/deploy/atrag-ssh"
    output = tmp_path / "should-not-exist"
    result = subprocess.run(
        ["bash", str(script)],
        env={"SSH_ORIGINAL_COMMAND": f"deploy {'a' * 40}; touch {output}"},
        check=False,
    )
    assert result.returncode == 64
    assert not output.exists()
