"""Deployment failures must preserve the active release and application data."""

import importlib.util
import io
import json
import subprocess
import tarfile
import tomllib
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest


@pytest.fixture
def deployment(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/deploy"
    monkeypatch.syspath_prepend(str(source))
    spec = importlib.util.spec_from_file_location(
        "atrag_deploy_server", source / "server.py"
    )
    assert spec is not None and spec.loader is not None
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
    "public_url", ["https://tax.example.com", "https://research.example.org"]
)
@pytest.mark.parametrize("recovering", [False, True])
def test_health_accepts_proxy_that_rejects_default_python_agent(
    deployment, monkeypatch, public_url, recovering
):
    active = active_release(deployment)
    schema_release = release(deployment, "b")[0] if recovering else active
    checked, calls = [], []

    class Response(io.BytesIO):
        status = 200

    def proxy(request, timeout):
        url = getattr(request, "full_url", request)
        agent = (
            request.get_header("User-agent", "")
            if hasattr(request, "get_header")
            else ""
        )
        if url.startswith("https:") and (
            not agent or agent.startswith("Python-urllib")
        ):
            raise HTTPError(url, 403, "Forbidden", Message(), None)
        checked.append(url)
        return Response(b"ok")

    def control(*args):
        calls.append(args)
        if args[-1] == "config":
            return json.dumps({"public_url": public_url})

    monkeypatch.setattr(deployment, "urlopen", proxy)
    monkeypatch.setattr(deployment, "control", control)
    monkeypatch.setattr(deployment.time, "sleep", lambda *args: None)
    deployment.health(active, schema_release=schema_release)
    assert checked == [
        "http://127.0.0.1:8501/_stcore/health",
        f"{public_url}/_stcore/health",
    ]
    assert ("run", schema_release.name, "config") in calls
    assert ("run", schema_release.name, "smoke") in calls
    assert ("verify-sha", active.name) in calls


def test_health_failure_reports_http_error(deployment, monkeypatch):
    active = active_release(deployment)

    def reject(request, timeout):
        raise HTTPError(request.full_url, 403, "Forbidden", Message(), None)

    monkeypatch.setattr(deployment, "urlopen", reject)
    monkeypatch.setattr(deployment.time, "sleep", lambda *args: None)
    with pytest.raises(RuntimeError, match="HTTP Error 403: Forbidden"):
        deployment.health(active, external=False)


@pytest.fixture
def provision(deployment, monkeypatch, tmp_path):
    import setup_config

    shared = tmp_path / "shared"
    shared.mkdir()
    hba = tmp_path / "pg_hba.conf"
    hba.write_text("# Existing database rules\n")
    queries = []

    def sql(query, database="postgres"):
        queries.append(query)
        if query == "SHOW hba_file":
            return str(hba)
        if query.startswith("SELECT count(*) FROM pg_hba_file_rules"):
            return "0"
        return ""

    import grp

    monkeypatch.setattr(setup_config, "SHARED", shared)
    monkeypatch.setattr(setup_config, "sql", sql)
    monkeypatch.setattr(setup_config.os, "geteuid", lambda: 0)
    monkeypatch.setattr(setup_config.os, "fchown", lambda *args: None)
    monkeypatch.setattr(grp, "getgrnam", lambda name: SimpleNamespace(gr_gid=0))
    return setup_config, shared, queries


def provision_input(monkeypatch, provision, public_host):
    module, _, _ = provision
    values = {
        "openai_api_key": "example-key",
        "client_id": "example-client",
        "client_secret": "example-secret",
        "public_host": public_host,
    }
    monkeypatch.setattr(module.sys, "stdin", io.StringIO(json.dumps(values)))
    return module


def test_provision_configures_oauth_and_service_for_supplied_host(
    provision, monkeypatch
):
    module = provision_input(monkeypatch, provision, "tax.example.com")
    module.main()
    _, shared, queries = provision
    auth = tomllib.loads((shared / "secrets.toml").read_text())
    assert auth["auth"]["redirect_uri"] == "https://tax.example.com/oauth2callback"
    assert (shared / "service.env").read_text() == (
        "STREAMLIT_BROWSER_SERVER_ADDRESS=tax.example.com\n"
    )
    assert (shared / "service.env").stat().st_mode & 0o777 == 0o640
    assert "CREATE DATABASE atrag_prod OWNER atrag_prod;" in queries


@pytest.mark.parametrize(
    "host",
    [None, "", "https://tax.example.com", "tax.example.com/path", "a\nb", "$(id)"],
)
def test_provision_rejects_invalid_host_before_writes(provision, monkeypatch, host):
    module = provision_input(monkeypatch, provision, host)
    with pytest.raises(ValueError):
        module.main()
    _, shared, queries = provision
    assert queries == []
    assert list(shared.iterdir()) == []


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
