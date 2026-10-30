import json

import pytest

from atrag.cli import main
from atrag.ingestion.sources import load_source_registry
from atrag.settings import PROJECT_ROOT


def test_config_check_outputs_safe_summary(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "api-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:db-secret@localhost/atrag")
    main(["config", "check"])
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["source"]["source_id"] == "pt-at-cirs"
    assert summary["source"]["source_url_status"] == "candidate"
    assert summary["settings"]["embedding_dimensions"] == 1536
    assert summary["settings"]["database_configured"] is True
    assert summary["settings"]["openai_configured"] is True
    assert "api-secret" not in output
    assert "db-secret" not in output
    assert "postgresql://" not in output
    assert "Código" in output


def test_invalid_config_reports_no_secrets(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "invalid-secret-dsn")
    with pytest.raises(SystemExit) as error:
        main(["config", "check"])
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "invalid-secret-dsn" not in output
    assert "configuration" in output.lower()


def test_missing_registry_is_reported(capsys, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["config", "check", "--registry", str(tmp_path / "missing.json")])
    assert error.value.code == 2
    assert "registry" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("arguments", [[], ["config"], ["ingest"]])
def test_command_is_required(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.fixture
def snapshot_registry(tmp_path, monkeypatch):
    original = tmp_path / "input.pdf"
    original.write_bytes(b"%PDF-1.7\nCLI fixture\n%%EOF\n")
    source = load_source_registry(PROJECT_ROOT / "data/sources/cirs.json")
    source = source.model_copy(update={"local_path": original, "expected_sha256": None})
    registry = tmp_path / "registry.json"
    registry.write_text(source.model_dump_json())
    monkeypatch.setenv("ATRAG_DATA_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    return registry


def test_snapshot_command_is_local_and_reusable(
    snapshot_registry, tmp_path, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    arguments = ["ingest", "snapshot", "--registry", str(snapshot_registry)]
    main(arguments)
    first = json.loads(capsys.readouterr().out)
    main(arguments)
    second = json.loads(capsys.readouterr().out)
    assert first["snapshot"] == second["snapshot"]
    assert first["run_id"] != second["run_id"]
    assert first["reused"] is False
    assert second["reused"] is True
    assert (tmp_path / "artifacts" / first["snapshot"]["local_path"]).is_file()


def test_snapshot_command_reports_invalid_source(snapshot_registry, capsys):
    snapshot_registry.write_text(
        snapshot_registry.read_text().replace("input.pdf", "missing.pdf")
    )
    with pytest.raises(SystemExit) as error:
        main(["ingest", "snapshot", "--registry", str(snapshot_registry)])
    assert error.value.code == 2
    assert "snapshot" in capsys.readouterr().err.lower()
