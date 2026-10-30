import json

import pytest

from atrag.cli import main


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


@pytest.mark.parametrize("arguments", [[], ["config"]])
def test_command_is_required(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
