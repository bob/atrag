import json

import pytest

from atrag.cli import main


@pytest.mark.parametrize("mode", ["fts", "hybrid"])
def test_search_rejects_obsolete_modes_before_configuration(mode, monkeypatch, capsys):
    monkeypatch.setattr(
        "atrag.cli.load_settings",
        lambda: pytest.fail("Removed modes must be rejected before configuration"),
    )
    with pytest.raises(SystemExit) as error:
        main(["search", "residência", "--snapshot", "b" * 64, "--mode", mode])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_package_command_uses_prepared_inputs(
    prepared_document, tmp_path, monkeypatch, capsys
):
    data, parsed, chunked, embedded = prepared_document
    monkeypatch.setenv("ATRAG_DATA_DIR", str(data))
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "package"
    main(
        [
            "ingest",
            "package",
            "--parsed",
            str(parsed),
            "--chunked",
            str(chunked),
            "--embedded",
            str(embedded),
            "--output",
            str(path),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["manifest"] == str(path / "manifest.json")
    assert result["fragment_count"] == 3


@pytest.mark.parametrize(
    "args,required",
    [
        (["ingest", "package"], "--parsed"),
        (["ingest", "index"], "--package"),
        (["search", "CIRS 16"], "--snapshot"),
    ],
)
def test_corpus_commands_require_inputs(args, required, capsys, monkeypatch):
    monkeypatch.setattr(
        "atrag.cli.load_settings",
        lambda: pytest.fail("Missing arguments must be rejected before configuration"),
    )
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert required in capsys.readouterr().err


@pytest.mark.parametrize(
    "args,message",
    [
        (["ingest", "index", "--package", "missing-secret.json"], "Indexing failed"),
        (
            ["search", "CIRS 16", "--snapshot", "b" * 64, "--language", "pt"],
            "Search failed",
        ),
    ],
)
def test_database_command_failures_do_not_leak_secrets(
    args, message, monkeypatch, capsys
):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert message in output
    assert "secret" not in output
