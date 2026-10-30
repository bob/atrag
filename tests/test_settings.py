import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from atrag.settings import load_settings


def test_explicit_root_environment_precedence_and_paths(project_root, monkeypatch):
    (project_root / ".env").write_text(
        "OPENAI_API_KEY=file-secret\n"
        "DATABASE_URL=postgresql://user:file-password@localhost/atrag\n"
        "ATRAG_DATA_DIR=./artifacts\n"
        "ATRAG_EMBEDDING_BATCH_SIZE=12\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "process-secret")
    monkeypatch.setenv("ATRAG_EMBEDDING_BATCH_SIZE", "7")
    monkeypatch.chdir(project_root.parent)

    settings = load_settings(project_root)

    assert settings.openai_api_key.get_secret_value() == "process-secret"
    assert settings.embedding_batch_size == 7
    assert settings.data_dir == project_root / "artifacts"
    assert "process-secret" not in repr(settings)
    assert "file-password" not in repr(settings)
    assert "file-password" not in settings.model_dump_json()
    with pytest.raises(ValidationError):
        settings.rrf_k = 4


def test_local_defaults_do_not_require_database_or_api_key(project_root):
    settings = load_settings(project_root)

    assert settings.openai_api_key is None
    assert settings.database_url is None
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.embedding_batch_size == 32
    assert settings.chunk_target_tokens == 700
    assert settings.chunk_max_tokens == 1200
    assert settings.rrf_k == 60
    assert settings.data_dir == project_root / "data"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ATRAG_EMBEDDING_BATCH_SIZE", "0"),
        ("ATRAG_RRF_K", "-1"),
        ("ATRAG_CHUNK_TARGET_TOKENS", "0"),
        ("ATRAG_CHUNK_MAX_TOKENS", "699"),
        ("OPENAI_EMBEDDING_DIMENSIONS", "0"),
        ("OPENAI_EMBEDDING_DIMENSIONS", "1537"),
        ("OPENAI_EMBEDDING_DIMENSIONS", "not-an-integer"),
        ("OPENAI_EMBEDDING_MODEL", "unconfigured-model"),
        ("DATABASE_URL", "sqlite:///local.db"),
        ("DATABASE_URL", "postgresql+psycopg://user@localhost/atrag"),
        ("DATABASE_URL", "postgresql://localhost:bad/atrag"),
        ("DATABASE_URL", "postgresql://localhost"),
        ("ATRAG_DATA_DIR", ""),
    ],
)
def test_invalid_configuration_fails_at_startup(project_root, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_settings(project_root)


def test_large_model_custom_dimensions_and_absolute_data_path(
    project_root, monkeypatch
):
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "3072")
    monkeypatch.setenv("ATRAG_DATA_DIR", str(project_root / "absolute"))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DATABASE_URL", "")
    settings = load_settings(project_root)
    assert settings.embedding_dimensions == 3072
    assert settings.data_dir == project_root / "absolute"
    assert settings.openai_api_key is None
    assert settings.database_url is None


def test_missing_dotenv_and_empty_process_value(project_root, monkeypatch):
    (project_root / ".env").write_text("OPENAI_API_KEY=file-secret\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert load_settings(project_root).openai_api_key is None
    (project_root / ".env").unlink()
    assert load_settings(project_root).data_dir == project_root / "data"


def test_settings_can_be_loaded_from_relative_root(project_root, monkeypatch):
    monkeypatch.chdir(project_root.parent)
    assert load_settings(Path(project_root.name)).data_dir == project_root / "data"


def test_chunk_target_can_equal_maximum(project_root, monkeypatch):
    monkeypatch.setenv("ATRAG_CHUNK_TARGET_TOKENS", "1200")
    settings = load_settings(project_root)
    assert settings.chunk_target_tokens == settings.chunk_max_tokens == 1200


def test_loading_settings_does_not_change_the_process_environment(project_root):
    (project_root / ".env").write_text(
        "OPENAI_API_KEY=file-secret\nUNRELATED_SETTING=local-value\n",
        encoding="utf-8",
    )
    before = dict(os.environ)
    assert (
        load_settings(project_root).openai_api_key.get_secret_value() == "file-secret"
    )
    assert dict(os.environ) == before


def test_each_load_reads_its_own_current_dotenv(tmp_path):
    roots = [tmp_path / name for name in ("first", "second")]
    for root, batch_size in zip(roots, (7, 12), strict=True):
        root.mkdir()
        (root / ".env").write_text(f"ATRAG_EMBEDDING_BATCH_SIZE={batch_size}\n")
    assert [load_settings(root).embedding_batch_size for root in roots] == [7, 12]
    (roots[0] / ".env").write_text("ATRAG_EMBEDDING_BATCH_SIZE=9\n")
    assert load_settings(roots[0]).embedding_batch_size == 9


def test_concurrent_settings_loads_are_independent(tmp_path):
    roots = [tmp_path / str(index) for index in range(8)]
    for index, root in enumerate(roots, start=1):
        root.mkdir()
        (root / ".env").write_text(f"ATRAG_EMBEDDING_BATCH_SIZE={index}\n")
    with ThreadPoolExecutor(max_workers=4) as executor:
        settings = list(executor.map(load_settings, roots))
    assert [value.embedding_batch_size for value in settings] == list(range(1, 9))
    assert [value.data_dir for value in settings] == [root / "data" for root in roots]


def test_dotenv_expansion_keeps_process_precedence_and_ignores_bare_keys(
    project_root, monkeypatch
):
    (project_root / ".env").write_text(
        "ATRAG_EMBEDDING_BATCH_SIZE=12\n"
        "ATRAG_RRF_K=${ATRAG_EMBEDDING_BATCH_SIZE}\n"
        "ATRAG_DATA_DIR=${REVIEW_OUTPUT_DIR:-artifacts}/cirs\n"
        "ATRAG_CHUNK_TARGET_TOKENS\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ATRAG_EMBEDDING_BATCH_SIZE", "7")
    monkeypatch.setenv("REVIEW_OUTPUT_DIR", "external")
    settings = load_settings(project_root)
    assert settings.embedding_batch_size == settings.rrf_k == 7
    assert settings.data_dir == project_root / "external/cirs"
    assert settings.chunk_target_tokens == 700


@pytest.mark.parametrize("disabled", ["1", "TRUE", "yes", "0"])
def test_dotenv_disabled_flag_is_preserved(project_root, monkeypatch, disabled):
    (project_root / ".env").write_text("ATRAG_RRF_K=77\n")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", disabled)
    expected = 77 if disabled == "0" else 60
    assert load_settings(project_root).rrf_k == expected
