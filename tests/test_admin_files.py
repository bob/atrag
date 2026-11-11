"""Downloads reject paths and symlinks outside trusted source storage."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from atrag.admin.files import checked_path, download_source, source_available
from atrag.admin.schemas import DocumentVersion
from atrag.interface.admin_views import date_label, size_label


@pytest.fixture
def version():
    return DocumentVersion(
        "doc",
        "b" * 64,
        "a" * 64,
        "Document",
        "pdf",
        None,
        f"raw/{'a' * 64}/source.pdf",
        100,
        datetime.now(UTC),
        datetime.now(UTC),
        False,
    )


def test_source_paths_are_confined_before_any_read(tmp_path, version):
    for path in ("../secret.pdf", "/etc/passwd", "raw/other/source.pdf"):
        changed = replace(version, local_path=path)
        assert not source_available(tmp_path, changed)
        with pytest.raises(ValueError):
            download_source(tmp_path, changed)
    with pytest.raises(ValueError):
        checked_path(tmp_path, "../secret")
    assert not source_available(tmp_path, replace(version, source_sha256="../"))
    assert not source_available(tmp_path, replace(version, source_format="docx"))


def test_symlink_cannot_escape_storage(tmp_path, version):
    storage = tmp_path / "storage"
    directory = storage / "raw" / version.source_sha256
    directory.mkdir(parents=True)
    (directory / "source.pdf").symlink_to(tmp_path / "private.pdf")
    assert not source_available(storage, version)
    with pytest.raises(ValueError):
        download_source(storage, version)


@pytest.mark.parametrize(
    "size, label",
    [
        (0, "0.0 B"),
        (1, "1.0 B"),
        (1023, "1023.0 B"),
        (1024, "1.0 KiB"),
        (1024**2, "1.0 MiB"),
        (1024**3, "1.0 GiB"),
    ],
)
def test_source_size_is_readable(size, label):
    assert size_label(size) == label


def test_unknown_login_date_is_explicit():
    assert date_label(None, "ru") == "Нет данных"
