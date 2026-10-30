import json
from datetime import UTC
from pathlib import Path

import pytest
from pydantic import ValidationError

from atrag.ingestion.sources import SourceDefinition, load_source_registry
from atrag.settings import PROJECT_ROOT


def test_cirs_registry_preserves_unknown_provenance_and_revision():
    source = load_source_registry(PROJECT_ROOT / "data/sources/cirs.json")
    assert source.source_id == "pt-at-cirs"
    assert source.document_type == "law"
    assert source.publisher == "Autoridade Tributária e Aduaneira"
    assert source.language == "pt"
    assert source.local_path == Path("docs/CIRS.pdf")
    assert source.source_url_status == "candidate"
    assert str(source.source_url).endswith("/Documents/CIRS.pdf")
    assert source.downloaded_at is None
    assert source.publication_date is None
    assert source.revision_publication_date is None
    assert "26/2026" in source.consolidation_label
    assert (
        source.expected_sha256
        == "e8671caafbcda885e5c87d451d8d2ef121b36bc7ab63eb2eb1ab474970b15a72"
    )


@pytest.fixture
def registry_data():
    return json.loads((PROJECT_ROOT / "data/sources/cirs.json").read_text())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_type", "website"),
        ("source_id", ""),
        ("language", "xx"),
        ("source_url", "file:///tmp/source.pdf"),
        ("source_url", None),
        ("source_url_status", "trusted"),
        ("expected_sha256", "abc"),
        ("downloaded_at", "2026-10-30T10:00:00"),
        ("unrecognized_field", "value"),
    ],
)
def test_invalid_registry_is_rejected(registry_data, field, value):
    registry_data[field] = value
    with pytest.raises(ValidationError):
        SourceDefinition.model_validate(registry_data)


def test_confirmed_download_time_requires_timezone(registry_data):
    registry_data["downloaded_at"] = "2026-10-30T10:00:00Z"
    source = SourceDefinition.model_validate(registry_data)
    assert source.downloaded_at.tzinfo == UTC


def test_registry_without_known_url(registry_data):
    registry_data.update(source_url=None, source_url_status="unknown")
    source = SourceDefinition.model_validate(registry_data)
    assert source.source_url is None


def test_malformed_registry_fails(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{ broken", encoding="utf-8")
    with pytest.raises(ValueError):
        load_source_registry(path)
