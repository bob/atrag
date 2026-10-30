"""Read registered source definitions from local JSON files."""

from pathlib import Path

from atrag.schemas.ingestion.sources import SourceDefinition


def load_source_registry(path: Path) -> SourceDefinition:
    return SourceDefinition.model_validate_json(path.read_text(encoding="utf-8"))
