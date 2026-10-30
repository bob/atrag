"""Publish complete UTF-8 artifacts within a run owned by one writer."""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[BaseModel]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(value.model_dump_json() + "\n")
    temporary.replace(path)
