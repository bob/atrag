"""Small prepared corpus; vectors are synthetic and never call a live model."""

import hashlib
import json
from datetime import date
from pathlib import Path

from atrag.ingestion.parse_cirs import parse_cirs
from atrag.ingestion.services.artifacts import write_json, write_jsonl
from atrag.ingestion.services.chunks import chunk_parsed
from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.schemas.ingestion.embeddings import FragmentEmbedding
from atrag.schemas.ingestion.extraction import (
    ExtractedPage,
    ExtractedTable,
    TablePart,
    TextBlock,
)
from atrag.schemas.ingestion.fragments import ChunkParameters, SearchFragment


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def refresh_manifest(path, group="output_sha256"):
    manifest = json.loads(path.read_bytes())
    manifest[group] = {
        name: digest((path.parent / name).read_bytes()) for name in manifest[group]
    }
    write_json(path, manifest)


def prepare_document(tmp_path, *, second_paragraph=None, source_id="pt-at-cirs"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "local"
    pdf = tmp_path / "original.pdf"
    pdf.write_bytes(
        b"%PDF-1.7\nSynthetic corpus " + tmp_path.name.encode() + b"\n%%EOF\n"
    )
    source = load_source_registry(Path("data/sources/cirs.json")).model_copy(
        update={
            "local_path": pdf,
            "source_id": source_id,
            "expected_sha256": None,
            "revision_publication_date": date(2026, 6, 3),
        }
    )
    snapshot = import_snapshot(source, data).snapshot
    texts = [
        source.document_title,
        "Artigo 16.º",
        "Residência fiscal",
        "1 - São residentes as pessoas que permaneçam em território português.",
        "a) Exceto nos casos previstos no artigo 68.º-A.",
        "2 - Não são residentes os trabalhadores abrangidos pela exceção.",
        "Artigo 68.º-A",
        "Taxa adicional de solidariedade",
        "1 - A taxa aplica-se ao rendimento coletável.",
    ]
    if second_paragraph is not None:
        texts[5] = second_paragraph
    part = TablePart(
        part_id="rates:1",
        table_id="rates",
        page_number=1,
        bbox=(85, 290, 500, 330),
        rows=[["Rendimento", "Taxa"], ["1000", "2,5"]],
    )
    table = ExtractedTable(
        table_id="rates",
        parts=[part],
        header_rows=[["Rendimento", "Taxa"]],
        rows=[["1000", "2,5"]],
        context="Taxas",
        review_status="verified",
        review_note="Synthetic checked table",
    )
    structure = parse_cirs(
        [
            ExtractedPage(
                snapshot_id=snapshot.snapshot_id,
                page_number=1,
                width=595,
                height=842,
                raw_text="\n".join(texts),
                blocks=[
                    TextBlock(
                        text=t,
                        bbox=(85, 100 + i * 20, 500, 110 + i * 20),
                        font_names=[
                            "Arial-BoldMT" if i in (0, 1, 2, 6, 7) else "ArialMT"
                        ],
                    )
                    for i, t in enumerate(texts)
                ],
                tables=[part],
                images=[],
            )
        ],
        [table],
    )
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    write_jsonl(parsed / "legal_units.jsonl", structure.units)
    write_json(parsed / "tables.json", {"tables": [table.model_dump(mode="json")]})
    write_json(
        parsed / "report.json",
        {
            "excluded_block_count": 0,
            "extraction_issues": [{"page_number": 1, "issues": ["image_needs_review"]}],
        },
    )
    write_json(
        parsed / "manifest.json",
        {
            "stage": "parse",
            "status": "succeeded",
            "parser_version": "2",
            "snapshot_id": snapshot.snapshot_id,
            "output_sha256": {
                n: "" for n in ("legal_units.jsonl", "tables.json", "report.json")
            },
        },
    )
    refresh_manifest(parsed / "manifest.json")
    chunk = chunk_parsed(parsed / "manifest.json", data, source, ChunkParameters())
    fragments = [
        SearchFragment.model_validate_json(l)
        for l in (chunk.directory / "fragments.jsonl").read_bytes().splitlines()
    ]
    embedded = tmp_path / "embedded"
    embedded.mkdir()
    input_indices = {
        value: index
        for index, value in enumerate(
            dict.fromkeys(f.embedding_input_hash for f in fragments)
        )
    }
    write_jsonl(
        embedded / "embeddings.jsonl",
        [
            FragmentEmbedding(
                model="text-embedding-3-small",
                dimensions=1536,
                input_hash=f.embedding_input_hash,
                embedding=[
                    float(input_indices[f.embedding_input_hash] == j)
                    for j in range(1536)
                ],
                snapshot_id=f.snapshot_id,
                fragment_id=f.fragment_id,
            )
            for f in fragments
        ],
    )
    write_json(
        embedded / "report.json",
        {"chunk_report": json.loads((chunk.directory / "report.json").read_bytes())},
    )
    write_json(
        embedded / "manifest.json",
        {
            "stage": "embed",
            "status": "succeeded",
            "snapshot_id": snapshot.snapshot_id,
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "fragment_count": len(fragments),
            "input_sha256": {
                n: digest((chunk.directory / n).read_bytes())
                for n in ("manifest.json", "fragments.jsonl", "report.json")
            },
            "output_sha256": {n: "" for n in ("embeddings.jsonl", "report.json")},
        },
    )
    refresh_manifest(embedded / "manifest.json")
    return (
        data,
        parsed / "manifest.json",
        chunk.directory / "manifest.json",
        embedded / "manifest.json",
    )


def prepare_revision(prepared, output, *, source_updates=None):
    """Change preparation with identical source bytes and cached synthetic vectors."""
    from atrag.ingestion.services.packages import build_package
    from atrag.schemas.ingestion.sources import SourceDefinition

    data, parsed, _chunked, embedded = prepared
    raw = next((data / "raw").glob("*/manifest.json"))
    source = SourceDefinition.model_validate(json.loads(raw.read_bytes())["source"])
    source = SourceDefinition.model_validate(
        {**source.model_dump(), **(source_updates or {})}
    )
    chunk = chunk_parsed(parsed, data, source, ChunkParameters(target_tokens=701))
    fragments = [
        SearchFragment.model_validate_json(line)
        for line in (chunk.directory / "fragments.jsonl").read_bytes().splitlines()
    ]
    vectors = {
        v["input_hash"]: v
        for v in (
            json.loads(line)
            for line in (embedded.parent / "embeddings.jsonl").read_bytes().splitlines()
        )
    }
    output.mkdir(parents=True)
    write_jsonl(
        output / "embeddings.jsonl",
        [
            FragmentEmbedding.model_validate(
                {
                    **vectors.get(f.embedding_input_hash, next(iter(vectors.values()))),
                    "input_hash": f.embedding_input_hash,
                    "fragment_id": f.fragment_id,
                }
            )
            for f in fragments
        ],
    )
    (output / "report.json").write_bytes((embedded.parent / "report.json").read_bytes())
    manifest = json.loads(embedded.read_bytes())
    manifest["input_sha256"] = {
        n: digest((chunk.directory / n).read_bytes())
        for n in ("manifest.json", "fragments.jsonl", "report.json")
    }
    write_json(output / "manifest.json", manifest)
    refresh_manifest(output / "manifest.json")
    return build_package(
        parsed,
        chunk.directory / "manifest.json",
        output / "manifest.json",
        data,
        output / "package",
    )


def index_legacy_document(package, engine):
    """Reproduce the pre-1.2 raw-hash corpus for migration compatibility tests."""
    from dataclasses import replace

    from atrag.ingestion.services.index import _insert_corpus
    from atrag.ingestion.services.packages import load_package
    from atrag.schemas.ingestion.snapshots import SourceSnapshot
    from atrag.schemas.ingestion.structure import LegalUnit

    document = load_package(package)
    snapshot = SourceSnapshot.model_validate_json(document.files["snapshot.json"])
    original = replace(
        document,
        snapshot=snapshot,
        manifest=document.manifest.model_copy(
            update={"snapshot_id": snapshot.snapshot_id}
        ),
        units=[
            LegalUnit.model_validate_json(l)
            for l in document.files["legal_units.jsonl"].splitlines()
        ],
        fragments=[
            SearchFragment.model_validate_json(l)
            for l in document.files["fragments.jsonl"].splitlines()
        ],
        embeddings=[
            FragmentEmbedding.model_validate_json(l)
            for l in document.files["embeddings.jsonl"].splitlines()
        ],
        reports={
            stage: json.loads(document.files[f"{stage}/report.json"])
            for stage in ("parse", "chunk", "embed")
        },
    )
    with engine.begin() as connection:
        _insert_corpus(connection, original)
    return snapshot.snapshot_id
