"""Publish and verify a portable prepared document, without PDF/OCR processing."""

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from atrag.ingestion.package_validation import digest, validate_document
from atrag.ingestion.preparation import preparation_identity, prepared_document
from atrag.ingestion.services.artifacts import write_json
from atrag.ingestion.services.snapshots import _publish_snapshot
from atrag.ingestion.source_files import source_file_names, validate_source_files
from atrag.schemas.ingestion.embeddings import FragmentEmbedding
from atrag.schemas.ingestion.extraction import ExtractionReview, MaterialTables
from atrag.schemas.ingestion.fragments import SearchFragment
from atrag.schemas.ingestion.packages import (
    CompletedStage,
    PackageManifest,
    PreparedDocument,
)
from atrag.schemas.ingestion.snapshots import SourceSnapshot
from atrag.schemas.ingestion.structure import LegalUnit

STAGE_FILES = {
    "parse": ("legal_units.jsonl", "tables.json", "parse/report.json"),
    "chunk": ("fragments.jsonl", "chunk/report.json"),
    "embed": ("embeddings.jsonl", "embed/report.json"),
}
CORE_FILES = {"snapshot.json"} | {
    name
    for stage, files in STAGE_FILES.items()
    for name in (*files, f"{stage}/manifest.json")
}


def _check_stages(files):
    stages = {}
    for stage, names in STAGE_FILES.items():
        manifest = CompletedStage.model_validate_json(files[f"{stage}/manifest.json"])
        if manifest.stage != stage:
            raise ValueError("Wrong prepared stage")
        for name in names:
            if manifest.output_sha256.get(Path(name).name) != digest(files[name]):
                raise ValueError("Prepared artifact checksum mismatch")
        stages[stage] = manifest
    for child, parent in (("chunk", "parse"), ("embed", "chunk")):
        for name in (*STAGE_FILES[parent], f"{parent}/manifest.json"):
            if stages[child].input_sha256.get(Path(name).name) != digest(files[name]):
                raise ValueError("Prepared artifact lineage mismatch")
    return stages


def _check_source(manifest, files):
    snapshot = SourceSnapshot.model_validate_json(files["snapshot.json"])
    names = source_file_names(snapshot)
    if manifest.schema_version == 3:
        if (manifest.source_format, manifest.source_files) != (
            snapshot.source.source_format,
            names,
        ):
            raise ValueError("Package source format or file composition mismatch")
    elif (
        snapshot.collection is not None
        or manifest.source_files is not None
        or manifest.source_format is not None
    ):
        raise ValueError("HTML source requires package schema version 3")
    if not set(names) <= files.keys():
        raise ValueError("Package is missing required files for its source")
    validate_source_files(snapshot, files)
    return snapshot


def _check_provenance(manifest, files, stages):
    snapshot = _check_source(manifest, files)
    raw_id = manifest.raw_snapshot_id or manifest.snapshot_id
    if {s.snapshot_id for s in stages.values()} != {raw_id}:
        raise ValueError("Prepared stages belong to different snapshots")
    expected = (
        snapshot.snapshot_id,
        stages["parse"].parser_version,
        stages["chunk"].chunker_version,
        stages["embed"].model,
        stages["embed"].dimensions,
        stages["embed"].fragment_count,
    )
    actual = (
        raw_id,
        manifest.parser_version,
        manifest.chunker_version,
        manifest.model,
        manifest.dimensions,
        manifest.fragment_count,
    )
    if actual != expected:
        raise ValueError("Package processing versions or identity mismatch")
    _check_reviews(files, snapshot.snapshot_id)
    return snapshot


def _check_reviews(files, snapshot_id):
    for name, raw in files.items():
        if name.startswith("reviews/") and (
            ExtractionReview.model_validate_json(raw).snapshot_id != snapshot_id
        ):
            raise ValueError("Manual review belongs to another snapshot")


def _read_document(manifest, files):
    stages = _check_stages(files)
    snapshot = _check_provenance(manifest, files, stages)
    raw_id = manifest.raw_snapshot_id or manifest.snapshot_id
    document = PreparedDocument(
        manifest=manifest,
        snapshot=snapshot,
        units=[
            LegalUnit.model_validate_json(l)
            for l in files["legal_units.jsonl"].splitlines()
        ],
        tables=MaterialTables.model_validate_json(files["tables.json"]).tables,
        fragments=[
            SearchFragment.model_validate_json(l)
            for l in files["fragments.jsonl"].splitlines()
        ],
        embeddings=[
            FragmentEmbedding.model_validate_json(l)
            for l in files["embeddings.jsonl"].splitlines()
        ],
        reports={
            stage: json.loads(files[f"{stage}/report.json"]) for stage in STAGE_FILES
        },
        files=files,
    )
    validate_document(
        replace(document, manifest=manifest.model_copy(update={"snapshot_id": raw_id}))
    )
    return prepared_document(document)


def _read_files(path, manifest):
    if not CORE_FILES <= manifest.files.keys():
        raise ValueError("Package is missing required files")
    files = {}
    for name, checksum in manifest.files.items():
        target = path.parent / name
        if not target.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("Package file path escapes its directory")
        content = target.read_bytes()
        if digest(content) != checksum:
            raise ValueError("Package file checksum mismatch")
        files[name] = content
    return files


def load_package(path: Path) -> PreparedDocument:
    manifest = PackageManifest.model_validate_json(path.read_bytes())
    return _read_document(manifest, _read_files(path, manifest))


def _collect(parsed, chunked, embedded, data_dir, reviews):
    files = {}
    for stage, path in (("parse", parsed), ("chunk", chunked), ("embed", embedded)):
        files[f"{stage}/manifest.json"] = path.read_bytes()
        for name in STAGE_FILES[stage]:
            files[name] = (path.parent / Path(name).name).read_bytes()
    stages = _check_stages(files)
    raw = data_dir / "raw" / stages["parse"].snapshot_id
    files["snapshot.json"] = (raw / "manifest.json").read_bytes()
    original = SourceSnapshot.model_validate_json(files["snapshot.json"])
    files.update(
        {name: (raw / name).read_bytes() for name in source_file_names(original)}
    )
    # Raw bytes keep their first import manifest. Prepared metadata belongs to
    # this preparation, whose fragments all carry the same checked source.
    snapshot = SourceSnapshot.model_validate_json(files["snapshot.json"])
    fragment = SearchFragment.model_validate_json(
        next(iter(files["fragments.jsonl"].splitlines()), b"{}")
    )
    files["snapshot.json"] = (
        snapshot.model_copy(update={"source": fragment.source})
        .model_dump_json()
        .encode()
    )
    for i, review in enumerate(reviews):
        files[f"reviews/{i}.json"] = review.read_bytes()
    return files, stages


def build_package(
    parsed: Path,
    chunked: Path,
    embedded: Path,
    data_dir: Path,
    output: Path,
    *,
    reviews=(),
) -> Path:
    files, stages = _collect(parsed, chunked, embedded, data_dir, reviews)
    snapshot = SourceSnapshot.model_validate_json(files["snapshot.json"])
    manifest = PackageManifest(
        schema_version=2 if snapshot.collection is None else 3,
        source_format=None
        if snapshot.collection is None
        else snapshot.source.source_format,
        source_files=None
        if snapshot.collection is None
        else source_file_names(snapshot),
        snapshot_id=stages["parse"].snapshot_id,
        raw_snapshot_id=stages["parse"].snapshot_id,
        parser_version=stages["parse"].parser_version,
        chunker_version=stages["chunk"].chunker_version,
        model=stages["embed"].model,
        dimensions=stages["embed"].dimensions,
        fragment_count=stages["embed"].fragment_count,
        files={name: digest(content) for name, content in sorted(files.items())},
    )
    manifest = manifest.model_copy(
        update={"snapshot_id": preparation_identity(manifest)}
    )
    _read_document(manifest, files)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".package-", dir=output.parent) as temporary:
        staging = Path(temporary) / "document"
        staging.mkdir()
        for name, content in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        reused = _publish_snapshot(staging, output)
        if reused:
            existing = load_package(output / "manifest.json").manifest
            if not _same_package(existing, manifest):
                raise ValueError("Destination contains a different prepared package")
    return output / "manifest.json"


def _same_package(existing, manifest):
    if existing.schema_version == manifest.schema_version == 3:
        return preparation_identity(existing) == preparation_identity(manifest)
    return existing == manifest
