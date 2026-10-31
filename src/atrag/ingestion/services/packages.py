"""Publish and verify a portable prepared document, without PDF/OCR processing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from atrag.ingestion.package_validation import digest, validate_document
from atrag.ingestion.services.artifacts import write_json
from atrag.ingestion.services.snapshots import _publish_snapshot
from atrag.schemas.ingestion.embeddings import FragmentEmbedding
from atrag.schemas.ingestion.extraction import ExtractionReview, ExtractionTables
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
CORE_FILES = {"source.pdf", "snapshot.json"} | {
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


def _read_document(manifest, files):
    stages = _check_stages(files)
    snapshot = SourceSnapshot.model_validate_json(files["snapshot.json"])
    pdf = files["source.pdf"]
    if not pdf.startswith(b"%PDF-"):
        raise ValueError("Package PDF signature mismatch")
    if len(pdf) != snapshot.size_bytes:
        raise ValueError("Package PDF size mismatch")
    if digest(pdf) != snapshot.snapshot_id:
        raise ValueError("Package snapshot checksum mismatch")
    if {s.snapshot_id for s in stages.values()} != {manifest.snapshot_id}:
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
        manifest.snapshot_id,
        manifest.parser_version,
        manifest.chunker_version,
        manifest.model,
        manifest.dimensions,
        manifest.fragment_count,
    )
    if actual != expected:
        raise ValueError("Package processing versions or identity mismatch")
    for name, raw in files.items():
        if name.startswith("reviews/") and (
            ExtractionReview.model_validate_json(raw).snapshot_id
            != snapshot.snapshot_id
        ):
            raise ValueError("Manual review belongs to another snapshot")
    document = PreparedDocument(
        manifest=manifest,
        snapshot=snapshot,
        units=[
            LegalUnit.model_validate_json(l)
            for l in files["legal_units.jsonl"].splitlines()
        ],
        tables=ExtractionTables.model_validate_json(files["tables.json"]).tables,
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
    validate_document(document)
    return document


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
    files.update(
        {
            "source.pdf": (raw / "source.pdf").read_bytes(),
            "snapshot.json": (raw / "manifest.json").read_bytes(),
        }
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
    manifest = PackageManifest(
        snapshot_id=stages["parse"].snapshot_id,
        parser_version=stages["parse"].parser_version,
        chunker_version=stages["chunk"].chunker_version,
        model=stages["embed"].model,
        dimensions=stages["embed"].dimensions,
        fragment_count=stages["embed"].fragment_count,
        files={name: digest(content) for name, content in sorted(files.items())},
    )
    _read_document(manifest, files)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".package-", dir=output.parent) as temporary:
        staging = Path(temporary) / "document"
        staging.mkdir()
        for name, content in files.items():
            target = staging / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(content)
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        reused = _publish_snapshot(staging, output)
        if reused and load_package(output / "manifest.json").manifest != manifest:
            raise ValueError("Destination contains a different prepared package")
    return output / "manifest.json"
