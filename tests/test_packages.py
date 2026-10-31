import json
import shutil

import pytest

from atrag.ingestion.services.packages import build_package, load_package


def test_package_is_portable_reproducible_and_preserves_inputs(
    prepared_document, tmp_path
):
    data, parsed, chunked, embedded = prepared_document
    before = {
        p: p.read_bytes()
        for parent in (parsed.parent, chunked.parent, embedded.parent)
        for p in parent.iterdir()
    }
    output = tmp_path / "nested" / "new" / "package"
    path = build_package(parsed, chunked, embedded, data, output)
    document = load_package(path)
    assert len(document.fragments) == 3
    assert document.manifest.model == "text-embedding-3-small"
    assert document.manifest.dimensions == 1536
    assert document.snapshot.source.language == "pt"
    assert document.reports["chunk"]["extraction_issues"]
    second = build_package(parsed, chunked, embedded, data, tmp_path / "second")
    assert path.read_bytes() == second.read_bytes()
    relocated = tmp_path / "server"
    shutil.copytree(output, relocated)
    assert load_package(relocated / "manifest.json").fragments == document.fragments
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize(
    "name",
    [
        "source.pdf",
        "legal_units.jsonl",
        "fragments.jsonl",
        "embeddings.jsonl",
        "tables.json",
        "chunk/report.json",
    ],
)
def test_package_checks_every_checksum(prepared_document, tmp_path, name):
    path = build_package(
        *prepared_document[1:], prepared_document[0], tmp_path / "package"
    )
    (path.parent / name).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        load_package(path)


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/tmp/outside",
        "a/../../outside",
        "a\\outside",
        "a//b",
        "./a",
        ".",
        "",
    ],
)
def test_manifest_paths_cannot_escape_package(prepared_document, tmp_path, name):
    path = build_package(
        *prepared_document[1:], prepared_document[0], tmp_path / "package"
    )
    manifest = json.loads(path.read_bytes())
    manifest["files"][name] = "a" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path"):
        load_package(path)


def test_package_rejects_mismatched_lineage_before_publication(
    prepared_document, tmp_path
):
    data, parsed, chunked, embedded = prepared_document
    manifest = json.loads(embedded.read_bytes())
    manifest["input_sha256"]["manifest.json"] = "a" * 64
    embedded.write_text(json.dumps(manifest))
    output = tmp_path / "package"
    with pytest.raises(ValueError, match="lineage"):
        build_package(parsed, chunked, embedded, data, output)
    assert not output.exists()


def test_package_reuses_identical_destination(prepared_document, tmp_path):
    data, parsed, chunked, embedded = prepared_document
    first = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    assert build_package(parsed, chunked, embedded, data, first.parent) == first


@pytest.mark.parametrize("damage", ["signature", "size", "hash"])
def test_snapshot_bytes_are_checked_before_publication(
    prepared_document, tmp_path, damage
):
    data, parsed, chunked, embedded = prepared_document
    pdf = next((data / "raw").glob("*/source.pdf"))
    raw = pdf.read_bytes()
    pdf.write_bytes(
        b"broken"
        if damage == "size"
        else b"xxxxx" + raw[5:]
        if damage == "signature"
        else raw.replace(b"Synthetic", b"Changed!!")
    )
    with pytest.raises(ValueError, match="PDF|checksum"):
        build_package(parsed, chunked, embedded, data, tmp_path / "package")


def test_manual_review_is_carried_and_checked(prepared_document, tmp_path):
    data, parsed, chunked, embedded = prepared_document
    snapshot_id = json.loads(parsed.read_bytes())["snapshot_id"]
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"snapshot_id": snapshot_id, "tables": []}))
    path = build_package(
        parsed, chunked, embedded, data, tmp_path / "package", reviews=[review]
    )
    assert load_package(path).files["reviews/0.json"] == review.read_bytes()
    review.write_text(json.dumps({"snapshot_id": "b" * 64, "tables": []}))
    with pytest.raises(ValueError, match="review belongs"):
        build_package(
            parsed, chunked, embedded, data, tmp_path / "wrong", reviews=[review]
        )


def test_missing_file_and_symlink_are_rejected(prepared_document, tmp_path):
    path = build_package(
        *prepared_document[1:], prepared_document[0], tmp_path / "package"
    )
    target = path.parent / "source.pdf"
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="path"):
        load_package(path)
    manifest = json.loads(path.read_bytes())
    del manifest["files"]["source.pdf"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="required files"):
        load_package(path)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("stage", "Wrong prepared stage"),
        ("output", "checksum"),
        ("snapshot", "different snapshots"),
        ("versions", "versions"),
    ],
)
def test_stage_boundaries_are_checked(prepared_document, tmp_path, change, reason):
    data, parsed, chunked, embedded = prepared_document
    if change in ("stage", "output"):
        manifest = json.loads(parsed.read_bytes())
        if change == "stage":
            manifest["stage"] = "chunk"
        else:
            manifest["output_sha256"]["tables.json"] = "b" * 64
        parsed.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=reason):
            build_package(parsed, chunked, embedded, data, tmp_path / "package")
        return
    path = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    manifest = json.loads(path.read_bytes())
    manifest["snapshot_id" if change == "snapshot" else "parser_version"] = "b" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=reason):
        load_package(path)


@pytest.mark.parametrize("name", ["/tmp/file", "../file", "a\\file", "a//file", "."])
def test_manifest_type_rejects_unsafe_paths(prepared_document, tmp_path, name):
    from atrag.schemas.ingestion.packages import PackageManifest

    path = build_package(
        *prepared_document[1:], prepared_document[0], tmp_path / "package"
    )
    manifest = json.loads(path.read_bytes())
    manifest["files"] = {name: "a" * 64}
    with pytest.raises(ValueError, match="path"):
        PackageManifest.model_validate(manifest)


def test_existing_package_cannot_be_replaced_by_different_content(
    prepared_document, tmp_path
):
    from fixtures.prepared import refresh_manifest

    data, parsed, chunked, embedded = prepared_document
    path = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    before = path.read_bytes()
    (embedded.parent / "report.json").write_text('{"different": true}')
    refresh_manifest(embedded)
    with pytest.raises(ValueError, match="different prepared package"):
        build_package(parsed, chunked, embedded, data, path.parent)
    assert path.read_bytes() == before


def test_snapshot_size_matches_even_when_pdf_checksum_is_valid(
    prepared_document, tmp_path
):
    data, parsed, chunked, embedded = prepared_document
    snapshot = next((data / "raw").glob("*/manifest.json"))
    manifest = json.loads(snapshot.read_bytes())
    manifest["size_bytes"] += 1
    snapshot.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="PDF size"):
        build_package(parsed, chunked, embedded, data, tmp_path / "package")
