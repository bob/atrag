"""Check portable source bytes without running a format parser on the server."""

import hashlib

from atrag.schemas.ingestion.collections import HTMLCollection


def source_file_names(snapshot):
    if snapshot.collection is None:
        return ["source.pdf"]
    return [
        "collection.json",
        *sorted(str(p.local_path) for p in snapshot.collection.pages),
    ]


def validate_source_files(snapshot, files):
    if snapshot.collection is None:
        pdf = files["source.pdf"]
        if not pdf.startswith(b"%PDF-"):
            raise ValueError("Package PDF signature mismatch")
        if len(pdf) != snapshot.size_bytes:
            raise ValueError("Package PDF size mismatch")
        if hashlib.sha256(pdf).hexdigest() != snapshot.source_sha256:
            raise ValueError("Package snapshot checksum mismatch")
        return
    collection = HTMLCollection.model_validate_json(files["collection.json"])
    if collection != snapshot.collection:
        raise ValueError("Package collection differs from snapshot")
    for page in collection.pages:
        raw = files[str(page.local_path)]
        if (hashlib.sha256(raw).hexdigest(), len(raw)) != (
            page.sha256,
            page.size_bytes,
        ):
            raise ValueError("Package HTML checksum or size mismatch")
