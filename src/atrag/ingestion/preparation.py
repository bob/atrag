"""Assign a prepared version after validating the raw pipeline, without editing it."""

import hashlib
import json
from dataclasses import replace


def preparation_identity(manifest) -> str:
    content = manifest.model_dump(
        mode="json", exclude={"snapshot_id", "schema_version"}
    )
    content["raw_snapshot_id"] = manifest.raw_snapshot_id or manifest.snapshot_id
    if manifest.schema_version == 3:
        # Run UUIDs/timestamps and API/cache accounting are evidence of an
        # attempt. They do not change the prepared material or its vectors.
        content["files"] = {
            name: checksum
            for name, checksum in manifest.files.items()
            if not name.endswith("/manifest.json") and name != "embed/report.json"
        }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def prepared_document(document):
    identity = preparation_identity(document.manifest)
    raw = document.snapshot.snapshot_id
    if (
        document.manifest.schema_version >= 2
        and document.manifest.snapshot_id != identity
    ):
        raise ValueError("Package identity differs from prepared snapshots")
    snapshot = document.snapshot.model_copy(
        update={"snapshot_id": identity, "preparation_sha256": identity}
    )
    units = [u.model_copy(update={"snapshot_id": identity}) for u in document.units]
    fragments, identities = [], {}
    for fragment in document.fragments:
        prepared = fragment.model_copy(update={"snapshot_id": identity})
        key = hashlib.sha256(
            prepared.model_dump_json(exclude={"fragment_id", "source"}).encode()
        ).hexdigest()
        identities[fragment.fragment_id] = key
        fragments.append(prepared.model_copy(update={"fragment_id": key}))
    vectors = [
        v.model_copy(
            update={"snapshot_id": identity, "fragment_id": identities[v.fragment_id]}
        )
        for v in document.embeddings
    ]
    return replace(
        document,
        snapshot=snapshot,
        units=units,
        fragments=fragments,
        embeddings=vectors,
        manifest=document.manifest.model_copy(
            update={"snapshot_id": identity, "raw_snapshot_id": raw}
        ),
        reports=_prepared_reports(document.reports, raw, identity),
    )


def _prepared_reports(value, raw, identity):
    if isinstance(value, dict):
        return {
            k: identity
            if k == "snapshot_id" and v == raw
            else _prepared_reports(v, raw, identity)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_prepared_reports(v, raw, identity) for v in value]
    return value
