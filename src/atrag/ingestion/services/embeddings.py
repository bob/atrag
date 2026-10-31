"""Validate a completed chunk run and checkpoint each paid embedding batch."""

import hashlib
import time
from importlib.metadata import version

import tiktoken

from atrag.ingestion.api.embeddings import request_embeddings
from atrag.ingestion.embeddings import input_batches, match_response, prepare_input
from atrag.ingestion.services.artifacts import write_json, write_jsonl
from atrag.ingestion.services.runs import record_run
from atrag.ingestion.services.snapshots import read_snapshot
from atrag.schemas.ingestion.embeddings import (
    ChunkManifest,
    EmbeddingRun,
    EmbeddingSpace,
    FragmentEmbedding,
)
from atrag.schemas.ingestion.fragments import SearchFragment
from atrag.schemas.ingestion.snapshots import SourceSnapshot

ARTIFACTS = ("fragments.jsonl", "report.json")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _load_chunked(path, settings, source, run):
    import json

    raw = {name: (path.parent / name).read_bytes() for name in ARTIFACTS}
    raw["manifest.json"] = path.read_bytes()
    run.manifest["input_sha256"] = {
        name: _digest(content) for name, content in raw.items()
    }
    parent = ChunkManifest.model_validate_json(raw["manifest.json"])
    for name in ARTIFACTS:
        if parent.output_sha256.get(name) != _digest(raw[name]):
            raise ValueError("Chunk artifact checksum mismatch")
    if (
        source.expected_sha256 is not None
        and source.expected_sha256 != parent.snapshot_id
    ):
        raise ValueError("Chunk snapshot does not match the registry")
    directory = settings.data_dir / "raw" / parent.snapshot_id
    snapshot = SourceSnapshot.model_validate_json(
        (directory / "manifest.json").read_bytes()
    )
    read_snapshot(directory, parent.snapshot_id, snapshot.size_bytes)
    fragments = [
        SearchFragment.model_validate_json(line)
        for line in raw["fragments.jsonl"].splitlines()
    ]
    _validate_fragments(fragments, parent, snapshot)
    return parent, fragments, json.loads(raw["report.json"])


def _validate_fragments(fragments, parent, snapshot):
    if len(fragments) != parent.fragment_count:
        raise ValueError("Chunk fragment count mismatch")
    if len({f.fragment_id for f in fragments}) != len(fragments):
        raise ValueError("Duplicate fragment identity")
    for fragment in fragments:
        if (
            fragment.snapshot_id != parent.snapshot_id
            or fragment.source != snapshot.source
        ):
            raise ValueError("Fragment provenance does not match its snapshot")


def _prepare(fragments):
    encoding = tiktoken.get_encoding("cl100k_base")
    inputs = {}
    for fragment in fragments:
        item = prepare_input(
            fragment.embedding_text, lambda text: len(encoding.encode_ordinary(text))
        )
        if (
            item.input_hash != fragment.embedding_input_hash
            or item.token_count != fragment.token_count
        ):
            raise ValueError("Fragment embedding hash or token count mismatch")
        inputs[item.input_hash] = item
    return inputs


def _fill_cache(store, client, space, inputs, batch_size, report, report_path, sleep):
    records = store.get(list(inputs))
    report["cache_hits"] = len(records)
    missing = [item for h, item in inputs.items() if h not in records]
    for batch in input_batches(missing, batch_size):
        raw = request_embeddings(client, space, batch, report, sleep)
        # Usage survives a malformed response or a failed cache write.
        write_json(report_path, report)
        vectors = match_response(raw, space, batch)
        store.save(vectors)
        records.update({r.input_hash: r for r in vectors})
        report["embedded_inputs"] += len(vectors)
        write_json(report_path, report)
    return records


def embed_chunked(
    chunked, settings, source, *, cache, client=None, sleep=time.sleep
) -> EmbeddingRun:
    space = EmbeddingSpace(
        model=settings.embedding_model, dimensions=settings.embedding_dimensions
    )
    report = {
        "schema_version": 1,
        **space.model_dump(),
        "requests": 0,
        "retries": 0,
        "cache_hits": 0,
        "embedded_inputs": 0,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
    with record_run(
        settings.data_dir,
        stage="embed",
        **space.model_dump(),
        batch_size=settings.embedding_batch_size,
        tokenizer_version=version("tiktoken"),
        openai_version=version("openai"),
        token_encoding="cl100k_base",
    ) as run:
        report_path = run.directory / "report.json"
        try:
            parent, fragments, chunk_report = _load_chunked(
                chunked, settings, source, run
            )
            report.update(
                snapshot_id=parent.snapshot_id,
                fragment_count=len(fragments),
                chunk_report=chunk_report,
            )
            inputs = _prepare(fragments)
            report["unique_inputs"] = len(inputs)
            with cache.writer(space) as store:
                records = _fill_cache(
                    store,
                    client,
                    space,
                    inputs,
                    settings.embedding_batch_size,
                    report,
                    report_path,
                    sleep,
                )
            output = [
                FragmentEmbedding(
                    **records[f.embedding_input_hash].model_dump(),
                    snapshot_id=f.snapshot_id,
                    fragment_id=f.fragment_id,
                )
                for f in fragments
            ]
            write_jsonl(run.directory / "embeddings.jsonl", output)
        finally:
            write_json(report_path, report)
        run.manifest.update(
            snapshot_id=parent.snapshot_id,
            fragment_count=len(fragments),
            output_sha256={
                name: _digest((run.directory / name).read_bytes())
                for name in ("embeddings.jsonl", "report.json")
            },
        )
    return EmbeddingRun(run.run_id, run.directory, len(fragments))
