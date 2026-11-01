"""Exact SQL addresses or cosine similarity within one indexed snapshot."""

import time

import tiktoken
from sqlalchemy import select

from atrag.db_models import FragmentRow, LegalUnitRow, SnapshotRow
from atrag.ingestion.api.embeddings import request_embeddings
from atrag.ingestion.embeddings import match_response, prepare_input
from atrag.retrieval.context import assemble_hits
from atrag.retrieval.parts import hit_coverage
from atrag.retrieval.query import (
    merge_filters,
    normalize_address,
)
from atrag.retrieval.query import (
    query_language as detect_query_language,
)
from atrag.schemas.ingestion.embeddings import EmbeddingSpace
from atrag.schemas.ingestion.fragments import SearchFragment
from atrag.schemas.retrieval import Ranking, SearchRequest, SearchResult

# Initial exact-search window; tune only after evaluating retrieval quality.
CANDIDATE_LIMIT = 50


def _query_vector(query, snapshot, client, usage, encoding):
    space = EmbeddingSpace(
        model=snapshot["embedding_model"], dimensions=snapshot["embedding_dimensions"]
    )
    item = prepare_input(query, lambda text: len(encoding.encode_ordinary(text)))
    raw = request_embeddings(client, space, [item], usage, time.sleep)
    return next(iter(match_response(raw, space, [item]))).embedding


def _filtered_query(request):
    query = select(FragmentRow.fragment_id).where(
        FragmentRow.snapshot_id == request.snapshot_id
    )
    for key, value in request.filters.model_dump(exclude_none=True).items():
        query = query.where(getattr(FragmentRow, key) == value)
    return query


def _rank_candidates(connection, request, snapshot, client, usage, encoding):
    query = _filtered_query(request)
    if normalize_address(request.query) is not None:
        identities = connection.scalars(
            query.order_by(
                FragmentRow.page_start,
                FragmentRow.legal_unit_id,
                FragmentRow.payload["part_index"].as_integer(),
                FragmentRow.fragment_id,
            )
        ).all()
        return [Ranking(fragment_id=identity, score=1.0) for identity in identities]
    vector = _query_vector(request.query, snapshot, client, usage, encoding)
    distance = FragmentRow.embedding.cosine_distance(vector)
    candidates = connection.execute(
        query.add_columns(distance)
        .order_by(distance, FragmentRow.fragment_id)
        .limit(CANDIDATE_LIMIT)
    )
    return [
        Ranking(fragment_id=identity, score=1 - value, vector_rank=rank)
        for rank, (identity, value) in enumerate(candidates, start=1)
    ]


def _load_evidence(connection, request, rankings):
    rows = connection.scalars(
        select(FragmentRow.payload).where(
            FragmentRow.snapshot_id == request.snapshot_id,
            FragmentRow.fragment_id.in_([r.fragment_id for r in rankings]),
        )
    )
    return {f.fragment_id: f for f in map(SearchFragment.model_validate, rows)}


def _missing_document(connection, request):
    code = request.filters.code
    if code is None:
        return None
    exists = connection.scalar(
        select(LegalUnitRow.legal_unit_id)
        .where(
            LegalUnitRow.snapshot_id == request.snapshot_id, LegalUnitRow.code == code
        )
        .limit(1)
    )
    return None if exists else f"document_not_loaded:{code}"


def _search(connection, request, client, usage):
    base = {
        "snapshot_id": request.snapshot_id,
        "query_language": request.query_language,
        "usage": usage,
    }
    snapshot = (
        connection.execute(
            select(SnapshotRow).where(SnapshotRow.snapshot_id == request.snapshot_id)
        )
        .mappings()
        .one_or_none()
    )
    if snapshot is None:
        return SearchResult(status="no_match", reason="snapshot_not_indexed", **base)
    report = snapshot["reports"]["chunk"]
    base.update(
        document_title=snapshot["document_title"],
        consolidation_label=snapshot["consolidation_label"],
        extraction_issues=report.get("extraction_issues", []),
        unresolved_references=report.get("unresolved_references", []),
        omissions=report.get("omissions", []),
        uncovered_unit_ids=report.get("uncovered_unit_ids", []),
        table_issues=report.get("table_issues", []),
        structure_issues=report.get("structure_issues", []),
    )
    missing = _missing_document(connection, request)
    if missing:
        return SearchResult(status="no_match", reason=missing, **base)
    encoding = tiktoken.get_encoding("cl100k_base")
    rankings = _rank_candidates(connection, request, snapshot, client, usage, encoding)
    fragments = _load_evidence(connection, request, rankings)
    hits, used, limited = assemble_hits(rankings, fragments, request, encoding)
    return SearchResult(
        status="ok" if hits else "no_match",
        reason=None
        if hits
        else "context_budget_exceeded"
        if limited
        else "address_not_found",
        hits=hits,
        context_tokens_used=used,
        unit_coverage=hit_coverage(hits),
        limitations=["context_budget_exceeded"] if limited else [],
        **base,
    )


def search_tax_documents(
    query,
    limit=5,
    query_language=None,
    *,
    snapshot_id,
    engine,
    client=None,
    filters=None,
    context_tokens=6000,
) -> SearchResult:
    request_values = {
        "query": query,
        "limit": limit,
        "query_language": query_language,
        "snapshot_id": snapshot_id,
        "context_tokens": context_tokens,
    }
    if filters is not None:
        request_values["filters"] = filters
    request = SearchRequest(**request_values)
    language = detect_query_language(request.query, request.query_language)
    request = request.model_copy(
        update={
            "query_language": language,
            "filters": merge_filters(request.filters, normalize_address(request.query)),
        }
    )
    usage = {
        "requests": 0,
        "retries": 0,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
    with (
        engine.connect().execution_options(
            isolation_level="REPEATABLE READ"
        ) as connection,
        connection.begin(),
    ):
        return _search(connection, request, client, usage)
