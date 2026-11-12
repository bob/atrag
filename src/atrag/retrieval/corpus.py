"""One ranking, query embedding and evidence budget across a pinned corpus."""

import tiktoken
from sqlalchemy import select

from atrag.db_models import FragmentRow, SnapshotRow
from atrag.retrieval.context import fit_hits, make_hit
from atrag.retrieval.query import merge_filters, normalize_address, query_language
from atrag.retrieval.search import CANDIDATE_LIMIT, _filtered_query, _query_vector
from atrag.schemas.ingestion.fragments import SearchFragment
from atrag.schemas.retrieval import Ranking, SearchRequest


def _candidates(connection, request, snapshots, client, usage, encoding):
    query = _filtered_query(request, snapshots).add_columns(FragmentRow.payload)
    if normalize_address(request.query) is not None:
        rows = connection.execute(
            query.order_by(
                FragmentRow.page_start,
                FragmentRow.legal_unit_id,
                FragmentRow.payload["part_index"].as_integer(),
                FragmentRow.snapshot_id,
                FragmentRow.fragment_id,
            )
        )
        return [
            make_hit(Ranking(fragment_id=i, score=1), SearchFragment.model_validate(p))
            for i, p in rows
        ]
    snapshot = (
        connection.execute(
            select(SnapshotRow).where(SnapshotRow.snapshot_id == snapshots[0])
        )
        .mappings()
        .one()
    )
    vector = _query_vector(request.query, snapshot, client, usage, encoding)
    distance = FragmentRow.embedding.cosine_distance(vector)
    rows = connection.execute(
        query.add_columns(distance)
        .order_by(
            distance,
            FragmentRow.snapshot_id,
            FragmentRow.fragment_id,
        )
        .limit(CANDIDATE_LIMIT)
    ).all()
    return [
        make_hit(
            Ranking(fragment_id=i, score=1 - d, vector_rank=rank),
            SearchFragment.model_validate(p),
        )
        for rank, (i, p, d) in enumerate(rows, start=1)
    ]


def search_corpus(request, snapshot_ids, *, engine, client, usage):
    """snapshot_ids is server-owned, already narrowed to the turn's release."""
    request = SearchRequest.model_validate(request)
    request = request.model_copy(
        update={
            "query_language": query_language(request.query, request.query_language),
            "filters": merge_filters(request.filters, normalize_address(request.query)),
        }
    )
    encoding = tiktoken.get_encoding("cl100k_base")
    with engine.connect() as connection:
        candidates = _candidates(
            connection, request, snapshot_ids, client, usage, encoding
        )
    return fit_hits(candidates, request, encoding)
