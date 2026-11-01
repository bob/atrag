"""Return ranked parts within budget; references stay available addresses."""

import json

from atrag.schemas.retrieval import SearchHit


def reference_targets(fragment):
    return sorted(
        {
            target
            for reference in fragment.references
            if reference.status == "resolved"
            for target in reference.target_ids
        }
    )


def evidence_cost(fragment, related_units, related_tables, encoding):
    # The budget covers legal evidence; PDF geometry and processing metadata
    # remain available to the caller without displacing the actual source text.
    content = {
        "text": fragment.embedding_text,
        "notes": [n.raw_text for n in fragment.editorial_notes],
        "related_units": [
            {
                "citation": u.citation_label,
                "text": u.text_original,
                "notes": [n.raw_text for n in u.editorial_notes],
            }
            for u in related_units
        ],
        "tables": [
            {"headers": t.header_rows, "rows": t.rows}
            for t in [*fragment.tables, *related_tables]
        ],
    }
    return len(encoding.encode_ordinary(json.dumps(content, ensure_ascii=False)))


def assemble_hits(rankings, fragments, request, encoding):
    hits, used, limited = [], 0, False
    for ranking in rankings:
        fragment = fragments[ranking.fragment_id]
        hit = SearchHit(
            **ranking.model_dump(),
            fragment=fragment,
            related_units=[],
            related_tables=[],
            available_reference_ids=reference_targets(fragment),
        )
        cost = evidence_cost(fragment, [], [], encoding)
        if used + cost > request.context_tokens:
            limited = True
            continue
        hits.append(hit)
        used += cost
        if len(hits) == request.limit:
            break
    return hits, used, limited
