"""Evaluate returned evidence, including corpus omissions, without a judge model."""

from math import ceil
from statistics import median


def validate_benchmark(benchmark, document):
    if benchmark.snapshot_id != document.snapshot.snapshot_id:
        raise ValueError("Benchmark snapshot differs from the package")
    units = {u.legal_unit_id: u for u in document.units}
    _validate_unit_pages(benchmark, units)
    _validate_unit_checks(benchmark, units)
    _validate_cases(benchmark, units)
    _validate_tables(benchmark, {t.table_id: t for t in document.tables})


def _validate_unit_pages(benchmark, units):
    for identity, pages in benchmark.unit_pages.items():
        unit = units.get(identity)
        if unit is None or tuple(pages) != (unit.page_start, unit.page_end):
            raise ValueError(f"Benchmark pages differ: {identity}")


def _validate_unit_checks(benchmark, units):
    for identity, expected in benchmark.unit_checks.items():
        if identity not in benchmark.unit_pages:
            raise ValueError(f"Missing reviewed unit pages: {identity}")
        unit = units[identity]
        if any(text not in unit.text_original for text in expected.contains):
            raise ValueError(f"Benchmark text differs: {identity}")
        for key, value in expected.model_dump(
            exclude={"contains"}, exclude_none=True
        ).items():
            if getattr(unit, key) != value:
                raise ValueError(f"Benchmark {key} differs: {identity}")


def _validate_cases(benchmark, units):
    for case in benchmark.cases:
        for identity in case.expected_unit_ids + case.required_context_ids:
            if identity not in benchmark.unit_pages:
                raise ValueError(f"Missing reviewed unit pages: {identity}")
        for identity in case.required_table_ids:
            if identity not in benchmark.tables:
                raise ValueError(f"Missing reviewed table: {identity}")
        _validate_quotes(case, units)
    _validate_expansions(benchmark)


def _validate_quotes(case, units):
    for identity, quotes in case.expected_quotes.items():
        if not quotes or any(not quote for quote in quotes):
            raise ValueError(f"Search quotes must be nonempty: {identity}")
        if any(quote not in units[identity].text_original for quote in quotes):
            raise ValueError(f"Search quote differs from source: {identity}")


def _validate_expansions(benchmark):
    for case in benchmark.expansion_cases:
        if case.request.snapshot_id != benchmark.snapshot_id:
            raise ValueError("Expansion snapshot differs from the benchmark")
        if not set(case.required_unit_ids) <= benchmark.unit_pages.keys():
            raise ValueError("Missing reviewed expansion unit pages")
        if not set(case.required_table_ids) <= benchmark.tables.keys():
            raise ValueError("Missing reviewed expansion table")


def _validate_tables(benchmark, tables):
    for identity, expected in benchmark.tables.items():
        table = tables.get(identity)
        if table is None:
            raise ValueError(f"Unknown benchmark table: {identity}")
        actual = {
            "header_rows": table.header_rows,
            "rows": table.rows,
            "pages": sorted({p.page_number for p in table.parts}),
        }
        if actual != expected.model_dump():
            raise ValueError(f"Benchmark table differs: {identity}")


def _citations_correct(result, document):
    source = document.snapshot.source
    actual = (result.snapshot_id, result.document_title, result.consolidation_label)
    expected = (
        document.snapshot.snapshot_id,
        source.document_title,
        source.consolidation_label,
    )
    if actual != expected:
        return False
    fragments = {f.fragment_id: f for f in document.fragments}
    units = {u.legal_unit_id: u for u in document.units}
    tables = {t.table_id: t for t in document.tables}
    return all(_hit_matches(hit, fragments, units, tables) for hit in result.hits)


def _hit_matches(hit, fragments, units, tables):
    return all(
        (
            hit.fragment == fragments.get(hit.fragment_id),
            all(u == units.get(u.legal_unit_id) for u in hit.related_units),
            all(t == tables.get(t.table_id) for t in hit.related_tables),
        )
    )


def _matching_evidence(case, result, document):
    matches = [
        hit
        for hit in result.hits[:5]
        if set(case.expected_unit_ids).intersection(hit.fragment.legal_unit_ids)
        and _expected_quote_present(case, hit.fragment)
    ]
    present = _delivered_units(
        result.hits[:5], {u.legal_unit_id: u for u in document.units}
    )
    tables = set()
    for hit in result.hits[:5]:
        tables.update(t.table_id for t in hit.fragment.tables + hit.related_tables)
    return matches, present, tables


def _delivered_units(hits, units):
    present, groups = set(), {}
    for hit in hits:
        fragment = hit.fragment
        groups.setdefault(fragment.legal_unit_id, []).append(fragment)
        evidence = [(i, fragment.text_original) for i in fragment.legal_unit_ids]
        evidence.extend((c.legal_unit_id, c.text_original) for c in fragment.context)
        evidence.extend((u.legal_unit_id, u.text_original) for u in hit.related_units)
        present.update(
            i for i, text in evidence if i in units and units[i].text_original in text
        )
    for identity, parts in groups.items():
        if _complete_parts(parts, units[identity]):
            present.add(identity)
    return present


def _complete_parts(parts, unit):
    cursor = 0
    for part in sorted(parts, key=lambda p: p.source_start):
        if (
            part.source_start != cursor
            or part.text_original
            != unit.text_original[part.source_start : part.source_end]
        ):
            return False
        cursor = part.source_end
    return cursor == len(unit.text_original)


def assess_expansion(case, result, document):
    units = {u.legal_unit_id: u for u in document.units}
    tables = {t.table_id: t for t in document.tables}
    delivered = _complete_text_ids(result.units, units)
    missing = sorted(set(case.required_unit_ids) - delivered)
    missing_tables = sorted(
        set(case.required_table_ids) - {t.table_id for t in result.tables}
    )
    citations = (
        result.snapshot_id == document.snapshot.snapshot_id
        and result.legal_unit_id == case.request.legal_unit_id
        and all(_text_matches(t, units, document.snapshot.source) for t in result.units)
        and all(t == tables.get(t.table_id) for t in result.tables)
    )
    return {
        "passed": result.complete and not missing and not missing_tables and citations,
        "complete": result.complete,
        "citations_correct": citations,
        "missing_unit_ids": missing,
        "missing_table_ids": missing_tables,
        "context_tokens_used": result.context_tokens_used,
    }


def _complete_text_ids(texts, units):
    return {
        t.legal_unit_id
        for t in texts
        if t.complete
        and t.legal_unit_id in units
        and t.text_original == units[t.legal_unit_id].text_original
    }


def _text_matches(text, units, source):
    unit = units.get(text.legal_unit_id)
    if unit is None:
        return False
    metadata = ("snapshot_id", "citation_label", "role", "legal_status", "locators")
    if any(getattr(text, key) != getattr(unit, key) for key in metadata):
        return False
    return all(
        (
            text.source.model_dump(exclude={"local_path", "expected_sha256"})
            == source.model_dump(exclude={"local_path", "expected_sha256"}),
            0 <= text.source_start < text.source_end <= len(unit.text_original),
            text.source_length == len(unit.text_original),
            text.text_original
            == unit.text_original[text.source_start : text.source_end],
            text.complete
            == (text.source_start == 0 and text.source_end == len(unit.text_original)),
            text.next_start
            == (text.source_end if text.source_end < len(unit.text_original) else None),
        )
    )


def _expected_quote_present(case, fragment):
    if not case.expected_quotes:
        return True
    return any(
        identity in fragment.legal_unit_ids
        and all(quote in fragment.text_original for quote in quotes)
        for identity, quotes in case.expected_quotes.items()
    )


def assess_case(case, result, document):
    matches, present, tables = _matching_evidence(case, result, document)
    missing_context = sorted(set(case.required_context_ids) - present)
    missing_tables = sorted(set(case.required_table_ids) - tables)
    recall = (
        (len(case.required_context_ids) - len(missing_context))
        / len(case.required_context_ids)
        if case.required_context_ids
        else 1.0
    )
    target_found = bool(matches)
    if case.expected_status == "no_match":
        target_found = not result.hits
    citations = _citations_correct(result, document)
    passed = all(
        (
            target_found,
            not missing_context,
            not missing_tables,
            citations,
            result.status == case.expected_status,
            result.reason == case.expected_reason,
            result.query_language == case.language,
        )
    )
    return {
        "target_found": target_found,
        "context_recall": recall,
        "citations_correct": citations,
        "missing_context_ids": missing_context,
        "missing_table_ids": missing_tables,
        "passed": passed,
        "status": result.status,
        "reason": result.reason,
        "fragment_ids": [h.fragment_id for h in result.hits],
        "retrieved_unit_ids": [
            u for h in result.hits for u in h.fragment.legal_unit_ids
        ],
        "limitations": result.limitations,
    }


def _group_summary(rows):
    semantic = [r for r in rows if r["kind"] == "semantic"]
    exact = [r for r in rows if r["kind"] == "exact"]
    durations = sorted(r["seconds"] for r in rows)
    return {
        "cases": len(rows),
        "semantic_cases": len(semantic),
        "exact_cases": len(exact),
        "exact_pass_rate": (
            sum(r["passed"] for r in exact) / len(exact) if exact else None
        ),
        "semantic_hit_at_5": (
            sum(r["target_found"] for r in semantic) / len(semantic)
            if semantic
            else None
        ),
        "context_recall": sum(r["context_recall"] for r in rows) / len(rows),
        "citations_correct": all(r["citations_correct"] for r in rows),
        "passed": all(r["passed"] for r in rows),
        "failed_case_ids": [r["case_id"] for r in rows if not r["passed"]],
        "median_seconds": median(durations),
        "p95_seconds": durations[ceil(len(rows) * 0.95) - 1],
        "total_tokens": sum(r["usage"]["usage"]["total_tokens"] for r in rows),
    }


def summarize(records):
    groups = {}
    for row in records:
        groups.setdefault(f"{row['mode']}:{row['language']}", []).append(row)
    return {key: _group_summary(rows) for key, rows in groups.items()}
