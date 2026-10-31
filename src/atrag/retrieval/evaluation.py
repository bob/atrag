"""Evaluate returned evidence, including corpus omissions, without a judge model."""

from math import ceil
from statistics import median


def validate_benchmark(benchmark, document):
    if benchmark.snapshot_id != document.snapshot.snapshot_id:
        raise ValueError("Benchmark snapshot differs from the package")
    units = {u.legal_unit_id: u for u in document.units}
    _validate_unit_pages(benchmark, units)
    _validate_unit_checks(benchmark, units)
    _validate_cases(benchmark)
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


def _validate_cases(benchmark):
    for case in benchmark.cases:
        for identity in case.expected_unit_ids + case.required_context_ids:
            if identity not in benchmark.unit_pages:
                raise ValueError(f"Missing reviewed unit pages: {identity}")
        for identity in case.required_table_ids:
            if identity not in benchmark.tables:
                raise ValueError(f"Missing reviewed table: {identity}")


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


def _matching_evidence(case, result):
    matches = [
        hit
        for hit in result.hits[:5]
        if set(case.expected_unit_ids).intersection(hit.fragment.legal_unit_ids)
    ]
    present = set()
    tables = set()
    for hit in matches:
        present.update(hit.fragment.legal_unit_ids + hit.fragment.context_unit_ids)
        present.update(u.legal_unit_id for u in hit.related_units)
        tables.update(t.table_id for t in hit.fragment.tables + hit.related_tables)
    return matches, present, tables


def assess_case(case, result, document):
    matches, present, tables = _matching_evidence(case, result)
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
