"""A retrieval evaluation must expose missed targets and missing context."""

import json

import pytest

from atrag.ingestion.services.packages import build_package, load_package
from atrag.retrieval.evaluation import assess_case, summarize, validate_benchmark
from atrag.schemas.evaluation import Benchmark, EvaluationCase
from atrag.schemas.retrieval import Ranking, SearchHit, SearchResult


@pytest.fixture
def document(prepared_document, tmp_path):
    data, parsed, chunked, embedded = prepared_document
    return load_package(
        build_package(parsed, chunked, embedded, data, tmp_path / "pkg")
    )


@pytest.fixture
def example(document):
    fragment = next(f for f in document.fragments if f.paragraph == "1")
    case = EvaluationCase(
        case_id="residency",
        query="How is residency determined?",
        language="en",
        kind="semantic",
        expected_unit_ids=fragment.legal_unit_ids,
        required_context_ids=[fragment.legal_unit_id],
    )
    hit = SearchHit(
        **Ranking(fragment_id=fragment.fragment_id, score=1 / 61).model_dump(),
        fragment=fragment,
        related_units=[],
        related_tables=[],
    )
    result = SearchResult(
        status="ok",
        snapshot_id=document.snapshot.snapshot_id,
        query_language="en",
        hits=[hit],
        document_title=fragment.source.document_title,
        consolidation_label=fragment.source.consolidation_label,
        usage={
            "requests": 0,
            "retries": 0,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        },
    )
    return case, result


def test_assessment_checks_original_evidence_and_context(document, example):
    case, result = example
    assessed = assess_case(case, result, document)
    assert assessed["target_found"] is True
    assert assessed["context_recall"] == 1
    assert assessed["citations_correct"] is True
    assert assessed["passed"] is True
    assert assessed["missing_context_ids"] == []


def test_parent_address_and_intro_do_not_count_as_full_context(document, example):
    case, result = example
    case.required_context_ids = result.hits[0].fragment.context_unit_ids
    assessed = assess_case(case, result, document)
    assert assessed["context_recall"] == 0
    assert assessed["missing_context_ids"] == case.required_context_ids
    assert not assessed["passed"]


def test_wrong_part_of_the_expected_unit_does_not_count_as_a_hit(document, example):
    case, result = example
    case = EvaluationCase(
        **{
            **case.model_dump(),
            "expected_quotes": {case.expected_unit_ids[0]: ["São residentes"]},
        }
    )
    assert assess_case(case, result, document)["target_found"]
    part = result.hits[0].fragment
    result.hits[0].fragment = part.model_copy(update={"text_original": "a) Exceto"})
    assert not assess_case(case, result, document)["target_found"]


def test_missing_target_is_not_found_through_an_unrelated_reference(document, example):
    case, result = example
    case = case.model_copy(update={"expected_unit_ids": ["pt:cirs:code:article:999"]})
    assessed = assess_case(case, result, document)
    assert assessed["target_found"] is False
    assert assessed["passed"] is False


def test_missing_context_and_corrupt_quote_fail_independently(document, example):
    case, result = example
    case = case.model_copy(
        update={"required_context_ids": ["missing", *case.required_context_ids]}
    )
    assert assess_case(case, result, document)["context_recall"] == 0.5
    result.hits[0].fragment = result.hits[0].fragment.model_copy(
        update={"text_original": "invented"}
    )
    assessed = assess_case(case, result, document)
    assert assessed["target_found"] is True
    assert assessed["missing_context_ids"] == ["missing", case.required_context_ids[1]]
    assert assessed["context_recall"] < 1
    assert assessed["citations_correct"] is False
    assert assessed["passed"] is False


def test_no_match_requires_reason_and_empty_hits(document, example):
    case, result = example
    case = EvaluationCase(
        case_id="missing",
        query="CIVA 53",
        language="ru",
        kind="exact",
        expected_status="no_match",
        expected_reason="document_not_loaded:CIVA",
    )
    result = result.model_copy(
        update={
            "status": "no_match",
            "reason": case.expected_reason,
            "hits": [],
            "query_language": "ru",
        }
    )
    assert assess_case(case, result, document)["passed"] is True
    result.reason = "address_not_found"
    assert assess_case(case, result, document)["passed"] is False


def test_benchmark_validates_manual_addresses_pages_and_snapshot(document, example):
    case, _ = example
    unit = next(
        u for u in document.units if u.legal_unit_id == case.expected_unit_ids[0]
    )
    identities = case.expected_unit_ids + case.required_context_ids
    benchmark = Benchmark(
        snapshot_id=document.snapshot.snapshot_id,
        cases=[case],
        unit_pages={
            u.legal_unit_id: [u.page_start, u.page_end]
            for u in document.units
            if u.legal_unit_id in identities
        },
    )
    validate_benchmark(benchmark, document)
    benchmark.unit_pages[unit.legal_unit_id] = [2, 2]
    with pytest.raises(ValueError, match="pages"):
        validate_benchmark(benchmark, document)
    benchmark.unit_pages = {}
    benchmark.snapshot_id = "b" * 64
    with pytest.raises(ValueError, match="snapshot"):
        validate_benchmark(benchmark, document)


def test_summary_keeps_languages_modes_and_failures_separate():
    records = [
        {
            "case_id": "a",
            "kind": "semantic",
            "language": "ru",
            "mode": "hybrid",
            "target_found": True,
            "passed": True,
            "citations_correct": True,
            "context_recall": 1,
            "seconds": 1.0,
            "usage": {"usage": {"total_tokens": 5}},
        },
        {
            "case_id": "b",
            "kind": "semantic",
            "language": "ru",
            "mode": "hybrid",
            "target_found": False,
            "passed": False,
            "citations_correct": True,
            "context_recall": 0,
            "seconds": 3.0,
            "usage": {"usage": {"total_tokens": 7}},
        },
        {
            "case_id": "c",
            "kind": "exact",
            "language": "en",
            "mode": "fts",
            "target_found": True,
            "passed": True,
            "citations_correct": True,
            "context_recall": 1,
            "seconds": 0.5,
            "usage": {"usage": {"total_tokens": 0}},
        },
    ]
    summary = summarize(records)
    assert summary["hybrid:ru"]["semantic_hit_at_5"] == 0.5
    assert summary["hybrid:ru"]["context_recall"] == 0.5
    assert summary["hybrid:ru"]["total_tokens"] == 12
    assert summary["hybrid:ru"]["p95_seconds"] == 3
    assert summary["hybrid:ru"]["failed_case_ids"] == ["b"]
    assert summary["fts:en"]["semantic_hit_at_5"] is None
    assert summary["fts:en"]["exact_pass_rate"] == 1
    assert summary["fts:en"]["exact_cases"] == 1
    assert summary["hybrid:ru"]["exact_pass_rate"] is None
    assert json.loads(json.dumps(summary)) == summary
    records.append({**records[-1], "case_id": "d", "passed": False})
    assert summarize(records)["fts:en"]["exact_pass_rate"] == 0.5


@pytest.mark.parametrize(
    "values",
    [
        {"expected_unit_ids": []},
        {"expected_status": "no_match"},
        {
            "expected_status": "no_match",
            "expected_reason": "address_not_found",
            "required_context_ids": ["absent"],
        },
        {
            "expected_status": "no_match",
            "expected_reason": "address_not_found",
            "required_table_ids": ["absent"],
        },
    ],
)
def test_invalid_expectation_is_rejected(values):
    with pytest.raises(ValueError):
        EvaluationCase(
            case_id="bad", query="CIRS 16", language="pt", kind="exact", **values
        )


def test_targets_after_fifth_hit_do_not_count(document, example):
    case, result = example
    other = next(f for f in document.fragments if f.article == "68-A")
    unrelated = result.hits[0].model_copy(
        update={"fragment_id": other.fragment_id, "fragment": other}
    )
    result.hits = [unrelated] * 5 + result.hits
    assert assess_case(case, result, document)["target_found"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("snapshot_id", "b" * 64),
        ("document_title", "wrong"),
        ("consolidation_label", "wrong"),
        ("query_language", "ru"),
        ("status", "no_match"),
        ("reason", "wrong"),
    ],
)
def test_wrong_result_metadata_fails(document, example, field, value):
    case, result = example
    setattr(result, field, value)
    assert assess_case(case, result, document)["passed"] is False


def test_tables_and_linked_evidence_are_checked(document, example):
    case, result = example
    table = document.tables[0]
    case.required_table_ids = [table.table_id]
    assert assess_case(case, result, document)["missing_table_ids"] == [table.table_id]
    result.hits[0].related_tables = [table]
    assert assess_case(case, result, document)["passed"] is True
    result.hits[0].related_tables = [table.model_copy(update={"rows": [["wrong"]]})]
    assert assess_case(case, result, document)["citations_correct"] is False
    result.hits[0].related_tables = []
    unit = document.units[0]
    result.hits[0].related_units = [unit.model_copy(update={"text_original": "wrong"})]
    assert assess_case(case, result, document)["citations_correct"] is False


def test_related_units_supply_required_context(document, example):
    case, result = example
    unit = next(u for u in document.units if u.article == "68-A")
    case.required_context_ids = [unit.legal_unit_id]
    assert assess_case(case, result, document)["context_recall"] == 0
    result.hits[0].related_units = [unit]
    assert assess_case(case, result, document)["passed"] is True


def test_benchmark_rejects_unknown_and_missing_expectations(document, example):
    case, _ = example
    benchmark = Benchmark(
        snapshot_id=document.snapshot.snapshot_id, cases=[case], unit_pages={}
    )
    with pytest.raises(ValueError, match="Missing reviewed unit"):
        validate_benchmark(benchmark, document)
    benchmark.cases = [
        EvaluationCase(
            case_id="no-match",
            query="CIVA 53",
            language="en",
            kind="exact",
            expected_status="no_match",
            expected_reason="document_not_loaded:CIVA",
        )
    ]
    benchmark.unit_pages = {"missing": (1, 1)}
    with pytest.raises(ValueError, match="pages"):
        validate_benchmark(benchmark, document)


def test_benchmark_checks_tables_and_independent_status_text_role(document, example):
    from atrag.schemas.evaluation import TableExpectation, UnitCheck

    case, _ = example
    ids = case.expected_unit_ids + case.required_context_ids
    unit = next(u for u in document.units if u.legal_unit_id == ids[0])
    table = document.tables[0]
    case.required_table_ids = [table.table_id]
    benchmark = Benchmark(
        snapshot_id=document.snapshot.snapshot_id,
        cases=[case],
        unit_pages={
            u.legal_unit_id: (u.page_start, u.page_end)
            for u in document.units
            if u.legal_unit_id in ids
        },
    )
    with pytest.raises(ValueError, match="Missing reviewed table"):
        validate_benchmark(benchmark, document)
    benchmark.tables[table.table_id] = TableExpectation(
        header_rows=table.header_rows, rows=table.rows, pages=[1]
    )
    validate_benchmark(benchmark, document)
    benchmark.tables[table.table_id].rows = [["wrong"]]
    with pytest.raises(ValueError, match="table differs"):
        validate_benchmark(benchmark, document)
    benchmark.tables = {"missing": TableExpectation(header_rows=[], rows=[], pages=[1])}
    case.required_table_ids = []
    with pytest.raises(ValueError, match="Unknown benchmark table"):
        validate_benchmark(benchmark, document)
    benchmark.tables = {}
    benchmark.unit_checks[unit.legal_unit_id] = UnitCheck(
        contains=[unit.text_original], role=unit.role, legal_status=unit.legal_status
    )
    validate_benchmark(benchmark, document)
    benchmark.unit_checks[unit.legal_unit_id].role = "wrong"
    with pytest.raises(ValueError, match="role differs"):
        validate_benchmark(benchmark, document)
    benchmark.unit_checks[unit.legal_unit_id].contains = ["invented"]
    with pytest.raises(ValueError, match="text differs"):
        validate_benchmark(benchmark, document)
    benchmark.unit_checks = {"missing": UnitCheck()}
    with pytest.raises(ValueError, match="Missing reviewed unit"):
        validate_benchmark(benchmark, document)


def test_unsafe_case_filename_and_duplicate_cases_are_rejected(example):
    case, result = example
    with pytest.raises(ValueError):
        EvaluationCase(**{**case.model_dump(), "case_id": "../overwrite"})
    with pytest.raises(ValueError, match="Duplicate"):
        Benchmark(snapshot_id=result.snapshot_id, unit_pages={}, cases=[case, case])


def test_reviewed_quotes_and_expansion_expectations_are_validated(document, example):
    from atrag.schemas.evaluation import ExpansionCase, UnitCheck
    from atrag.schemas.retrieval import ContextRequest

    case, _ = example
    identity = case.expected_unit_ids[0]
    units = {u.legal_unit_id: u for u in document.units}
    benchmark = Benchmark(
        snapshot_id=document.snapshot.snapshot_id,
        cases=[case],
        unit_pages={i: (u.page_start, u.page_end) for i, u in units.items()},
    )
    for quotes in [[], [""], ["Invented legal evidence"]]:
        case.expected_quotes = {identity: quotes}
        with pytest.raises(ValueError, match="quote"):
            validate_benchmark(benchmark, document)
    case.expected_quotes = {identity: [units[identity].text_original]}
    validate_benchmark(benchmark, document)
    benchmark.unit_checks[identity] = UnitCheck(contains=["São residentes"])
    validate_benchmark(benchmark, document)
    with pytest.raises(ValueError, match="expected target"):
        EvaluationCase(**{**case.model_dump(), "expected_quotes": {"absent": ["text"]}})
    expansion = ExpansionCase(
        case_id="original",
        language="pt",
        kind="original",
        request=ContextRequest(
            snapshot_id=document.snapshot.snapshot_id, legal_unit_id=identity
        ),
        required_unit_ids=[identity],
    )
    benchmark.expansion_cases = [expansion]
    validate_benchmark(benchmark, document)
    expansion.request.snapshot_id = "b" * 64
    with pytest.raises(ValueError, match="snapshot"):
        validate_benchmark(benchmark, document)
    expansion.request.snapshot_id = benchmark.snapshot_id
    expansion.required_unit_ids = ["absent"]
    with pytest.raises(ValueError, match="unit pages"):
        validate_benchmark(benchmark, document)
    expansion.required_unit_ids = [identity]
    expansion.required_table_ids = ["absent"]
    with pytest.raises(ValueError, match="expansion table"):
        validate_benchmark(benchmark, document)


def test_all_reviewed_units_can_be_required_by_one_expansion(document, example):
    from atrag.schemas.evaluation import ExpansionCase
    from atrag.schemas.retrieval import ContextRequest

    case, _ = example
    identity = case.expected_unit_ids[0]
    unit = next(u for u in document.units if u.legal_unit_id == identity)
    case.expected_unit_ids = [identity]
    case.required_context_ids = [identity]
    benchmark = Benchmark(
        snapshot_id=unit.snapshot_id,
        cases=[case],
        unit_pages={identity: (unit.page_start, unit.page_end)},
        expansion_cases=[
            ExpansionCase(
                case_id="all",
                language="pt",
                kind="original",
                request=ContextRequest(
                    snapshot_id=unit.snapshot_id, legal_unit_id=identity
                ),
                required_unit_ids=[identity],
            )
        ],
    )
    validate_benchmark(benchmark, document)


def test_context_requires_every_adjacent_part_not_just_a_unit_address(tmp_path):
    from fixtures.prepared import prepare_document

    data, parsed, chunked, embedded = prepare_document(
        tmp_path / "split", second_paragraph="2 - " + "Condição fiscal. " * 220
    )
    document = load_package(
        build_package(parsed, chunked, embedded, data, tmp_path / "split-pkg")
    )
    parts = [f for f in document.fragments if f.paragraph == "2"]
    assert 1 < len(parts) <= 5
    identity = parts[0].legal_unit_id
    case = EvaluationCase(
        case_id="split",
        query="CIRS 16",
        language="pt",
        kind="exact",
        expected_unit_ids=[identity],
        expected_quotes={identity: ["2 - "]},
        required_context_ids=[identity],
    )
    result = SearchResult(
        status="ok",
        snapshot_id=document.snapshot.snapshot_id,
        query_language="pt",
        document_title=document.snapshot.source.document_title,
        consolidation_label=document.snapshot.source.consolidation_label,
        usage={},
        hits=[
            SearchHit(
                fragment_id=p.fragment_id,
                score=1,
                fragment=p,
                related_units=[],
                related_tables=[],
            )
            for p in parts
        ],
    )
    assert assess_case(case, result, document)["context_recall"] == 1
    result.hits = result.hits[:1]
    assert assess_case(case, result, document)["context_recall"] == 0
    result.hits *= len(parts)
    assert assess_case(case, result, document)["context_recall"] == 0


def test_expanded_table_citations_match_reviewed_cells(document):
    from atrag.retrieval.evaluation import assess_expansion
    from atrag.retrieval.expansion import expand_context
    from atrag.schemas.evaluation import ExpansionCase
    from atrag.schemas.retrieval import ContextRequest

    units = {u.legal_unit_id: u for u in document.units}
    selected = ContextRequest(
        snapshot_id=document.snapshot.snapshot_id,
        legal_unit_id="pt:cirs:code:article:68-A",
    )
    case = ExpansionCase(
        case_id="table",
        language="pt",
        kind="original",
        request=selected,
        required_unit_ids=[selected.legal_unit_id],
        required_table_ids=["rates"],
    )
    result = expand_context(
        selected,
        document.snapshot.source,
        units,
        {t.table_id: t for t in document.tables},
    )
    assert assess_expansion(case, result, document)["passed"]
    result.tables[0] = result.tables[0].model_copy(update={"rows": [["1000", "99"]]})
    assessed = assess_expansion(case, result, document)
    assert not assessed["passed"]
    assert not assessed["citations_correct"]


@pytest.mark.parametrize("fail", [False, True])
def test_expansion_report_preserves_outputs_timing_and_completed_work(
    document, example, tmp_path, monkeypatch, fail
):
    from itertools import count

    from atrag.retrieval.evaluate import evaluate_search
    from atrag.retrieval.expansion import expand_context
    from atrag.schemas.evaluation import ExpansionCase
    from atrag.schemas.retrieval import ContextRequest

    case, search_result = example
    units = {u.legal_unit_id: u for u in document.units}
    selected = ContextRequest(
        snapshot_id=document.snapshot.snapshot_id,
        legal_unit_id=case.expected_unit_ids[0],
    )
    expanded = expand_context(selected, document.snapshot.source, units, {})
    cases = [
        ExpansionCase(
            case_id=name,
            kind="original",
            language="en",
            request=selected,
            required_unit_ids=[selected.legal_unit_id],
        )
        for name in ("one", "two")
    ]
    benchmark = Benchmark(
        snapshot_id=document.snapshot.snapshot_id,
        cases=[case],
        unit_pages={i: (u.page_start, u.page_end) for i, u in units.items()},
        expansion_cases=cases,
    )
    path = tmp_path / "benchmark.json"
    path.write_text(benchmark.model_dump_json())
    package = tmp_path / "manifest.json"
    package.write_text("{}")
    calls = []

    def load(request, *, engine):
        calls.append(request)
        if fail and len(calls) == 2:
            raise ValueError("interrupted expansion")
        return expanded

    clock = count(1)
    monkeypatch.setattr("atrag.retrieval.evaluate.perf_counter", lambda: next(clock))
    monkeypatch.setattr("atrag.retrieval.evaluate.load_package", lambda p: document)
    monkeypatch.setattr(
        "atrag.retrieval.evaluate.search_tax_documents", lambda *a, **k: search_result
    )
    monkeypatch.setattr("atrag.retrieval.evaluate.load_context", load)
    if fail:
        with pytest.raises(ValueError, match="interrupted"):
            evaluate_search(package, path, None, tmp_path / "out")
        directory = next((tmp_path / "out/runs").iterdir())
    else:
        directory = evaluate_search(package, path, None, tmp_path / "out")
    report = json.loads((directory / "report.json").read_bytes())
    assert len(report["expansion_records"]) == (1 if fail else 2)
    assert all(
        r["seconds"] == 1 for r in report["records"] + report["expansion_records"]
    )
    assert all(r["passed"] for r in report["expansion_records"])
    assert json.loads(
        (directory / "one-context.json").read_bytes()
    ) == expanded.model_dump(mode="json")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    assert manifest["status"] == ("failed" if fail else "succeeded")


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_saves_outputs_and_completed_results_on_failure(
    document, example, tmp_path, monkeypatch, fail
):
    from atrag.ingestion.package_validation import digest
    from atrag.retrieval.evaluate import evaluate_search

    case, result = example
    ids = case.expected_unit_ids + case.required_context_ids
    benchmark = Benchmark(
        snapshot_id=document.snapshot.snapshot_id,
        cases=[case],
        unit_pages={
            u.legal_unit_id: (u.page_start, u.page_end)
            for u in document.units
            if u.legal_unit_id in ids
        },
    )
    path = tmp_path / "benchmark.json"
    path.write_text(benchmark.model_dump_json())
    package = tmp_path / "manifest.json"
    package.write_text("{}")
    calls = []

    def search(query, limit, language, **kwargs):
        calls.append((query, limit, language, kwargs))
        if fail and len(calls) == 2:
            raise ValueError("synthetic failure")
        return result

    monkeypatch.setattr("atrag.retrieval.evaluate.load_package", lambda p: document)
    monkeypatch.setattr("atrag.retrieval.evaluate.search_tax_documents", search)
    if fail:
        with pytest.raises(ValueError, match="synthetic"):
            evaluate_search(package, path, None, tmp_path / "out")
        directory = next((tmp_path / "out/runs").iterdir())
    else:
        directory = evaluate_search(package, path, None, tmp_path / "out")
    report = json.loads((directory / "report.json").read_bytes())
    manifest = json.loads((directory / "manifest.json").read_bytes())
    assert manifest["status"] == ("failed" if fail else "succeeded")
    assert manifest["benchmark_sha256"] == digest(path.read_bytes())
    assert manifest["package_sha256"] == digest(package.read_bytes())
    assert report["limit"] == 5
    assert report["context_tokens"] == 6000
    assert report["rrf_k"] == 60
    assert report["embedding_model"] == document.manifest.model
    assert report["embedding_dimensions"] == 1536
    assert report["schema_version"] == 1
    assert report["snapshot_id"] == document.snapshot.snapshot_id
    assert report["corpus_issues"] == document.reports["chunk"]
    assert "unverified" in report["vector_provenance"]
    assert len(report["records"]) == (1 if fail else 3)
    assert report["records"][0]["passed"] is True
    assert report["records"][0]["usage"] == result.usage
    assert calls[0][:3] == (case.query, 5, case.language)
    assert calls[0][3]["snapshot_id"] == document.snapshot.snapshot_id
    assert calls[0][3]["context_tokens"] == 6000
    assert calls[0][3]["rrf_k"] == 60
    assert calls[0][3]["filters"] == case.filters
    assert json.loads(
        (directory / f"{case.case_id}-vector.json").read_bytes()
    ) == result.model_dump(mode="json")
    if not fail:
        assert [c[3]["mode"] for c in calls] == ["vector", "fts", "hybrid"]
        for name, sha256 in manifest["output_sha256"].items():
            assert digest((directory / name).read_bytes()) == sha256
