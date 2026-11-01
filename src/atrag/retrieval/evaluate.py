"""Run a reviewed benchmark through the public search service and save evidence."""

from time import perf_counter
from typing import get_args

from atrag.ingestion.package_validation import digest
from atrag.ingestion.services.artifacts import write_json
from atrag.ingestion.services.packages import load_package
from atrag.ingestion.services.runs import record_run
from atrag.retrieval.evaluation import (
    assess_case,
    assess_expansion,
    summarize,
    validate_benchmark,
)
from atrag.retrieval.search import search_tax_documents
from atrag.retrieval.text import load_context
from atrag.schemas.evaluation import Benchmark
from atrag.schemas.retrieval import RetrievalMode


def evaluate_search(
    package,
    benchmark_path,
    engine,
    data_dir,
    *,
    client=None,
    context_tokens=6000,
    rrf_k=60,
):
    with record_run(data_dir, stage="evaluate") as run:
        benchmark_raw = benchmark_path.read_bytes()
        benchmark = Benchmark.model_validate_json(benchmark_raw)
        document = load_package(package)
        validate_benchmark(benchmark, document)
        run.manifest.update(
            snapshot_id=benchmark.snapshot_id,
            benchmark_sha256=digest(benchmark_raw),
            package_sha256=digest(package.read_bytes()),
        )
        report = {
            "schema_version": 1,
            "benchmark_schema_version": benchmark.schema_version,
            "snapshot_id": benchmark.snapshot_id,
            "embedding_model": document.manifest.model,
            "embedding_dimensions": document.manifest.dimensions,
            "vector_provenance": "unverified; establish separately from benchmark scores",
            "limit": 5,
            "context_tokens": context_tokens,
            "rrf_k": rrf_k,
            "corpus_issues": document.reports["chunk"],
            "records": [],
            "summary": {},
            "expansion_records": [],
        }
        try:
            for case in benchmark.cases:
                for mode in get_args(RetrievalMode):
                    started = perf_counter()
                    result = search_tax_documents(
                        case.query,
                        5,
                        None if case.detect_language else case.language,
                        snapshot_id=benchmark.snapshot_id,
                        engine=engine,
                        client=client,
                        context_tokens=context_tokens,
                        rrf_k=rrf_k,
                        mode=mode,
                        filters=case.filters,
                    )
                    elapsed = perf_counter() - started
                    report["records"].append(
                        dict(
                            case_id=case.case_id,
                            query=case.query,
                            kind=case.kind,
                            language=case.language,
                            mode=mode,
                            seconds=elapsed,
                            usage=result.usage,
                            **assess_case(case, result, document),
                        )
                    )
                    write_json(
                        run.directory / f"{case.case_id}-{mode}.json",
                        result.model_dump(mode="json"),
                    )
                    report["summary"] = summarize(report["records"])
                    write_json(run.directory / "report.json", report)
            _evaluate_expansions(benchmark, engine, document, run.directory, report)
        finally:
            write_json(run.directory / "report.json", report)
        run.manifest["output_sha256"] = {
            path.name: digest(path.read_bytes())
            for path in sorted(run.directory.glob("*.json"))
            if path.name != "manifest.json"
        }
    return run.directory


def _evaluate_expansions(benchmark, engine, document, directory, report):
    for case in benchmark.expansion_cases:
        started = perf_counter()
        result = load_context(case.request, engine=engine)
        report["expansion_records"].append(
            {
                "case_id": case.case_id,
                "language": case.language,
                "kind": case.kind,
                "seconds": perf_counter() - started,
                **assess_expansion(case, result, document),
            }
        )
        write_json(
            directory / f"{case.case_id}-context.json", result.model_dump(mode="json")
        )
        write_json(directory / "report.json", report)
