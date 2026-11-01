"""Source warnings must follow returned evidence rather than its whole corpus."""

from copy import deepcopy
from types import SimpleNamespace

from atrag.retrieval.diagnostics import hit_diagnostics


def test_diagnostics_follow_content_units_pages_and_tables():
    hits = [
        SimpleNamespace(
            fragment=SimpleNamespace(
                legal_unit_id="paragraph",
                legal_unit_ids=["child"],
                context_unit_ids=["parent"],
                locators=[SimpleNamespace(page_number=4)],
                table_ids=["inline"],
                tables=[
                    SimpleNamespace(
                        table_id="inline", parts=[SimpleNamespace(page_number=5)]
                    )
                ],
            ),
            related_units=[
                SimpleNamespace(
                    legal_unit_id="expanded",
                    locators=[SimpleNamespace(page_number=7)],
                    table_ids=["expanded-table"],
                )
            ],
            related_tables=[
                SimpleNamespace(
                    table_id="shared", parts=[SimpleNamespace(page_number=8)]
                )
            ],
            available_reference_ids=["referenced-only"],
        )
    ]
    identities = [
        "paragraph",
        "child",
        "expanded",
        "parent",
        "referenced-only",
        "unreturned",
    ]
    report = {
        "extraction_issues": [
            {"page_number": page, "issues": [f"warning-{page}"]}
            for page in (1, 4, 5, 7, 8, 117)
        ],
        "structure_issues": [
            {"legal_unit_id": identity, "issues": [identity]} for identity in identities
        ],
        "unresolved_references": [
            {"legal_unit_id": identity, "reference": {"raw_text": identity}}
            for identity in identities
        ],
        "omissions": [
            {"legal_unit_id": "paragraph", "table_ids": ["missing-table"]},
            {"legal_unit_id": "unreturned", "table_ids": ["unreturned-table"]},
        ],
        "uncovered_unit_ids": identities,
        "table_issues": [
            "inline",
            "expanded-table",
            "shared",
            "missing-table",
            "unreturned-table",
        ],
    }
    original = deepcopy(report)
    result = hit_diagnostics(hits, report)
    assert result["extraction_issues"] == report["extraction_issues"][1:5]
    assert result["structure_issues"] == report["structure_issues"][:3]
    assert result["unresolved_references"] == report["unresolved_references"][:3]
    assert result["uncovered_unit_ids"] == identities[:3]
    assert result["omissions"] == report["omissions"][:1]
    assert result["table_issues"] == report["table_issues"][:4]
    assert report == original


def test_no_hits_return_no_corpus_warnings():
    report = {
        "extraction_issues": [{"page_number": 117, "issues": ["needs_review"]}],
        "structure_issues": [
            {"legal_unit_id": "unreturned", "issues": ["needs_review"]}
        ],
        "unresolved_references": [{"legal_unit_id": "unreturned"}],
        "omissions": [{"legal_unit_id": "unreturned", "table_ids": ["rates"]}],
        "uncovered_unit_ids": ["unreturned"],
        "table_issues": ["rates"],
    }
    assert hit_diagnostics([], report) == {key: [] for key in report}


def test_missing_optional_report_fields_are_empty():
    result = hit_diagnostics([], {})
    assert set(result) == {
        "extraction_issues",
        "structure_issues",
        "unresolved_references",
        "omissions",
        "uncovered_unit_ids",
        "table_issues",
    }
    assert all(value == [] for value in result.values())
