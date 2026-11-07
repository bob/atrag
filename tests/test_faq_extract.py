import json
from pathlib import Path

import pytest
from fixtures.faq_pipeline import card

from atrag.ingestion.faq_extract import parse_faq_page

FIXTURES = Path(__file__).parent / "fixtures" / "faq"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())["records"]
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())["fixtures"]
URL = EXPECTED[0]["page_url"]


def collapsed(text):
    return " ".join(text.split())


@pytest.mark.parametrize(
    "fixture", [f for f in MANIFEST if f["faq_ids"]], ids=lambda f: f["file"]
)
def test_reviewed_records_keep_complete_content_and_locations(fixture):
    expected = [r for r in EXPECTED if r["fixture"] == fixture["file"]]
    page = parse_faq_page(
        (FIXTURES / fixture["file"]).read_bytes(),
        fixture["page_url"],
        tuple(expected[0]["topic_path"]),
    )
    assert page.detected_count == len(expected) == len(page.faqs)
    assert page.issues == ()
    assert [r.entry.faq_id for r in page.faqs] == fixture["faq_ids"]
    for ordinal, (record, reference) in enumerate(
        zip(page.faqs, expected, strict=True), 1
    ):
        entry = record.entry
        assert entry.entry_id == reference["entry_id"]
        assert collapsed(entry.question) == collapsed(reference["question"])
        assert collapsed(entry.answer) == collapsed(reference["answer"])
        assert entry.anchor == reference["anchor"]
        assert str(entry.page_url) == reference["page_url"]
        assert entry.topic_path == tuple(reference["topic_path"])
        assert entry.ordinal == ordinal
        assert [link.model_dump(mode="json") for link in entry.links] == reference[
            "links"
        ]
        tables = [t for t in page.tables if t.table_id in record.table_ids]
        assert [
            [[c.model_dump() for c in row] for row in t.parts[0].cells] for t in tables
        ] == reference["tables"]
        assert [collapsed(i.text) for i in record.list_items if i.ordered] == reference[
            "ordered_list_items"
        ]
        assert record.answer_normalized == collapsed(entry.answer)


def parse(html):
    return parse_faq_page(
        f'<div id="faqAccordion">{html}</div>'.encode(), URL, ("IRS", "Topic")
    )


def test_reordering_only_changes_ordinal_and_menu_is_excluded():
    first = card() + card("43", position=2)
    a = parse(first)
    b = parse(card("43", position=2) + card())
    assert {r.entry.entry_id for r in a.faqs} == {r.entry.entry_id for r in b.faqs}
    assert [r.entry.ordinal for r in b.faqs] == [1, 2]
    page = parse_faq_page(
        ("<nav>Menu</nav>" + first + "<footer>SharePoint</footer>").encode(),
        URL,
        ("IRS",),
    )
    assert page.faqs[0].entry.answer == "First condition.\nSecond condition."


@pytest.mark.parametrize(
    "html,kind",
    [
        (card(answer="<p> </p>"), "empty_answer"),
        (card(number=""), "invalid_question"),
        (card().replace('id="collapse1"', 'id="other"'), "answer_binding"),
        (
            card().replace('aria-labelledby="heading1"', 'aria-labelledby="other"'),
            "answer_binding",
        ),
        (card() + card(position=2), "duplicate_faq_id"),
        (card() + card("43"), "duplicate_dom_id"),
        (card(answer='<img src="rates.png" alt="Rates">'), "unsupported_element"),
        (
            card(answer='<table><tr><td colspan="bad">Rates</td></tr></table>'),
            "unsupported_table",
        ),
    ],
)
def test_losses_and_unsupported_content_have_located_issues(html, kind):
    page = parse(html)
    assert any(i.kind == kind and i.ordinal is not None for i in page.issues)
    assert page.detected_count == len(page.faqs) + len(page.rejected_ordinals)
    assert page.review_status == "needs_review"


def test_unknown_template_is_reported():
    page = parse_faq_page(b"<main>Server error</main>", URL, ("IRS",))
    assert page.faqs == ()
    assert [i.kind for i in page.issues] == ["unknown_template"]


def test_answer_without_a_question_blocks_verification():
    page = parse(card() + '<div class="card-body">Unpaired answer</div>')
    assert page.review_status == "needs_review"
    assert any(i.kind == "answer_count_mismatch" for i in page.issues)


@pytest.mark.parametrize(
    "html",
    [
        card().replace('id="heading1"', ""),
        card().replace('aria-controls="collapse1"', ""),
        card().replace('href="#collapse1"', 'href="#wrong"'),
        card().replace('class="card-body"', 'class="other"'),
        card(answer='<div class="card-header">Another question</div>'),
    ],
)
def test_ambiguous_answer_binding_is_rejected(html):
    page = parse(html)
    assert not page.faqs
    assert any(i.kind == "answer_binding" for i in page.issues)


def test_comments_and_active_content_are_excluded_from_answer():
    page = parse(
        card(
            answer="<p>Visible<!-- hidden --><script>script</script><style>style</style></p>"
        )
    )
    assert page.faqs[0].entry.answer == "Visible"


@pytest.mark.parametrize(
    "answer",
    [
        '<a href="javascript:alert(1)">Unsafe link</a>',
        "<li>Item without a list</li>",
    ],
)
def test_invalid_content_is_explained(answer):
    page = parse(card(answer=answer))
    assert not page.faqs
    assert page.rejected_ordinals == (1,)
    assert any(i.kind == "invalid_content" for i in page.issues)


@pytest.mark.parametrize(
    "table",
    [
        "<table>Text without rows</table>",
        "<table><tr></tr></table>",
        "<table><tr><td><table><tr><td>Nested</td></tr></table></td></tr></table>",
    ],
)
def test_unsupported_table_shape_is_reported(table):
    page = parse(card(answer="Before" + table))
    assert any(i.kind == "unsupported_table" for i in page.issues)
    assert page.faqs[0].entry.answer.startswith("Before")


def test_unsupported_image_with_visible_text_keeps_text_and_requires_review():
    page = parse(card(answer='<p>See rates</p><img src="rates.png">'))
    assert page.faqs[0].entry.answer == "See rates"
    assert page.review_status == "needs_review"


def test_reference_mentions_remain_separate_from_faq_identity():
    page = parse(card(answer="Nos termos do artigo 53.º do CIVA."))
    [record] = page.faqs
    assert record.references == ("artigo 53.º",)
    assert record.entry.entry_id == "faqs-00267:42"


def test_same_published_number_on_another_page_has_another_identity():
    one = parse(card()).faqs[0].entry
    two = (
        parse_faq_page(card().encode(), URL.replace("00267", "00720"), ("IVA",))
        .faqs[0]
        .entry
    )
    assert one.entry_id != two.entry_id


def test_tables_share_contract_without_pdf_fields_and_keep_spans():
    page = parse(
        card(
            answer='<table><tr><th rowspan="2">A</th><th>B</th></tr><tr><td>C</td></tr></table>'
        )
    )
    [table] = page.tables
    part = table.parts[0]
    assert str(part.location.page_url) == URL
    assert part.location.anchor == "heading1"
    assert part.cells[0][0].rowspan == 2
    assert table.header_rows == [["A", "B"]]
    assert table.rows == [["C"]]
    assert "page_number" not in part.model_dump()
    assert "bbox" not in part.model_dump()
    assert table.review_status == "verified"
    assert type(table).model_validate_json(table.model_dump_json()) == table


def test_saved_collection_extraction_is_complete_and_repeatable(faq_snapshot, tmp_path):
    from atrag.ingestion.services.faq_extract import extract_collection

    first = extract_collection(faq_snapshot, tmp_path)
    second = extract_collection(faq_snapshot, tmp_path)
    assert first["directory"] != second["directory"]
    assert first["page_count"] == 2
    assert first["question_count"] == first["extracted_count"] == 1
    assert first["review_status"] == "verified"
    for name in ("faqs.jsonl", "tables.json", "report.json"):
        assert (Path(first["directory"]) / name).read_bytes() == (
            Path(second["directory"]) / name
        ).read_bytes()
    record = json.loads((Path(first["directory"]) / "faqs.jsonl").read_bytes())
    assert record["entry"]["topic_path"] == ["IRS", "Topic"]
    assert record["entry"]["entry_id"] == "faqs-00267:42"


def test_corrupt_collection_cannot_produce_extracted_records(faq_snapshot, tmp_path):
    from atrag.ingestion.services.faq_extract import extract_collection

    page = faq_snapshot.collection.pages[1]
    (tmp_path / faq_snapshot.local_path.parent / page.local_path).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        extract_collection(faq_snapshot, tmp_path)
    assert not list((tmp_path / "runs").glob("*/faqs.jsonl"))
    manifests = [
        json.loads(p.read_text()) for p in (tmp_path / "runs").glob("*/manifest.json")
    ]
    assert any(
        m["stage"] == "extract_faq" and m["status"] == "failed"
        for m in manifests
        if "stage" in m
    )


def test_changed_snapshot_metadata_is_rejected(faq_snapshot, tmp_path):
    from datetime import timedelta

    from atrag.ingestion.services.faq_extract import extract_collection

    changed = faq_snapshot.model_copy(
        update={"ingested_at": faq_snapshot.ingested_at + timedelta(seconds=1)}
    )
    with pytest.raises(ValueError, match="manifest changed"):
        extract_collection(changed, tmp_path)


@pytest.mark.parametrize(
    "faq_snapshot", [{"final_url": URL.replace("00267", "00720")}], indirect=True
)
def test_redirect_uses_actual_page_identity_and_location(faq_snapshot, tmp_path):
    from atrag.ingestion.services.faq_extract import extract_collection

    result = extract_collection(faq_snapshot, tmp_path)
    record = json.loads((Path(result["directory"]) / "faqs.jsonl").read_bytes())
    assert record["entry"]["page_url"] == URL.replace("00267", "00720")
    assert record["entry"]["entry_id"] == "faqs-00720:42"


@pytest.mark.parametrize(
    "faq_snapshot", [{"extra_url": URL + "?page=2"}], indirect=True
)
def test_collection_identity_conflicts_require_review(faq_snapshot, tmp_path):
    from atrag.ingestion.services.faq_extract import extract_collection

    result = extract_collection(faq_snapshot, tmp_path)
    assert result["review_status"] == "needs_review"
    assert result["conflicting_entry_ids"] == ["faqs-00267:42"]


@pytest.mark.parametrize(
    "faq_snapshot",
    [{"title": ""}, {"answer": '<p>Rates</p><img src="rates.png">'}],
    indirect=True,
)
def test_collection_report_blocks_incomplete_content(faq_snapshot, tmp_path):
    from atrag.ingestion.services.faq_extract import extract_collection

    result = extract_collection(faq_snapshot, tmp_path)
    assert result["review_status"] == "needs_review"
    report = json.loads((Path(result["directory"]) / "report.json").read_bytes())
    assert any(page["issues"] for page in report["pages"])
    assert (
        report["question_count"] == report["extracted_count"] + report["rejected_count"]
    )


def test_cli_extract_uses_saved_input_without_network(
    faq_snapshot, tmp_path, monkeypatch, capsys
):
    from atrag.cli import main
    from atrag.settings import PROJECT_ROOT

    monkeypatch.setenv("ATRAG_DATA_DIR", str(tmp_path))
    main(
        [
            "ingest",
            "extract",
            "--registry",
            str(PROJECT_ROOT / "data/sources/faq.json"),
            "--source",
            str(tmp_path / faq_snapshot.local_path),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["extracted_count"] == 1
    assert (Path(result["directory"]) / "faqs.jsonl").exists()
