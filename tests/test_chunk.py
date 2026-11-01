import json
from pathlib import Path

import pytest
import tiktoken

from atrag.ingestion.chunk import chunk_structure
from atrag.ingestion.parse_cirs import parse_cirs
from atrag.ingestion.services.sources import load_source_registry
from atrag.schemas.ingestion.extraction import ExtractedPage, ExtractedTable, TextBlock
from atrag.schemas.ingestion.fragments import ChunkParameters

FIXTURES = Path(__file__).parent / "fixtures/cirs"
SOURCE = load_source_registry(Path("data/sources/cirs.json"))
ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk(result, target=700, maximum=1200):
    return chunk_structure(
        result,
        SOURCE,
        parser_version="2",
        parameters=ChunkParameters(target_tokens=target, max_tokens=maximum),
        count_tokens=lambda text: len(ENCODING.encode_ordinary(text)),
    )


@pytest.fixture
def sample():
    pages = [
        ExtractedPage.model_validate(p)
        for p in json.loads((FIXTURES / "structure-pages.json").read_text())
    ]
    tables = [
        ExtractedTable.model_validate(t)
        for t in json.loads((FIXTURES / "structure-tables.json").read_text())["tables"]
    ]
    return parse_cirs(pages, tables)


def synthetic(*texts):
    blocks = [
        TextBlock(
            text=text,
            bbox=(260, 100 + i * 15, 335, 110 + i * 15),
            font_names=["Arial-BoldMT" if i < 3 else "ArialMT"],
        )
        for i, text in enumerate(
            [SOURCE.document_title, "Artigo 1.º", "Condições", *texts]
        )
    ]
    return parse_cirs(
        [
            ExtractedPage(
                snapshot_id="a" * 64,
                page_number=1,
                width=595,
                height=842,
                printed_page_labels=["i"],
                raw_text="\n".join(b.text for b in blocks),
                blocks=blocks,
                tables=[],
                images=[],
            )
        ],
        [],
    )


def test_manual_paragraphs_tables_and_distinct_instruments(sample):
    result = chunk(sample, target=2000, maximum=3000)
    by_id = {f.legal_unit_id: f for f in result.fragments}
    first = by_id["pt:cirs:code:article:2-A:paragraph:1"]
    unit = next(u for u in sample.units if u.legal_unit_id == first.legal_unit_id)
    assert first.text_original == unit.text_original
    assert first.text_normalized == unit.text_normalized
    assert first.page_start == 26 and first.page_end == 26
    assert first.printed_page_labels == ["25"]
    assert "Não" in first.embedding_text
    assert SOURCE.document_title in first.embedding_text
    assert "Delimitação negativa" in first.embedding_text
    assert "pt:cirs:code:article:2-A" in first.context_unit_ids
    assert any(i.endswith("subparagraph:f") for i in first.legal_unit_ids)
    assert "pt:decreto-lei:442-a/88:approving_decree:article:1" in by_id
    assert "pt:cirs:code:article:1:paragraph:1" in by_id
    for article, rate, rows in [("68", "44,60", 9), ("68-A", "2,5", 2)]:
        fragment = by_id[f"pt:cirs:code:article:{article}:paragraph:1"]
        assert rate in fragment.text_original
        assert len(fragment.tables[0].rows) == rows
        assert fragment.tables[0].review_status == "verified"
    assert by_id["pt:cirs:code:article:16:paragraph:8"].legal_status == "repealed"
    assert by_id["pt:cirs:code:article:16:paragraph:8"].editorial_notes
    assert (
        "Disposição transitória"
        not in by_id["pt:cirs:code:article:16:paragraph:1"].embedding_text
    )
    assert any(
        f.role == "transitional" and "Disposição transitória" in f.embedding_text
        for f in result.fragments
    )
    closing = [f for f in result.fragments if f.role == "closing"]
    assert len(closing) == 1
    assert "Visto e aprovado" in closing[0].text_original
    assert all(
        f.kind in ("article", "paragraph", "document", "note") for f in result.fragments
    )
    assert {f.role for f in result.fragments if f.kind == "note"} == {
        "closing",
        "transitional",
    }
    assert len(by_id) == len(result.fragments)


def test_long_paragraph_splits_with_intro_and_preserves_exception_in_source():
    result = synthetic(
        "1 - Só se aplica nas seguintes condições:",
        "a) " + "Rendimentos de trabalho. " * 35,
        "b) " + "Rendimentos de capital. " * 35,
        "c) Exceto quando não há rendimento.",
    )
    output = chunk(result, target=250, maximum=500)
    assert len(output.fragments) > 1
    for fragment in output.fragments:
        assert "Só se aplica nas seguintes condições:" in fragment.embedding_text
        assert fragment.token_count == len(
            ENCODING.encode_ordinary(fragment.embedding_text)
        )
        assert fragment.token_count <= 500
        assert fragment.legal_unit_id not in fragment.context_unit_ids
    assert not output.omissions
    assert output.fragments[0].text_original.startswith("1 -")
    assert (
        "".join(f.text_original for f in output.fragments)
        == result.units[2].text_original
    )
    assert "Exceto quando não há rendimento." in output.fragments[-1].text_original
    units = {u.legal_unit_id: u for u in result.units}
    for part in output.fragments:
        assert len(part.legal_unit_ids) == len(set(part.legal_unit_ids))
        assert all(
            units[i].text_original in part.text_original
            for i in part.legal_unit_ids
            if i != part.legal_unit_id
        )


def test_oversize_text_is_split_without_truncation():
    result = synthetic("1 - Não se aplica. " + "Condição indivisível; " * 150)
    output = chunk(result, target=100, maximum=150)
    assert len(output.fragments) > 1
    assert not output.omissions
    assert all(f.token_count <= 150 for f in output.fragments)
    assert (
        "".join(f.text_original for f in output.fragments)
        == result.units[2].text_original
    )


def test_unverified_table_excludes_only_affected_unit(sample):
    unsafe = sample.tables[1].model_copy(update={"review_status": "needs_review"})
    result = sample.model_copy(update={"tables": [sample.tables[0], unsafe]})
    output = chunk(result, target=2000, maximum=3000)
    assert any(f.article == "68" and f.tables for f in output.fragments)
    assert not any(f.article == "68-A" and f.tables for f in output.fragments)
    assert any(o.reason == "unverified_table" for o in output.omissions)
    assert all("25 |" not in f.embedding_text for f in output.fragments)


def test_identity_is_deterministic_and_sensitive_to_content_context_and_limits():
    result = synthetic("1 - Não se aplica ao número anterior.")
    first = chunk(result)
    assert first == chunk(result)
    fragment = first.fragments[0]
    assert "a" * 64 not in fragment.embedding_text
    assert "pt:cirs" not in fragment.embedding_text
    assert fragment.references[0].status == "unresolved"
    assert fragment.issues
    assert fragment.fragment_id != chunk(result, target=701).fragments[0].fragment_id
    changed = synthetic("1 - Não se aplica ao número seguinte.")
    assert fragment.fragment_id != chunk(changed).fragments[0].fragment_id


def test_no_overlap_between_neighboring_articles(sample):
    output = chunk(sample, target=2000, maximum=3000)
    fragment = next(f for f in output.fragments if f.article == "67")
    assert "Taxas gerais" not in fragment.embedding_text
    assert "2,5" not in fragment.embedding_text
    assert fragment.parent_id


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        ChunkParameters(target_tokens=100, max_tokens=99)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "orphan",
        "snapshot",
        "empty",
        "table",
        "duplicate_table",
        "child_text",
    ],
)
def test_invalid_structure_is_rejected_at_boundary(sample, damage):
    units, tables = list(sample.units), list(sample.tables)
    if damage == "duplicate":
        units.append(units[-1])
    elif damage == "orphan":
        units[-1] = units[-1].model_copy(update={"parent_id": "absent"})
    elif damage == "snapshot":
        units[-1] = units[-1].model_copy(update={"snapshot_id": "b" * 64})
    elif damage == "empty":
        units = []
    elif damage == "table":
        units[-1] = units[-1].model_copy(update={"table_ids": ["absent"]})
    elif damage == "duplicate_table":
        tables.append(tables[-1])
    else:
        index = next(i for i, u in enumerate(units) if u.kind == "paragraph")
        units[index] = units[index].model_copy(
            update={"text_original": "Absent from parent"}
        )
    with pytest.raises(ValueError):
        chunk(sample.model_copy(update={"units": units, "tables": tables}))


def test_exact_token_boundaries_keep_the_complete_paragraph():
    result = synthetic(
        "1 - Condições:", "a) Não há rendimento.", "b) Não há atividade."
    )
    size = chunk(result).fragments[0].token_count
    output = chunk(result, target=size, maximum=size)
    assert len(output.fragments) == 1
    assert not output.omissions
    assert output.fragments[0].text_original == result.units[2].text_original


def test_exception_in_another_paragraph_remains_available_as_original():
    result = synthetic(
        "1 - A regra aplica-se.", "2 - O número anterior não se aplica aos residentes."
    )
    output = chunk(result)
    first = next(f for f in output.fragments if f.paragraph == "1")
    assert "não se aplica aos residentes" not in first.embedding_text
    from atrag.retrieval.parts import unit_text

    exception = next(u for u in result.units if u.paragraph == "2")
    explicit = unit_text(exception, SOURCE, {u.legal_unit_id: u for u in result.units})
    assert "não se aplica aos residentes" in explicit.text_original
    assert explicit.complete


def test_primary_issues_and_technical_source_dates_do_not_enter_embedding():
    result = synthetic("1 - (Revogado pela Lei n.º 1/2020)")
    output = chunk(result)
    fragment = output.fragments[0]
    assert fragment.legal_status == "repealed"
    assert fragment.editorial_notes
    assert fragment.page_start == fragment.page_end == 1
    assert "2026-10-30" not in fragment.embedding_text
    assert "1/2020" in fragment.embedding_text
