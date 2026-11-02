"""Public chunking contract: bounded search parts and lossless original text."""

import json
from pathlib import Path

import pytest
import tiktoken

from atrag.ingestion.chunk import chunk_structure
from atrag.ingestion.services.sources import load_source_registry
from atrag.schemas.ingestion.extraction import ExtractedTable, TablePart
from atrag.schemas.ingestion.fragments import ChunkParameters
from atrag.schemas.ingestion.structure import LegalUnit, StructureResult

SOURCE = load_source_registry(Path("data/sources/cirs.json"))
ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk(structure, target=700, maximum=1200):
    return chunk_structure(
        structure,
        SOURCE,
        parser_version="2",
        parameters=ChunkParameters(target_tokens=target, max_tokens=maximum),
        count_tokens=lambda text: len(ENCODING.encode_ordinary(text)),
    )


@pytest.fixture
def article16():
    raw = json.loads(
        (Path(__file__).parent / "fixtures/cirs/article16-units.json").read_text()
    )
    units = [LegalUnit.model_validate(u) for u in raw]
    # Keep the independently preserved source fixture intact; this isolated
    # structure has no preceding chapter/section from the rest of the corpus.
    units[0] = units[0].model_copy(update={"parent_id": None})
    return StructureResult(
        units=units, tables=[], excluded_block_count=0, extraction_issues=[]
    )


def one_paragraph(article16, text):
    paragraph = article16.units[1].model_copy(
        update={"text_original": text, "text_normalized": text, "table_ids": []}
    )
    article = article16.units[0].model_copy(
        update={"text_original": "Artigo 16.º\nResidência\n" + text}
    )
    return article16.model_copy(update={"units": [article, paragraph]})


def test_every_article16_paragraph_is_primary_and_note_is_independent(article16):
    result = chunk(article16)
    assert not result.omissions
    for unit in article16.units:
        if unit.kind not in ("paragraph", "note"):
            continue
        parts = [f for f in result.fragments if f.legal_unit_id == unit.legal_unit_id]
        assert parts, unit.legal_unit_id
        assert "".join(f.text_original for f in parts) == unit.text_original
        assert all(f.token_count <= 1200 for f in parts)
    paragraphs = [f for f in result.fragments if f.kind == "paragraph"]
    assert {f.paragraph for f in paragraphs} == {str(n) for n in range(1, 17)}
    assert all("Disposição transitória" not in f.embedding_text for f in paragraphs)
    assert next(f for f in paragraphs if f.paragraph == "8").legal_status == "repealed"
    assert {f.role for f in result.fragments if f.kind == "note"} == {"transitional"}


@pytest.mark.parametrize(
    "text",
    [
        "1 - " + "Condição jurídica.\n" * 350,
        "1 - " + "ação não há isenção; " * 400 + "fim.",
        "1 - " + "çã€" * 1500,
        "1 - " + "Repetição exata.  \n\n" * 350 + "  ",
    ],
)
def test_large_text_has_exact_adjacent_ranges_and_round_trips(article16, text):
    structure = one_paragraph(article16, text)
    result = chunk(structure, target=180, maximum=230)
    assert not result.omissions
    assert len(result.fragments) > 1
    cursor = 0
    for index, part in enumerate(result.fragments):
        assert part.source_start == cursor
        assert part.source_end > cursor
        assert part.text_original == text[part.source_start : part.source_end]
        assert part.part_index == index
        assert part.part_count == len(result.fragments)
        assert part.token_count == len(ENCODING.encode_ordinary(part.embedding_text))
        assert part.token_count <= 230
        assert part.snapshot_id == structure.units[1].snapshot_id
        assert part.locators == structure.units[1].locators
        cursor = part.source_end
    assert cursor == len(text)
    assert "".join(f.text_original for f in result.fragments) == text
    assert result == chunk(structure, target=180, maximum=230)


def test_exact_limit_keeps_whole_source_and_one_less_splits(article16):
    structure = one_paragraph(article16, "1 - " + "Ação tributária. " * 70)
    size = chunk(structure).fragments[0].token_count
    whole = chunk(structure, target=size, maximum=size).fragments
    split = chunk(structure, target=size - 1, maximum=size - 1).fragments
    assert len(whole) == 1
    assert len(split) > 1
    assert "".join(f.text_original for f in split) == whole[0].text_original
    assert all(f.token_count <= size - 1 for f in split)


def test_impossible_envelope_fails_preparation_explicitly(article16):
    structure = one_paragraph(article16, "1 - Sim.")
    with pytest.raises(ValueError, match="[Ee]nvelope"):
        chunk(structure, target=1, maximum=1)


def table_structure(article16, rows):
    headers = [["Rendimento (euros)", "Taxa (percentagem)"]]
    text = "1 - Taxas:\n[Table rates]\n" + "\n".join(
        " | ".join(cell.strip() for cell in row) for row in [*headers, *rows]
    )
    structure = one_paragraph(article16, text)
    structure.units[1] = structure.units[1].model_copy(update={"table_ids": ["rates"]})
    table = ExtractedTable(
        table_id="rates",
        parts=[
            TablePart(
                part_id="rates:1",
                table_id="rates",
                page_number=48,
                bbox=(80, 100, 500, 750),
                rows=[*headers, *rows],
            )
        ],
        header_rows=headers,
        rows=rows,
        context="Taxas em euros e percentagem",
        review_status="verified",
    )
    return structure.model_copy(update={"tables": [table]})


def test_table_parts_keep_whole_rows_headers_and_exact_source(article16):
    rows = [[f"De {n * 1000} até {(n + 1) * 1000}", "12,50"] for n in range(40)]
    structure = table_structure(article16, rows)
    result = chunk(structure, target=200, maximum=260)
    assert not result.omissions
    assert len(result.fragments) > 1
    delivered = []
    cursor = 0
    for fragment in result.fragments:
        assert fragment.source_start == cursor
        cursor = fragment.source_end
        assert fragment.token_count <= 260
        for table, span in zip(fragment.tables, fragment.table_ranges, strict=True):
            assert table.table_id == span.table_id == "rates"
            assert table.rows == rows[span.row_start : span.row_end]
            assert table.header_rows == structure.tables[0].header_rows
            assert all(not part.rows for part in table.parts)
            assert "Rendimento (euros)" in fragment.embedding_text
            assert "Taxa (percentagem)" in fragment.embedding_text
            from atrag.ingestion.table_parts import table_text

            assert len(ENCODING.encode_ordinary(table_text(fragment.tables))) <= 260
            delivered.extend(table.rows)
    assert delivered == rows
    assert (
        "".join(f.text_original for f in result.fragments)
        == structure.units[1].text_original
    )


def test_an_indivisible_table_row_has_an_explicit_omission(article16):
    structure = table_structure(article16, [["Valor " * 500, "12,50"]])
    result = chunk(structure, target=120, maximum=180)
    assert not result.fragments
    assert result.omissions[0].reason == "table_row_over_token_limit"
    assert result.omissions[0].table_ids == ["rates"]


def test_reassembly_checks_source_identity_order_gaps_and_text(article16):
    from atrag.retrieval.parts import reassemble_unit

    structure = one_paragraph(article16, "1 - " + "Ação tributária. \n" * 150)
    unit = structure.units[1]
    parts = chunk(structure, target=140, maximum=200).fragments
    assert reassemble_unit(unit, parts) == unit.text_original
    bad_parts = [
        [],
        parts[1:],
        list(reversed(parts)),
        [parts[0], *parts],
        [parts[0].model_copy(update={"snapshot_id": "b" * 64}), *parts[1:]],
        [parts[0].model_copy(update={"text_original": "wrong"}), *parts[1:]],
        [
            *parts[:-1],
            parts[-1].model_copy(
                update={
                    "source_end": parts[-1].source_end - 1,
                    "text_original": parts[-1].text_original[:-1],
                }
            ),
        ],
        [
            parts[0].model_copy(update={"source_end": parts[0].source_end + 1}),
            *parts[1:],
        ],
    ]
    for damaged in bad_parts:
        with pytest.raises(ValueError):
            reassemble_unit(unit, damaged)
    for updates in [
        {"source_end": 0, "text_original": ""},
        {"part_index": parts[0].part_count},
    ]:
        with pytest.raises(ValueError):
            reassemble_unit(unit, [parts[0].model_copy(update=updates), *parts[1:]])
    from atrag.ingestion.part_validation import validate_part

    for updates in [
        {"source_start": 0, "source_end": 0, "text_original": ""},
        {"part_index": parts[0].part_count},
    ]:
        with pytest.raises(ValueError):
            validate_part(unit, parts[0].model_copy(update=updates))


def test_explicit_original_and_continuation_include_citation_and_note_address(
    article16,
):
    from atrag.retrieval.parts import unit_text

    units = {u.legal_unit_id: u for u in article16.units}
    unit = next(u for u in article16.units if u.paragraph == "8")
    full = unit_text(unit, SOURCE, units, context_tokens=6000)
    assert full.text_original == unit.text_original
    assert full.complete
    assert full.available_note_ids == [article16.units[-1].legal_unit_id]
    assert full.source.source_url == SOURCE.source_url
    assert full.locators == unit.locators
    assert full.context_tokens_used == len(
        ENCODING.encode_ordinary(
            "\n\n".join(
                [
                    full.citation_label,
                    *[n.raw_text for n in unit.editorial_notes],
                    full.text_original,
                ]
            )
        )
    )
    note = article16.units[-1]
    start, delivered = 0, []
    while True:
        result = unit_text(note, SOURCE, units, start=start, context_tokens=250)
        assert result.role == "transitional"
        assert result.context_tokens_used <= 250
        assert result.source_start == start
        assert not result.complete
        delivered.append(result.text_original)
        if result.next_start is None:
            break
        assert result.next_start > start
        start = result.next_start
    assert "".join(delivered) == note.text_original
    article = unit_text(article16.units[0], SOURCE, units)
    assert article.complete
    assert article.text_original == article16.units[0].text_original
    assert note.text_original not in article.text_original
    assert article.available_reference_ids
    assert "pt:cirs:code:article:16:paragraph:15" in article.available_reference_ids
    assert (
        "pt:cirs:code:article:101:paragraph:1:subparagraph:d"
        not in article.available_reference_ids
    )


def test_neighbor_selection_and_original_boundaries_are_explicit(article16):
    from atrag.retrieval.parts import neighbor_parts, unit_text

    structure = one_paragraph(article16, "1 - " + "Condição tributária. " * 150)
    unit = structure.units[1]
    parts = chunk(structure, target=130, maximum=190).fragments
    selected = neighbor_parts(unit, parts, 1, before=1, after=1)
    assert selected.parts == parts[:3]
    assert not selected.complete
    assert not selected.omitted_part_indices
    assert neighbor_parts(unit, parts, 1).parts == parts[1:3]
    all_parts = neighbor_parts(unit, parts, 0, after=len(parts))
    assert all_parts.complete
    assert all_parts.parts == parts
    expected = sum(
        len(
            ENCODING.encode_ordinary(
                json.dumps(
                    {
                        "text": p.embedding_text,
                        "notes": [n.raw_text for n in p.editorial_notes],
                        "related_units": [],
                        "tables": [],
                    },
                    ensure_ascii=False,
                )
            )
        )
        for p in parts
    )
    assert all_parts.context_tokens_used == expected
    limited = neighbor_parts(unit, parts, 0, after=1, context_tokens=1)
    assert not limited.parts
    assert limited.omitted_part_indices == [0, 1]
    single = neighbor_parts(unit, parts, 0, after=0)
    boundary = neighbor_parts(
        unit, parts, 0, after=1, context_tokens=single.context_tokens_used
    )
    assert boundary.parts == parts[:1]
    assert boundary.omitted_part_indices == [1]
    assert boundary.context_tokens_used == single.context_tokens_used
    with pytest.raises(ValueError):
        neighbor_parts(unit, parts, 0, context_tokens=0)
    for kwargs in [
        {"start": -1},
        {"start": len(unit.text_original)},
        {"context_tokens": 0},
    ]:
        with pytest.raises(ValueError):
            unit_text(unit, SOURCE, {}, **kwargs)
    for index, kwargs in [
        (-1, {}),
        (len(parts), {}),
        (0, {"before": -1}),
        (0, {"after": -1}),
    ]:
        with pytest.raises(ValueError):
            neighbor_parts(unit, parts, index, **kwargs)


def test_search_reports_actual_part_ranges_and_unit_completeness(article16):
    from atrag.retrieval.parts import hit_coverage
    from atrag.schemas.retrieval import SearchHit

    structure = one_paragraph(article16, "1 - " + "Condição tributária. " * 150)
    parts = chunk(structure, target=130, maximum=190).fragments
    hits = [
        SearchHit(
            fragment_id=p.fragment_id,
            score=1,
            fragment=p,
            related_units=[],
            related_tables=[],
        )
        for p in parts
    ]
    partial = hit_coverage(hits[:1])[0]
    assert partial.part_count == len(parts)
    assert partial.returned_part_indices == [0]
    assert partial.source_ranges == [(0, parts[0].source_end)]
    assert not partial.complete
    full = hit_coverage(hits)[0]
    assert full.complete
    assert full.legal_unit_id == structure.units[1].legal_unit_id
    assert full.returned_part_indices == list(range(len(parts)))
    assert full.source_ranges[-1][1] == len(structure.units[1].text_original)
    assert hit_coverage([]) == []


def test_preferred_boundaries_and_soft_target_are_observable():
    from atrag.ingestion.text_parts import text_ranges

    text = "Regra curta. Outra regra. Fim."
    assert text_ranges(text, lambda a, b: b - a, 13, 20) == [
        (0, 13),
        (13, 26),
        (26, 30),
    ]
    text = "1 - Intro:\na) Uma regra.\nb) Outra regra."
    ranges = text_ranges(text, lambda a, b: b - a, 25, 40)
    assert ranges[0][1] == text.index("b)")
    assert "".join(text[a:b] for a, b in ranges) == text
    text = "1 - Intro:\na) A\nb) B\nc) C"
    assert text_ranges(text, lambda a, b: b - a, 19, 30)[0][1] == text.index("b)")
    # Later parts have an envelope that already occupies the soft target.
    assert text_ranges("abcd", lambda a, b: b - a + (2 if a else 0), 2, 4) == [
        (0, 2),
        (2, 4),
    ]
    with pytest.raises(ValueError, match="source character"):
        text_ranges("aΩ", lambda a, b: 3 if "Ω" in "aΩ"[a:b] else b - a, 1, 2)


def test_table_spans_cover_headers_rows_newlines_and_trailing_text(article16):
    from atrag.ingestion.table_parts import (
        table_rows,
        table_text,
        validate_table_boundaries,
    )

    structure = table_structure(article16, [["1000", "12,50"], ["2000", "20,00"]])
    # A table may start at the first character and may have trailing conditions.
    text = structure.units[1].text_original.removeprefix("1 - Taxas:\n")
    text += "\nCondição posterior. " * 100
    unit = structure.units[1].model_copy(update={"text_original": text})
    article = structure.units[0].model_copy(
        update={"text_original": "Artigo 16.º\nResidência\n" + text}
    )
    structure = structure.model_copy(update={"units": [article, unit]})
    table = structure.tables[0]
    spans = table_rows(unit, {table.table_id: table})
    first_end = text.index("2000")
    second_end = text.index("\nCondição") + 1
    assert [(row.start, row.end) for row in spans] == [
        (0, first_end),
        (first_end, second_end),
    ]
    for span in spans:
        validate_table_boundaries(spans, span.start, span.end)
    output = chunk(structure, target=130, maximum=190)
    assert not output.omissions
    assert "".join(p.text_original for p in output.fragments) == text
    assert all(p.token_count <= 190 for p in output.fragments)
    assert "ç" in table_text([table.model_copy(update={"header_rows": [["Condição"]]})])


def test_table_row_at_the_exact_limit_is_accepted(article16):
    from atrag.ingestion.table_parts import (
        RowSpan,
        TableRowTooLarge,
        bounded_table_ranges,
    )

    text = "12345"
    row = RowSpan("rates", 0, 0, 5)
    assert bounded_table_ranges(text, [row], lambda a, b: b - a, 5, 5) == [(0, 5)]
    assert bounded_table_ranges(
        "1\n2\n",
        [RowSpan("rates", 0, 0, 2), RowSpan("rates", 1, 2, 4)],
        lambda a, b: b - a,
        4,
        4,
    ) == [(0, 4)]
    with pytest.raises(TableRowTooLarge):
        bounded_table_ranges(text, [row], lambda a, b: b - a, 4, 4)


def test_table_source_mismatches_and_split_rows_are_rejected(article16):
    from atrag.ingestion.table_parts import table_rows, validate_table_boundaries

    structure = table_structure(article16, [["1000", "12,50"], ["2000", "20,00"]])
    unit, table = structure.units[1], structure.tables[0]
    rows = table_rows(unit, {table.table_id: table})
    for start, end in [
        (rows[0].start + 1, rows[0].end),
        (rows[0].start, rows[0].end - 1),
    ]:
        with pytest.raises(ValueError, match="table row"):
            validate_table_boundaries(rows, start, end)
    for wrong in [
        table.model_copy(update={"rows": [*table.rows, ["3000", "30,00"]]}),
        table.model_copy(update={"rows": table.rows[:1]}),
        table.model_copy(update={"rows": [["wrong"], *table.rows[1:]]}),
    ]:
        with pytest.raises(ValueError, match="rows"):
            chunk(structure.model_copy(update={"tables": [wrong]}))
    with pytest.raises(ValueError, match="absent"):
        table_rows(
            unit.model_copy(update={"text_original": "missing"}),
            {table.table_id: table},
        )


def test_small_soft_target_uses_maximum_but_cannot_split_one_character(article16):
    structure = one_paragraph(article16, "1 - " + "çã€" * 100)
    result = chunk(structure, target=1, maximum=120)
    assert (
        "".join(p.text_original for p in result.fragments)
        == structure.units[1].text_original
    )
    assert all(p.token_count <= 120 for p in result.fragments)
    from atrag.ingestion.text_parts import text_ranges

    assert text_ranges("€", lambda a, b: 1 + 4 * (b - a), 2, 5) == [(0, 1)]
    assert text_ranges("x", lambda a, b: 5, 5, 5) == [(0, 1)]
    with pytest.raises(ValueError, match="source character"):
        text_ranges("€", lambda a, b: 10 * (b - a), 1, 2)
