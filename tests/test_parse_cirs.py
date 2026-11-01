import json
from pathlib import Path

import pytest

from atrag.ingestion.parse_cirs import parse_cirs
from atrag.schemas.ingestion.extraction import ExtractedPage, ExtractedTable, TextBlock

FIXTURES = Path(__file__).parent / "fixtures/cirs"
SNAPSHOT = "a" * 64
TITLE = "Código do Imposto sobre o Rendimento das Pessoas Singulares"


def block(text, top=100, *, heading=False):
    return TextBlock(
        text=text,
        bbox=(260 if heading else 85, top, 420 if heading else 510, top + 10),
        font_names=["Arial-BoldMT" if heading else "ArialMT"],
        font_sizes=[10],
    )


def page(*blocks, number=1):
    return ExtractedPage(
        snapshot_id=SNAPSHOT,
        page_number=number,
        width=595,
        height=842,
        printed_page_labels=[str(number - 1)],
        raw_text="\n".join(b.text for b in blocks),
        blocks=list(blocks),
        tables=[],
        images=[],
    )


def parse(*blocks):
    return parse_cirs([page(block(TITLE, 70, heading=True), *blocks)], [])


def units_by_id(result):
    return {u.legal_unit_id: u for u in result.units}


def test_manual_cirs_sample():
    pages = [
        ExtractedPage.model_validate(p)
        for p in json.loads((FIXTURES / "structure-pages.json").read_text())
    ]
    tables = [
        ExtractedTable.model_validate(t)
        for t in json.loads((FIXTURES / "structure-tables.json").read_text())["tables"]
    ]
    result = parse_cirs(pages, tables)
    units = units_by_id(result)
    expected = [
        (
            "pt:decreto-lei:442-a/88:approving_decree:article:1",
            "Aprovação do Código",
            7,
            7,
        ),
        (
            "pt:decreto-lei:442-a/88:approving_decree:article:16",
            "As modificações do Código",
            11,
            11,
        ),
        ("pt:cirs:code:article:1", "Base do imposto", 23, 23),
        (
            "pt:cirs:code:article:2-A",
            "Delimitação negativa dos rendimentos da categoria A",
            26,
            27,
        ),
        ("pt:cirs:code:article:16", "Residência", 48, 49),
        ("pt:cirs:code:article:67", "Revisão dos atos de fixação", 81, 81),
        ("pt:cirs:code:article:68", "Taxas gerais", 81, 82),
        ("pt:cirs:code:article:68-A", "Taxa adicional de solidariedade", 82, 82),
    ]
    for key, title, start, end in expected:
        unit = units[key]
        assert (unit.unit_title, unit.page_start, unit.page_end) == (title, start, end)
        assert unit.snapshot_id == pages[0].snapshot_id
        assert unit.code == ("CIRS" if unit.document_scope == "code" else None)
    assert len(units) == len(result.units)
    assert all(u.page_start != 2 for u in result.units)
    assert units["pt:cirs:code:article:1"].section == "I"
    assert units["pt:cirs:code:article:68"].chapter == "III"
    assert "Visto e aprovado" not in units[expected[1][0]].text_original
    assert any(u.role == "preamble" for u in result.units)
    seventh = units["pt:cirs:code:article:16:paragraph:7"]
    assert (seventh.page_start, seventh.page_end) == (48, 49)
    assert seventh.printed_page_labels == ["47", "48"]
    assert seventh.citation_label == "CIRS, artigo 16.º, n.º 7"
    assert "distinto daqueles" in seventh.text_normalized
    for n in range(8, 13):
        unit = units[f"pt:cirs:code:article:16:paragraph:{n}"]
        assert unit.legal_status == "repealed"
        assert unit.editorial_notes[0].kind == "repeal"
    notes = [
        u
        for u in result.units
        if u.parent_id == "pt:cirs:code:article:16" and u.kind == "note"
    ]
    assert len(notes) == 1
    assert notes[0].role == "transitional"
    assert notes[0].page_end == 50
    assert "Lei n.º 82/2023" in notes[0].text_original
    assert (
        "Disposição transitória" not in units["pt:cirs:code:article:16"].text_original
    )
    first = units["pt:cirs:code:article:68:paragraph:1"]
    assert first.page_end == 82
    assert first.table_ids == [tables[0].table_id]
    assert len(tables[0].rows) == 9
    assert "44,60" in first.text_original
    assert first.text_original.count("44,60") == 1
    assert "73-A/2025" in first.editorial_notes[0].raw_text
    solidarity = units["pt:cirs:code:article:68-A:paragraph:1"]
    assert solidarity.table_ids == [tables[1].table_id]
    assert "2,5" in solidarity.text_original
    assert units[
        "pt:cirs:code:article:2-A:paragraph:1:subparagraph:f"
    ].parent_id.endswith("paragraph:1")


def test_heading_layout_mentions_unnumbered_article_and_nested_items():
    result = parse(
        block("Artigo 1.º Base do imposto ........ 22"),
        block("Artigo 1.º"),  # A body mention alone is not a heading.
        block("Artigo único", 140, heading=True),
        block("Regra especial", 152, heading=True),
        block("1 - Não se aplica:", 170),
        block("a) Com exceção de:", 190),
        block("1) Um caso;", 210),
        block("ii) Uma subdivisão.", 230),
    )
    units = units_by_id(result)
    article = units["pt:cirs:code:article:unico"]
    assert article.unit_title == "Regra especial"
    assert "Não se aplica" in article.text_normalized
    assert "pt:cirs:code:article:1" not in units
    nested = units[
        "pt:cirs:code:article:unico:paragraph:1:subparagraph:a:item:1:item:ii"
    ]
    assert nested.text_original == "ii) Uma subdivisão."
    assert nested.locators[0].bbox == (85, 230, 510, 240)


def test_repeal_belongs_to_own_unit_and_amendment_stays_located():
    units = units_by_id(
        parse(
            block("Artigo 2.º-A", heading=True),
            block("Título", 112, heading=True),
            block("1 - Regra:", 130),
            block("a) Não é revogado o benefício.", 150),
            block("b) (Revogada pela Lei n.º 1/2020)", 170),
            block("2 – Texto. (Redação da Lei n.º 2/2021)", 190),
        )
    )
    assert units["pt:cirs:code:article:2-A"].legal_status == "unknown"
    assert units["pt:cirs:code:article:2-A:paragraph:1"].legal_status == "unknown"
    assert (
        units["pt:cirs:code:article:2-A:paragraph:1:subparagraph:a"].legal_status
        == "unknown"
    )
    assert (
        units["pt:cirs:code:article:2-A:paragraph:1:subparagraph:b"].legal_status
        == "repealed"
    )
    note = units["pt:cirs:code:article:2-A:paragraph:2"].editorial_notes[0]
    assert (note.kind, note.locators[0].page_number) == ("amendment", 1)


def test_relative_exact_external_and_ambiguous_references():
    result = parse(
        block("Artigo 1.º", heading=True),
        block("Um", 112, heading=True),
        block("1 - Sem aplicação do número anterior e do artigo 999.º.", 130),
        block("2 - Nos termos do número anterior e do artigo seguinte.", 150),
        block("3 - Nos termos do artigo 1.º do Código do IRC e do artigo 2.º-A.", 170),
        block("4 - Ver os n.os 1 e 2 e a alínea anterior.", 190),
        block("Artigo 2.º-A", 220, heading=True),
        block("Dois", 232, heading=True),
        block("Ver o n.º 2 do artigo 1.º e o artigo anterior.", 250),
    )
    units = units_by_id(result)
    refs = units["pt:cirs:code:article:1:paragraph:2"].references
    assert [(r.raw_text, r.target_ids) for r in refs] == [
        ("número anterior", ["pt:cirs:code:article:1:paragraph:1"]),
        ("artigo seguinte", ["pt:cirs:code:article:2-A"]),
    ]
    first = units["pt:cirs:code:article:1:paragraph:1"].references
    assert all(r.status == "unresolved" for r in first)
    third = units["pt:cirs:code:article:1:paragraph:3"].references
    assert third[0].status == "unresolved"
    assert third[0].reason == "external_instrument"
    assert third[1].target_ids == ["pt:cirs:code:article:2-A"]
    fourth = units["pt:cirs:code:article:1:paragraph:4"].references
    assert fourth[0].target_ids == [
        "pt:cirs:code:article:1:paragraph:1",
        "pt:cirs:code:article:1:paragraph:2",
    ]
    assert fourth[1].status == "unresolved"
    assert "n.os 1 e 2" in [r.raw_text for r in fourth]
    last = units["pt:cirs:code:article:2-A"].references
    assert last[0].target_ids == ["pt:cirs:code:article:1:paragraph:2"]
    assert last[1].target_ids == ["pt:cirs:code:article:1"]


def test_duplicate_addresses_fail_instead_of_overwriting():
    with pytest.raises(ValueError, match="Duplicate legal address"):
        parse(block("Artigo 1.º", heading=True), block("Artigo 1.º", 150, heading=True))


def test_wrapped_line_starting_with_letter_is_not_a_new_subparagraph():
    units = units_by_id(
        parse(
            block("Artigo 2.º", heading=True),
            block("Título", 112, heading=True),
            block("4 - Casos das alíneas a),", 130),
            block("b) e c) do n.º 1:", 141.5),
            block("a) Primeiro;", 165),
            block("b) Segundo.", 185),
        )
    )
    assert units["pt:cirs:code:article:2:paragraph:4"].text_normalized.startswith(
        "4 - Casos das alíneas a), b) e c)"
    )
    assert (
        units["pt:cirs:code:article:2:paragraph:4:subparagraph:b"].text_original
        == "b) Segundo."
    )


def test_identity_is_independent_of_snapshot_and_text_revision():
    first = page(
        block(TITLE, 70, heading=True),
        block("Artigo 1.º", heading=True),
        block("Texto original.", 140),
    )
    second = first.model_copy(
        update={
            "snapshot_id": "b" * 64,
            "blocks": [*first.blocks[:2], block("Outra redação.", 140)],
        }
    )
    left = parse_cirs([first], []).units[-1]
    right = parse_cirs([second], []).units[-1]
    assert left.legal_unit_id == right.legal_unit_id
    assert left.text_original != right.text_original


@pytest.mark.parametrize("marker", ["7(*)(**) -", "7* -", "7 (*)-", ".7(*) -"])
def test_annotated_paragraph_number_keeps_its_address(marker):
    units = units_by_id(
        parse(
            block("Artigo 10.º", heading=True),
            block("Título", 112, heading=True),
            block(marker + " Não se aplica:", 140),
            block("a) Uma condição.", 160),
        )
    )
    assert units["pt:cirs:code:article:10:paragraph:7:subparagraph:a"].paragraph == "7"


def test_wrapped_amendment_note_preserves_its_full_text_and_two_locators():
    units = units_by_id(
        parse(
            block("Artigo 1.º", heading=True),
            block("Título", 112, heading=True),
            block("1 - Texto. (Redação da Lei n.º", 140),
            block("1/2020, de 1 de janeiro)", 151.5),
        )
    )
    note = units["pt:cirs:code:article:1:paragraph:1"].editorial_notes[0]
    assert note.raw_text == "(Redação da Lei n.º\n1/2020, de 1 de janeiro)"
    assert len(note.locators) == 2


def test_footnote_quoted_paragraph_is_not_part_of_article_numbering():
    quoted = block("2 - Texto de outra lei.", 181.5).model_copy(
        update={"font_sizes": [8]}
    )
    units = units_by_id(
        parse(
            block("Artigo 78.º-E", heading=True),
            block("Título", 112, heading=True),
            block("10(*) - Uma regra.", 140),
            block("(*) Norma transitória: Lei n.º 1/2020", 170),
            quoted,
            block("11 - Regra seguinte.", 205),
        )
    )
    assert "pt:cirs:code:article:78-E:paragraph:2" not in units
    assert units["pt:cirs:code:article:78-E:paragraph:10:note:1"].role == "transitional"
    assert (
        units["pt:cirs:code:article:78-E:paragraph:11"].text_original
        == "11 - Regra seguinte."
    )


def test_qualified_and_plural_references_do_not_resolve_to_the_wrong_unit():
    units = units_by_id(
        parse(
            block("Artigo 1.º", heading=True),
            block("Título", 112, heading=True),
            block("1 - Texto:", 140),
            block("a) Uma condição;", 160),
            block("b) Outra condição.", 180),
            block(
                "2 - Ver a alínea a) do n.º 1 do artigo 1.º e os artigos 1.º e 2.º.",
                200,
            ),
            block("Nota 1: Disposição transitória", 220),
            block("Ver o número anterior.", 240),
            block("Artigo 2.º", 265, heading=True),
            block("Dois", 277, heading=True),
            block("Texto.", 300),
        )
    )
    refs = units["pt:cirs:code:article:1:paragraph:2"].references
    assert refs[0].target_ids == ["pt:cirs:code:article:1:paragraph:1:subparagraph:a"]
    assert refs[1].target_ids == ["pt:cirs:code:article:1", "pt:cirs:code:article:2"]
    assert refs[1].raw_text == "artigos 1.º e 2.º"
    assert units["pt:cirs:code:article:1:note:1"].references[0].status == "unresolved"


def test_relative_reference_context_does_not_leak_from_children_to_parents():
    units = units_by_id(
        parse(
            block("Artigo 1.º", heading=True),
            block("Título", 112, heading=True),
            block("1 - Condições:", 140),
            block("a) Uma condição.", 160),
            block("b) Nos termos da alínea anterior e do número seguinte.", 180),
            block("2 - Outro ponto.", 200),
        )
    )
    assert units["pt:cirs:code:article:1"].references == []
    assert units["pt:cirs:code:article:1:paragraph:1"].references == []
    refs = units["pt:cirs:code:article:1:paragraph:1:subparagraph:b"].references
    assert refs[0].target_ids == ["pt:cirs:code:article:1:paragraph:1:subparagraph:a"]
    assert refs[1].target_ids == ["pt:cirs:code:article:1:paragraph:2"]


def test_roman_subitems_belong_to_their_alinea():
    units = units_by_id(
        parse(
            block("Artigo 1.º", heading=True),
            block("Título", 112, heading=True),
            block("1 - Condições:", 140),
            block("a) São os seguintes: (Redação da Lei n.º 1/2020)", 160),
            block("i) Primeiro;", 180),
            block("ii) Segundo.", 200),
            block("b) Outra condição.", 220),
        )
    )
    assert units[
        "pt:cirs:code:article:1:paragraph:1:subparagraph:a:item:i"
    ].parent_id.endswith("subparagraph:a")
    assert units[
        "pt:cirs:code:article:1:paragraph:1:subparagraph:a:item:ii"
    ].parent_id.endswith("subparagraph:a")


def test_subsection_and_article_reference_across_sections():
    units = units_by_id(
        parse(
            block("CAPÍTULO I", 90, heading=True),
            block("SECÇÃO I", 115, heading=True),
            block("SUBSECÇÃO I", 140, heading=True),
            block("Artigo 1.º", 165, heading=True),
            block("Um", 177, heading=True),
            block("Ver o artigo seguinte.", 200),
            block("SECÇÃO II", 225, heading=True),
            block("Artigo 2.º-A", 250, heading=True),
            block("Dois", 262, heading=True),
            block("Ver o artigo anterior.", 285),
        )
    )
    assert units["pt:cirs:code:article:1"].parent_id.endswith("subsection:I")
    assert units["pt:cirs:code:article:1"].references[0].target_ids == [
        "pt:cirs:code:article:2-A"
    ]
    assert units["pt:cirs:code:article:2-A"].section == "II"


def test_explicit_cirs_reference_from_decree_and_abbreviations():
    result = parse_cirs(
        [
            page(
                block("Decreto-Lei n.º 442-A/88, de 30 de Novembro", 70, heading=True),
                block("Artigo 1.º", 100, heading=True),
                block("Aprovação", 112, heading=True),
                block("Ver o art.º 1.º do CIRS.", 140),
                block(TITLE, 180, heading=True),
                block("Artigo 1.º", 210, heading=True),
                block("Título", 222, heading=True),
                block("1 - Ver o nº 1 e o artigo seguinte.", 250),
            )
        ],
        [],
    )
    units = units_by_id(result)
    assert units["pt:decreto-lei:442-a/88:approving_decree:article:1"].references[
        0
    ].target_ids == ["pt:cirs:code:article:1"]
    refs = units["pt:cirs:code:article:1:paragraph:1"].references
    assert refs[0].target_ids == ["pt:cirs:code:article:1:paragraph:1"]
    assert refs[1].status == "unresolved"


def test_transition_article_role_and_multiple_notes():
    units = units_by_id(
        parse(
            block("CAPÍTULO I", 85, heading=True),
            block("Artigo 1.º", heading=True),
            block("Regime transitório", 112, heading=True),
            block("1 - Uma regra.", 140),
            block("Nota 1: Nota editorial.", 165),
            block("Nota 2: Outra nota.", 185),
            block("Artigo 2.º", 210, heading=True),
            block("Título", 222, heading=True),
            block("1 - Texto com referência ao regime transitório.", 250),
        )
    )
    assert units["pt:cirs:code:article:1"].role == "transitional"
    assert units["pt:cirs:code:article:1:paragraph:1"].role == "transitional"
    assert units["pt:cirs:code:article:1:note:1"].role == "editorial_note"
    assert units["pt:cirs:code:article:1:note:2"].role == "editorial_note"
    assert units["pt:cirs:code:article:2:paragraph:1"].role == "norm"


def test_body_letters_table_blocks_and_margins_are_not_headings():
    content = [
        block(TITLE, 70, heading=True),
        block("Artigo 1.º", 100, heading=True),
        block("Título", 112, heading=True),
        block("1 - Condições:", 140),
    ]
    for i, label in enumerate(("a", "c", "d", "i")):
        content.append(block(f"{label}) Uma condição.", 160 + i * 20))
    content.append(
        block("Artigo 999.º", 250, heading=True).model_copy(
            update={"bbox": (85, 250, 150, 260)}
        )
    )
    content.append(
        block("Artigo 998.º", 280, heading=True).model_copy(update={"role": "footer"})
    )
    units = units_by_id(parse_cirs([page(*content)], []))
    for label in ("a", "c", "d", "i"):
        assert (
            units[f"pt:cirs:code:article:1:paragraph:1:subparagraph:{label}"].kind
            == "subparagraph"
        )
    assert "pt:cirs:code:article:999" not in units
    assert "pt:cirs:code:article:998" not in units


def test_local_decree_references_keep_the_instrument():
    result = parse_cirs(
        [
            page(
                block("Decreto-Lei n.º 442-A/88, de 30 de Novembro", 70, heading=True),
                block("Artigo 1.º", 100, heading=True),
                block("Aprovação", 112, heading=True),
                block("Ver o artigo 1.º.", 140),
                block("Artigo 2.º", 165, heading=True),
                block("Entrada em vigor", 177, heading=True),
                block("Ver o artigo anterior.", 200),
            )
        ],
        [],
    )
    articles = [u for u in result.units if u.kind == "article"]
    assert articles[0].references[0].target_ids == [articles[0].legal_unit_id]
    assert articles[1].references[0].target_ids == [articles[0].legal_unit_id]
    assert articles[0].code is None


def test_excluded_blocks_and_mixed_snapshot_boundary():
    p = page(
        block("Índice", 70),
        block("Artigo 1.º ................... 7", 90),
        block(TITLE, 120, heading=True),
        block("Artigo 1.º", 150, heading=True),
    )
    result = parse_cirs([p], [])
    assert result.excluded_block_count == 2
    with pytest.raises(ValueError, match="one snapshot"):
        parse_cirs(
            [p, p.model_copy(update={"snapshot_id": "b" * 64, "page_number": 2})], []
        )


def test_ellipsis_in_a_source_quotation_is_preserved():
    result = parse(
        block("Artigo 1.º", 100, heading=True),
        block("Nota: Norma transitória:", 130),
        block('"[...]', 150),
        block("2 - Regra citada.", 170),
        block('[...]"', 190),
    )
    note = next(u for u in result.units if u.kind == "note")
    assert (
        note.text_original
        == 'Nota: Norma transitória:\n"[...]\n2 - Regra citada.\n[...]"'
    )
    assert result.excluded_block_count == 0


def test_relative_paragraph_does_not_jump_into_previous_article():
    units = units_by_id(
        parse(
            block("Artigo 1.º", heading=True),
            block("Título", 112, heading=True),
            block("1 - Uma regra.", 140),
            block("Artigo 2.º", 165, heading=True),
            block("Título", 177, heading=True),
            block("1 - Ver o número anterior.", 200),
            block("Nota: Ver o n.º 1 do artigo 1.º do Código do IRS.", 225),
        )
    )
    assert (
        units["pt:cirs:code:article:2:paragraph:1"].references[0].status == "unresolved"
    )
    assert units["pt:cirs:code:article:2:note:1"].references[0].target_ids == [
        "pt:cirs:code:article:1:paragraph:1"
    ]


def test_same_row_uses_horizontal_reading_order_and_numbered_item_labels():
    units = parse(
        block("Artigo 1.º", heading=True),
        block("Título", 112, heading=True),
        block("Segunda parte.", 140).model_copy(update={"bbox": (350, 140, 540, 150)}),
        block("Primeira parte.", 140),
        block("1 - Condições:", 155),
        block("a) Condições:", 170),
        block("1) Primeiro;", 190),
        block("2) Segundo.", 210),
    ).units
    article = next(u for u in units if u.kind == "article")
    assert "Primeira parte.\nSegunda parte." in article.text_original
    items = [u for u in units if u.kind == "item"]
    assert [u.label for u in items] == ["1", "2"]
    assert items[0].parent_id == items[1].parent_id


def test_annotated_repeal_and_note_locators_do_not_capture_neighboring_text():
    units = units_by_id(
        parse(
            block("Artigo 10.º", heading=True),
            block("Título", 112, heading=True),
            block("18(*) - (Revogado.)", 140),
            block("19 - Uma regra.", 165),
            block("(Redação da Lei n.º 1/2020)", 185),
            block("Outra linha da regra.", 205),
        )
    )
    assert units["pt:cirs:code:article:10:paragraph:18"].legal_status == "repealed"
    note = units["pt:cirs:code:article:10:paragraph:19"].editorial_notes[0]
    assert len(note.locators) == 1
    assert note.locators[0].bbox == (85, 185, 510, 195)


@pytest.mark.parametrize(
    "qualifier",
    [
        "do Estatuto dos Benefícios Fiscais",
        ", do Código do IRC",
        "do EBF",
        "do Regulamento",
    ],
)
def test_external_statute_is_not_the_current_code(qualifier):
    units = units_by_id(
        parse(
            block("Artigo 1.º", heading=True),
            block("Título", 112, heading=True),
            block(f"Ver o artigo 1.º {qualifier}.", 140),
            block("Nota: Outra lei menciona o artigo 1.º.", 165),
        )
    )
    ref = units["pt:cirs:code:article:1"].references[0]
    assert ref.status == "unresolved"
    assert ref.reason == "external_instrument"
    assert units["pt:cirs:code:article:1:note:1"].references[0].status == "unresolved"


def test_separate_numbered_block_at_close_spacing():
    units = units_by_id(
        parse(
            block("Artigo 1.º", heading=True),
            block("Título", 112, heading=True),
            block("1 - Uma regra.", 140),
            block("2 - Outra regra.", 153),
        )
    )
    assert (
        units["pt:cirs:code:article:1:paragraph:2"].text_original == "2 - Outra regra."
    )


def test_left_bold_article_mention_at_heading_margin_stays_in_the_body():
    units = units_by_id(
        parse(
            block("Artigo 1.º", heading=True),
            block("Título", 112, heading=True),
            block("Um texto.", 140),
            block("Artigo 999.º", 165, heading=True).model_copy(
                update={"bbox": (120, 165, 296.5, 175)}
            ),
        )
    )
    assert "pt:cirs:code:article:999" not in units
    assert "Artigo 999.º" in units["pt:cirs:code:article:1"].text_original


@pytest.mark.parametrize("pages", [[], [page(block("Artigo 1.º", heading=True))]])
def test_missing_instrument_boundary_is_rejected(pages):
    with pytest.raises(ValueError):
        parse_cirs(pages, [])


@pytest.mark.parametrize(
    "reference",
    [
        "n.º 1 do artigo anterior",
        "n.º 1 do artigo seguinte",
        "alínea a) do número anterior",
        "alínea anterior do n.º 1",
        "alínea a) do n.º 1 do artigo anterior",
    ],
)
def test_qualified_relative_reference_never_discards_its_parent(reference):
    units = units_by_id(
        parse(
            block("Artigo 1.º", 90, heading=True),
            block("1 - Uma regra.", 115),
            block("Artigo 2.º", 140, heading=True),
            block("1 - Primeira regra.", 165),
            block("2 - Condições:", 190),
            block("a) Primeira condição.", 215),
            block(f"b) Ver a {reference}.", 240),
            block("3 - Outra regra.", 265),
            block("Artigo 3.º", 290, heading=True),
            block("1 - Última regra.", 315),
        )
    )
    [ref] = units["pt:cirs:code:article:2:paragraph:2:subparagraph:b"].references
    assert ref.raw_text == reference
    assert ref.status == "unresolved"
    assert ref.target_ids == []
    assert ref.reason == "missing_or_ambiguous_target"


def test_explicit_cirs_relative_reference_needs_an_anchor_in_the_code():
    units = units_by_id(
        parse_cirs(
            [
                page(
                    block(
                        "Decreto-Lei n.º 442-A/88, de 30 de Novembro", 70, heading=True
                    ),
                    block("Artigo 1.º", 100, heading=True),
                    block("Uma regra.", 125),
                    block("Artigo 2.º", 150, heading=True),
                    block("Ver o artigo anterior do CIRS.", 175),
                    block(TITLE, 200, heading=True),
                    block("Artigo 1.º", 225, heading=True),
                    block("Uma regra.", 250),
                    block("Artigo 2.º", 275, heading=True),
                    block("Ver o artigo anterior do CIRS.", 300),
                )
            ],
            [],
        )
    )
    [decree_ref] = units[
        "pt:decreto-lei:442-a/88:approving_decree:article:2"
    ].references
    assert decree_ref.status == "unresolved"
    assert decree_ref.target_ids == []
    [code_ref] = units["pt:cirs:code:article:2"].references
    assert code_ref.target_ids == ["pt:cirs:code:article:1"]
