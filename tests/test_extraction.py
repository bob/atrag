import pytest

from atrag.ingestion.extraction import join_tables, page_text, remove_margins
from atrag.schemas.ingestion.extraction import ExtractedPage, TablePart, TextBlock


def block(text, top, bottom=None, x0=85, x1=550):
    return TextBlock(
        text=text, bbox=(x0, top, x1, top + 10 if bottom is None else bottom)
    )


@pytest.fixture
def pages():
    def part(n, top, bottom, rows):
        return TablePart(
            part_id=f"p{n}",
            table_id=f"p{n}",
            page_number=n,
            bbox=(100, top, 500, bottom),
            rows=rows,
        )

    return [
        ExtractedPage(
            snapshot_id="a" * 64,
            page_number=1,
            width=595,
            height=842,
            raw_text="",
            blocks=[],
            images=[],
            tables=[
                part(1, 640, 720, [["Income (euros)", "Rate (%)"], ["até 10", "2,5"]])
            ],
        ),
        ExtractedPage(
            snapshot_id="a" * 64,
            page_number=2,
            width=595,
            height=842,
            raw_text="",
            blocks=[],
            images=[],
            tables=[part(2, 80, 120, [["superior a 10", "5"]])],
        ),
    ]


@pytest.mark.parametrize(
    "change",
    [
        "same_page",
        "nonadjacent",
        "bottom",
        "top",
        "columns",
        "alignment",
        "trailing",
        "leading",
        "different_headers",
    ],
)
def test_distinct_tables_are_not_joined(pages, change):
    first, second = pages
    p1, p2 = first.tables[0], second.tables[0]
    if change == "same_page":
        pages = [
            first.model_copy(
                update={"tables": [p1, p2.model_copy(update={"page_number": 1})]}
            )
        ]
    elif change == "nonadjacent":
        pages[1] = second.model_copy(
            update={"tables": [p2.model_copy(update={"page_number": 3})]}
        )
    elif change == "bottom":
        pages[0] = first.model_copy(
            update={"tables": [p1.model_copy(update={"bbox": (100, 300, 500, 400)})]}
        )
    elif change == "top":
        pages[1] = second.model_copy(
            update={"tables": [p2.model_copy(update={"bbox": (100, 300, 500, 400)})]}
        )
    elif change == "columns":
        pages[1] = second.model_copy(
            update={"tables": [p2.model_copy(update={"rows": [["10", "5", "extra"]]})]}
        )
    elif change == "alignment":
        pages[1] = second.model_copy(
            update={"tables": [p2.model_copy(update={"bbox": (120, 80, 520, 120)})]}
        )
    elif change == "trailing":
        pages[0] = first.model_copy(update={"blocks": [block("New provision", 725)]})
    elif change == "leading":
        pages[1] = second.model_copy(update={"blocks": [block("New provision", 60)]})
    else:
        pages[1] = second.model_copy(
            update={
                "tables": [
                    p2.model_copy(update={"rows": [["Another", "Table"], *p2.rows]})
                ]
            }
        )
    _, tables = join_tables(pages)
    assert len(tables) == 2


def test_repeated_table_headers_do_not_become_data(pages):
    second = pages[1]
    part = second.tables[0]
    pages[1] = second.model_copy(
        update={
            "tables": [
                part.model_copy(
                    update={"rows": [["Income (euros)", "Rate (%)"], *part.rows]}
                )
            ]
        }
    )
    updated, [table] = join_tables(pages)
    assert table.rows == [["até 10", "2,5"], ["superior a 10", "5"]]
    text = page_text(updated[1], {table.table_id: table})
    assert text.count("Rate (%)") == 1
    assert "superior a 10 | 5" in text


def test_only_repeated_margins_and_printed_numbers_are_removed(pages):
    for n, page in enumerate(pages):
        pages[n] = page.model_copy(
            update={
                "tables": [],
                "blocks": [
                    block("Common header", 50),
                    block(f"Unique heading {n}", 65),
                    block("Common header", 100),
                    block("não", 200),
                    block("se aplica ao n.º 1, alínea a), 68-A e 2,5 %.", 211),
                    block("Common footer", 750),
                    block(str(10 + n), 785),
                ],
            }
        )
    updated = remove_margins(pages)
    for n, page in enumerate(updated):
        assert [b.role for b in page.blocks] == [
            "header",
            "body",
            "body",
            "body",
            "body",
            "footer",
            "page_number",
        ]
        assert page.printed_page_labels == [str(10 + n)]
        text = page_text(page, {})
        assert f"Unique heading {n}" in text
        assert text.count("Common header") == 1
        assert "Common footer" not in text
        assert "não se aplica ao n.º 1, alínea a), 68-A e 2,5 %." in text


def test_table_text_order_and_edge_notes_are_preserved(pages):
    page = pages[0]
    page = page.model_copy(
        update={
            "blocks": [
                block("Before", 200),
                block("After", 750),
                block("duplicate top", 640, 650, x0=100, x1=500),
                block("duplicate bottom", 710, 720, x0=100, x1=500),
                block("left note", 650, 660, x0=85, x1=99),
                block("right note", 660, 670, x0=501, x1=550),
                block("overlapping top note", 630, 650, x0=100, x1=500),
                block("overlapping bottom note", 710, 730, x0=100, x1=500),
            ]
        }
    )
    updated, [table] = join_tables([page])
    text = page_text(updated[0], {table.table_id: table})
    assert "duplicate" not in text
    assert text == (
        "Before overlapping top note\n"
        "[Table p1]\nIncome (euros) | Rate (%)\naté 10 | 2,5\n"
        "left note right note overlapping bottom note After"
    )


def test_only_the_latest_adjacent_table_is_extended(pages):
    first, second = pages
    independent = first.tables[0].model_copy(
        update={
            "part_id": "earlier",
            "table_id": "earlier",
            "bbox": (100, 200, 500, 300),
        }
    )
    pages[0] = first.model_copy(update={"tables": [independent, *first.tables]})
    p2 = second.tables[0].model_copy(update={"bbox": (102, 80, 502, 720)})
    pages[1] = second.model_copy(update={"tables": [p2]})
    p3 = p2.model_copy(
        update={
            "part_id": "p3",
            "table_id": "p3",
            "page_number": 3,
            "bbox": (104, 80, 504, 120),
            "rows": [["above 20", "10"]],
        }
    )
    pages.append(second.model_copy(update={"page_number": 3, "tables": [p3]}))
    updated, tables = join_tables(pages)
    assert len(tables) == 2
    assert tables[0].table_id == "earlier"
    assert [part.part_id for part in tables[1].parts] == ["p1", "p2", "p3"]
    assert [row[-1] for row in tables[1].rows] == ["2,5", "5", "10"]
    assert updated[-1].tables[0].table_id == "p1"


@pytest.mark.parametrize("side", ["leading", "trailing"])
def test_provision_touching_table_border_prevents_join(pages, side):
    n = 0 if side == "trailing" else 1
    page = pages[n]
    text = (
        block("separate provision", 720)
        if n == 0
        else block("separate provision", 70, 80)
    )
    pages[n] = page.model_copy(update={"blocks": [text]})
    _, tables = join_tables(pages)
    assert len(tables) == 2


def test_table_context_keeps_unheaded_introduction_at_border(pages):
    page = pages[0]
    page = page.model_copy(
        update={
            "blocks": [
                block("Header", 50).model_copy(update={"role": "header"}),
                block("Important condition", 200),
                block("1 - tabela seguinte", 630, 640),
                block("After table", 750),
            ]
        }
    )
    _, [table] = join_tables([page])
    assert table.context == "Important condition\n1 - tabela seguinte"


def test_margins_do_not_include_boundary_or_body_numbers(pages):
    for n, page in enumerate(pages):
        pages[n] = page.model_copy(
            update={
                "height": 1000,
                "blocks": [
                    block("At header boundary", 85),
                    block("At footer boundary", 860),
                    block("900", 900),
                    block("2026", 200),
                    block("Final paragraph", 950),
                ],
            }
        )
    for page in remove_margins(pages):
        assert [b.role for b in page.blocks] == [
            "body",
            "body",
            "footer",
            "body",
            "footer",
        ]
        assert page.printed_page_labels == []


def test_image_and_table_regions_fill_gaps_without_hiding_warnings(pages):
    from atrag.ingestion.services.extract import _page_issues
    from atrag.schemas.ingestion.extraction import ImageRegion

    page = pages[0].model_copy(
        update={
            "height": 1000,
            "raw_text": "a" * 100,
            "tables": [],
            "blocks": [block("body", 240)],
        }
    )
    assert _page_issues(page, {}) == []
    assert _page_issues(
        page.model_copy(update={"blocks": [block("body", 241)]}), {}
    ) == ["large_blank_region"]
    assert _page_issues(page.model_copy(update={"raw_text": "a" * 99}), {}) == [
        "short_text"
    ]
    page = page.model_copy(
        update={"blocks": [block("before", 100, 110), block("after", 400, 410)]}
    )
    assert _page_issues(page, {}) == ["large_blank_region"]
    image = ImageRegion(
        image_id="img",
        source_sha256="a" * 64,
        bbox=(350, 110, 500, 400),
        source_size=(1, 1),
        role="content",
        ocr_status="not_requested",
        review_status="verified",
    )
    assert _page_issues(page.model_copy(update={"images": [image]}), {}) == []
    unreviewed = image.model_copy(update={"review_status": "needs_review"})
    assert _page_issues(page.model_copy(update={"images": [unreviewed]}), {}) == [
        "image_needs_review:img:not_requested"
    ]
    part = pages[0].tables[0].model_copy(update={"bbox": image.bbox})
    page = page.model_copy(update={"tables": [part]})
    page, [table] = join_tables([page])
    assert _page_issues(page[0], {table.table_id: table}) == ["table_needs_review"]


def test_continuation_at_page_region_boundary_and_its_own_text(pages):
    first, second = pages
    p1 = first.tables[0].model_copy(update={"bbox": (100, 640, 500, 750)})
    p2 = second.tables[0].model_copy(update={"bbox": (100, 200, 500, 240)})
    pages = [
        first.model_copy(update={"height": 1000, "tables": [p1]}),
        second.model_copy(
            update={"height": 1000, "tables": [p2], "blocks": [block("5", 205, 215)]}
        ),
    ]
    _, tables = join_tables(pages)
    assert len(tables) == 1
    assert len(tables[0].rows) == 2


def test_continuation_text_inside_the_table_is_not_a_new_provision(pages):
    pages[1] = pages[1].model_copy(update={"blocks": [block("5", 85, 95)]})
    _, tables = join_tables(pages)
    assert len(tables) == 1
