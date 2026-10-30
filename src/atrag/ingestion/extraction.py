"""Pure transformations: conservative normalization, margins and table continuity."""

import math
import re
from collections import Counter

from atrag.schemas.ingestion.extraction import (
    ExtractedPage,
    ExtractedTable,
    TablePart,
    TextBlock,
)


def normalize_text(text: str) -> str:
    # Only discretionary hyphens are safe to join without a language dictionary.
    text = re.sub(r"\u00ad\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text.replace("\u00ad", "")).strip()


def _margin_role(block: TextBlock, height: float) -> str:
    if block.bbox[1] < height * 0.085:
        return "header"
    if block.bbox[1] > height * 0.86:
        return "footer"
    return "body"


def _block_role(block: TextBlock, height: float, repeated: set) -> str:
    if block.bbox[1] > height * 0.9 and block.text.strip().isdigit():
        return "page_number"
    role = _margin_role(block, height)
    if (role, block.text) in repeated:
        return role
    return "body"


def remove_margins(pages: list[ExtractedPage]) -> list[ExtractedPage]:
    counts = Counter(
        key
        for page in pages
        for key in {(_margin_role(b, page.height), b.text) for b in page.blocks}
    )
    repeated = {
        key for key, n in counts.items() if n >= max(2, math.ceil(len(pages) / 2))
    }
    result = []
    for page in pages:
        blocks = [
            b.model_copy(update={"role": _block_role(b, page.height, repeated)})
            for b in page.blocks
        ]
        result.append(
            page.model_copy(
                update={
                    "blocks": blocks,
                    "printed_page_labels": [
                        b.text for b in blocks if b.role == "page_number"
                    ],
                }
            )
        )
    return result


def _headers(rows: list[list[str | None]]) -> list[list[str | None]]:
    headers = []
    for row in rows:
        if re.search(r"\d", " ".join(cell or "" for cell in row)):
            break
        headers.append(row)
    return headers


def table_context(page: ExtractedPage, part: TablePart) -> str:
    preceding = [
        b.text for b in page.blocks if b.role == "body" and b.bbox[3] <= part.bbox[1]
    ]
    # A source locator, not a parsed/resolved legal address (Step 4).
    start = 0
    for i, text in enumerate(preceding):
        if re.fullmatch(r"Artigo\s+\d+\.º(?:-[A-Z])?", text.strip()):
            start = i
    return "\n".join(preceding[start:])


def _compatible_columns(table: ExtractedTable, current: TablePart) -> bool:
    previous = table.parts[-1]
    if len(previous.rows[0]) != len(current.rows[0]):
        return False
    if any(abs(previous.bbox[i] - current.bbox[i]) > 2 for i in (0, 2)):
        return False
    headers = _headers(current.rows)
    return not headers or headers == table.header_rows


def _intervening_text(before, after, previous, current) -> bool:
    trailing = [
        b for b in before.blocks if b.role == "body" and b.bbox[1] >= previous.bbox[3]
    ]
    leading = [
        b for b in after.blocks if b.role == "body" and b.bbox[3] <= current.bbox[1]
    ]
    return bool(trailing or leading)


def _continues(
    table: ExtractedTable, current: TablePart, pages: list[ExtractedPage]
) -> bool:
    previous = table.parts[-1]
    if current.page_number != previous.page_number + 1:
        return False
    before, after = pages[previous.page_number - 1], pages[current.page_number - 1]
    if previous.bbox[3] < before.height * 0.75 or current.bbox[1] > after.height * 0.2:
        return False
    if not _compatible_columns(table, current):
        return False
    return not _intervening_text(before, after, previous, current)


def join_tables(
    pages: list[ExtractedPage],
) -> tuple[list[ExtractedPage], list[ExtractedTable]]:
    tables: list[ExtractedTable] = []
    updated = []
    for page in pages:
        parts = []
        for part in page.tables:
            if tables and _continues(tables[-1], part, pages):
                previous = tables[-1]
                part = part.model_copy(update={"table_id": previous.table_id})
                tables[-1] = previous.model_copy(
                    update={
                        "parts": [*previous.parts, part],
                        "rows": [
                            *previous.rows,
                            *part.rows[len(_headers(part.rows)) :],
                        ],
                    }
                )
            else:
                headers = _headers(part.rows)
                tables.append(
                    ExtractedTable(
                        table_id=part.table_id,
                        parts=[part],
                        header_rows=headers,
                        rows=part.rows[len(headers) :],
                        context=table_context(page, part),
                    )
                )
            parts.append(part)
        updated.append(page.model_copy(update={"tables": parts}))
    return updated, tables


def _in_table(block: TextBlock, parts: list[TablePart]) -> bool:
    return any(
        part.bbox[1] <= block.bbox[1]
        and block.bbox[3] <= part.bbox[3]
        and part.bbox[0] <= block.bbox[0]
        and block.bbox[2] <= part.bbox[2]
        for part in parts
    )


def _part_text(part: TablePart, table: ExtractedTable) -> str:
    rows = part.rows
    if not _headers(rows):
        rows = [*table.header_rows, *rows]
    text = "\n".join(" | ".join(normalize_text(c or "") for c in row) for row in rows)
    return f"[Table {part.table_id}]\n{text}"


def _join_segments(segments: list[tuple[float, str, bool]]) -> str:
    result, lines = [], []
    for _, text, is_table in sorted(segments):
        if is_table:
            result.extend([normalize_text("\n".join(lines)), text])
            lines = []
        else:
            lines.append(text)
    result.append(normalize_text("\n".join(lines)))
    return "\n".join(text for text in result if text)


def page_text(page: ExtractedPage, tables: dict[str, ExtractedTable]) -> str:
    segments = [
        (b.bbox[1], b.text, False)
        for b in page.blocks
        if b.role == "body" and not _in_table(b, page.tables)
    ]
    segments.extend(
        (p.bbox[1], _part_text(p, tables[p.table_id]), True) for p in page.tables
    )
    return _join_segments(segments)
