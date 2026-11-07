"""Map the parser's table rendering to indivisible, checked source rows."""

import json
import re
from dataclasses import dataclass

from atrag.ingestion.extraction import _headers, _part_text, normalize_text
from atrag.ingestion.text_parts import pack_ranges, text_ranges
from atrag.schemas.ingestion.extraction import HTMLTablePart
from atrag.schemas.ingestion.fragments import TableRange


@dataclass(frozen=True)
class RowSpan:
    table_id: str
    row_index: int
    start: int
    end: int


class TableRowTooLarge(ValueError):
    def __init__(self, table_id, token_count):
        self.table_id = table_id
        self.token_count = token_count


def _row_text(row):
    return " | ".join(normalize_text(cell or "") for cell in row)


def _part_rows(text, part, table, start, row_index):
    rendered = _part_text(part, table)
    left = text.find(rendered, start)
    if left < 0:
        raise ValueError("Table part is absent from its legal unit")
    data = part.rows[len(_headers(part.rows)) :]
    if list(map(_row_text, data)) != list(
        map(_row_text, table.rows[row_index : row_index + len(data)])
    ):
        raise ValueError("Checked table rows differ from their source parts")
    cursor = left + len(rendered) - len("\n".join(map(_row_text, data)))
    spans = []
    for row in data:
        end = cursor + len(_row_text(row))
        if text[end : end + 1] == "\n":
            end += 1
        spans.append(RowSpan(table.table_id, row_index, left, end))
        row_index += 1
        cursor = left = end
    return spans, left, row_index


def table_rows(unit, tables):
    spans = []
    for identity in unit.table_ids:
        table = tables[identity]
        start = row_index = 0
        for part in table.parts:
            locate = _html_rows if isinstance(part, HTMLTablePart) else _part_rows
            rows, start, row_index = locate(
                unit.text_original, part, table, start, row_index
            )
            spans.extend(rows)
        if row_index != len(table.rows):
            raise ValueError("Table rows do not match their source parts")
    return sorted(spans, key=lambda row: row.start)


def _html_rows(text, part, table, start, row_index):
    # Match DOM cell text across original paragraph/newline boundaries. Keep
    # the source answer unchanged, including layout tables and cell spans.
    patterns = [
        r"\s+".join(re.escape(token) for token in " ".join(c.text for c in row).split())
        for row in part.cells
    ]
    if not any(patterns):
        return [], start, row_index + len(table.rows)
    matches = list(
        re.finditer(r"\s*".join(f"({pattern})" for pattern in patterns), text[start:])
    )
    if not matches:
        raise ValueError("HTML table cells are absent from the source answer")
    if len(matches) != 1:
        raise ValueError("Cannot locate ambiguous HTML table text without DOM offsets")
    match = matches[0]
    headers = len(table.header_rows)
    spans = []
    for index in range(headers, len(patterns)):
        left = start + (match.start() if index == headers else match.start(index + 1))
        right = start + (
            match.start(index + 2) if index + 1 < len(patterns) else match.end()
        )
        spans.append(RowSpan(table.table_id, row_index, left, right))
        row_index += 1
    return spans, start + match.end(), row_index


def select_tables(rows, tables, start, end):
    selected = {}
    for row in rows:
        if row.start < end and start < row.end:
            selected.setdefault(row.table_id, []).append(row.row_index)
    slices, ranges = [], []
    for identity, indices in selected.items():
        left, right = min(indices), max(indices) + 1
        table = tables[identity]
        # Search payloads carry only selected numeric rows. Geometry remains
        # conservative; the complete table stays in the document table store.
        slices.append(
            table.model_copy(
                update={
                    "rows": table.rows[left:right],
                    "parts": [
                        part.model_copy(update={"rows": []}) for part in table.parts
                    ],
                    "context": "",
                }
            )
        )
        ranges.append(TableRange(table_id=identity, row_start=left, row_end=right))
    return slices, ranges


def validate_table_boundaries(rows, start, end):
    for row in rows:
        if row.start < start < row.end or row.start < end < row.end:
            raise ValueError("Part boundary splits a table row")


def table_header_text(tables):
    return "\n".join(_row_text(row) for table in tables for row in table.header_rows)


def table_text(tables):
    return json.dumps(
        [{"headers": t.header_rows, "rows": t.rows} for t in tables],
        ensure_ascii=False,
    )


def bounded_table_ranges(text, rows, measure, target, maximum):
    atoms, cursor = [], 0
    for row in rows:
        if cursor != row.start:
            atoms.extend(_gap_ranges(text, cursor, row.start, measure, target, maximum))
        size = measure(row.start, row.end)
        if size > maximum:
            raise TableRowTooLarge(row.table_id, size)
        atoms.append((row.start, row.end))
        cursor = row.end
    if cursor != len(text):
        atoms.extend(_gap_ranges(text, cursor, len(text), measure, target, maximum))
    return list(pack_ranges(iter(atoms), lambda a, b: measure(a, b) <= target))


def _gap_ranges(text, start, end, measure, target, maximum):
    ranges = text_ranges(
        text[start:end], lambda a, b: measure(start + a, start + b), target, maximum
    )
    return [(start + a, start + b) for a, b in ranges]
