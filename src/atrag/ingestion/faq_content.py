"""Ordered text, list and table adapters for the saved FAQ DOM."""

from urllib.parse import urljoin, urlsplit

from bs4 import Comment, NavigableString, Tag

from atrag.schemas.ingestion.extraction import (
    ExtractedTable,
    HTMLTableCell,
    HTMLTablePart,
)
from atrag.schemas.ingestion.faq import FAQLink
from atrag.schemas.ingestion.faq_extraction import FAQListItem
from atrag.schemas.materials import HTMLLocation

BLOCKS = {
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "table",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "blockquote",
}
IGNORED = {"script", "style", "noscript"}


def plain_text(node: Tag) -> str:
    return " ".join(node.get_text(" ", strip=True).split())


def _text(node) -> str:
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in IGNORED:
        return ""
    if node.name == "br":
        return "\n"
    text = " ".join(_text(child) for child in node.children)
    return f"\n{text}\n" if node.name in BLOCKS else text


def answer_text(body: Tag) -> str:
    # Collapse source indentation within blocks, retain the observed boundaries.
    return "\n".join(
        line for raw in _text(body).splitlines() if (line := " ".join(raw.split()))
    )


def answer_links(body: Tag, page_url: str) -> tuple[FAQLink, ...]:
    links = []
    for link in body.select("a[href]"):
        href = str(link["href"])
        url = urljoin(page_url, href)
        if urlsplit(url).scheme not in ("https", "http"):
            raise ValueError(f"Unsupported answer link: {href}")
        links.append(FAQLink(text=plain_text(link), href=href, url=url))
    return tuple(links)


def list_items(body: Tag) -> tuple[FAQListItem, ...]:
    items = []
    for item in body.select("li"):
        parents = item.find_parents(("ol", "ul"))
        if not parents:
            raise ValueError("List item has no list")
        items.append(
            FAQListItem(
                text=plain_text(item),
                ordered=parents[0].name == "ol",
                depth=len(parents),
            )
        )
    return tuple(items)


def _table_rows(table: Tag) -> tuple[list, list, list]:
    if table.find("table") is not None:
        raise ValueError("Nested table requires review")
    cells, headers, rows = [], [], []
    for row in table.select("tr"):
        tags = row.find_all(("th", "td"), recursive=False)
        if not tags:
            raise ValueError("Table row has no cells")
        parsed = [
            HTMLTableCell(
                text=plain_text(c),
                rowspan=int(str(c.get("rowspan", "1"))),
                colspan=int(str(c.get("colspan", "1"))),
            )
            for c in tags
        ]
        cells.append(parsed)
        values = [c.text for c in parsed]
        (headers if all(c.name == "th" for c in tags) else rows).append(values)
    if not cells:
        raise ValueError("Table has no rows")
    return cells, headers, rows


def answer_tables(
    body: Tag, entry_id: str, location: HTMLLocation, question: str
) -> tuple[ExtractedTable[HTMLTablePart], ...]:
    tables = []
    for ordinal, table in enumerate(body.select("table"), 1):
        cells, headers, rows = _table_rows(table)
        table_id = f"{entry_id}:table:{ordinal}"
        part = HTMLTablePart(
            part_id=table_id,
            table_id=table_id,
            location=location,
            cells=cells,
            rows=[*[list(r) for r in headers], *rows],
        )
        tables.append(
            ExtractedTable[HTMLTablePart](
                table_id=table_id,
                parts=[part],
                header_rows=headers,
                rows=rows,
                context=question,
                review_status="verified",
                review_note="DOM cells and spans preserved; no inferred PDF coordinates",
            )
        )
    return tuple(tables)
