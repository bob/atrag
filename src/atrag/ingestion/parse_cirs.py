"""Pure, layout-aware parser for the AT CIRS compilation family."""

import re
from dataclasses import dataclass, field

from atrag.ingestion.extraction import _in_table, _part_text, normalize_text
from atrag.ingestion.legal_references import article_number, resolve_references
from atrag.schemas.ingestion.extraction import ExtractedPage, ExtractedTable, TextBlock
from atrag.schemas.ingestion.structure import (
    EditorialNote,
    LegalUnit,
    SourceLocator,
    StructureResult,
)

PARSER_VERSION = "3"
CODE_TITLE = "Código do Imposto sobre o Rendimento das Pessoas Singulares"
ARTICLE_HEADING = re.compile(r"Artigo\s+(\d+\.º(?:-[A-Z])?|único)", re.IGNORECASE)
DIVISION = re.compile(r"(CAPÍTULO|SECÇÃO|SUBSECÇÃO)\s+([IVXLCDM]+)", re.IGNORECASE)
PARAGRAPH = re.compile(r"^\.?(\d+(?:-[A-Z])?)\s*(?:\(\*+\)|\*+)*\s*[-–—]\s*")
SUBPARAGRAPH = re.compile(r"^([a-z])\)\s+", re.IGNORECASE)
ITEM = re.compile(r"^(\d+|[ivxlcdm]+)\)\s+", re.IGNORECASE)
EDITORIAL = re.compile(
    r"\((Reda[çc][ãa]o|Aditad[oa]|Revogad[oa]|Revoga[çc][ãa]o)\b[^)]*\)", re.IGNORECASE
)
LEVELS = (
    "document",
    "chapter",
    "section",
    "subsection",
    "article",
    "paragraph",
    "subparagraph",
    "item",
    "note",
)


@dataclass
class _Segment:
    page: ExtractedPage
    block: TextBlock
    table_id: str | None = None

    def locator(self):
        return SourceLocator(
            page_number=self.page.page_number,
            printed_page_labels=self.page.printed_page_labels,
            bbox=self.block.bbox,
        )


@dataclass
class _Draft:
    address: dict
    segments: list[_Segment] = field(default_factory=list)
    title: list[str] = field(default_factory=list)
    own_segments: list[_Segment] = field(default_factory=list)
    status: str = "unknown"
    resume: bool = False

    def finish(self) -> LegalUnit:
        locators = [s.locator() for s in self.segments]
        text = "\n".join(s.block.text for s in self.segments)
        return LegalUnit(
            **self.address,
            unit_title=" ".join(self.title),
            legal_status=self.status,
            snapshot_id=self.segments[0].page.snapshot_id,
            page_start=locators[0].page_number,
            page_end=locators[-1].page_number,
            printed_page_labels=list(
                dict.fromkeys(
                    label for l in locators for label in l.printed_page_labels
                )
            ),
            locators=locators,
            text_original=text,
            text_normalized=normalize_text(text),
            editorial_notes=_editorial_notes(self.own_segments),
            table_ids=list(
                dict.fromkeys(s.table_id for s in self.segments if s.table_id)
            ),
            issues=sorted({issue for s in self.segments for issue in s.page.issues}),
        )


def _citation(address: dict) -> str:
    labels = [
        "CIRS" if address["instrument_id"] == "pt:cirs" else "Decreto-Lei n.º 442-A/88"
    ]
    for name, prefix in (
        ("chapter", "capítulo"),
        ("section", "secção"),
        ("article", "artigo"),
        ("paragraph", "n.º"),
        ("subparagraph", "alínea"),
    ):
        value = address.get(name)
        if value is None:
            continue
        if name in ("chapter", "section") and address.get("article"):
            continue
        if name == "article":
            number, separator, suffix = value.partition("-")
            value = "único" if value == "unico" else f"{number}.º{separator}{suffix}"
        labels.append(f"{prefix} {value}")
    if address["kind"] in ("item", "note"):
        labels.append(f"{address['kind']} {address['label']}")
    return ", ".join(labels)


def _segments(pages: list[ExtractedPage], tables: list[ExtractedTable]):
    by_id = {t.table_id: t for t in tables}
    for page in pages:
        segments = [
            _Segment(page, b)
            for b in page.blocks
            if b.role == "body" and not _in_table(b, page.tables)
        ]
        segments.extend(
            _Segment(
                page,
                TextBlock(bbox=p.bbox, text=_part_text(p, by_id[p.table_id])),
                p.table_id,
            )
            for p in page.tables
        )
        yield from sorted(segments, key=lambda s: (s.block.bbox[1], s.block.bbox[0]))


def _heading(segment: _Segment) -> bool:
    block, page = segment.block, segment.page
    centered = (
        abs((block.bbox[0] + block.bbox[2]) / 2 - page.width / 2) < page.width * 0.15
    )
    return centered and any("bold" in f.lower() for f in block.font_names)


def _editorial_notes(segments: list[_Segment]) -> list[EditorialNote]:
    text = "\n".join(s.block.text for s in segments)
    line_locators = [s.locator() for s in segments for _ in s.block.text.split("\n")]
    notes = []
    for match in EDITORIAL.finditer(text):
        word = match[1].lower()
        kind = (
            "repeal"
            if word.startswith("revog")
            else "addition"
            if word.startswith("aditad")
            else "amendment"
        )
        notes.append(
            EditorialNote(
                kind=kind,
                raw_text=match[0],
                locators=line_locators[
                    text[: match.start()].count("\n") : text[: match.end()].count("\n")
                    + 1
                ],
            )
        )
    return notes


def _document_identity(text):
    if re.fullmatch(r"Decreto-Lei n\.[ºo] 442-A/88, de .+", text):
        return "pt:decreto-lei:442-a/88", "approving_decree"
    return {CODE_TITLE: ("pt:cirs", "code"), "Preâmbulo": ("pt:cirs", "preamble")}.get(
        text
    )


class _Parser:
    def __init__(self):
        self.drafts: list[_Draft] = []
        self.stack: list[_Draft] = []
        self.ids: set[str] = set()
        self.title_owner: _Draft | None = None
        self.excluded = 0
        self.previous: _Segment | None = None

    def _start(self, kind: str, label: str, role="norm", **address):
        parent = self.stack[-1].address if self.stack else {}
        inherited = {
            k: parent.get(k)
            for k in (
                "instrument_id",
                "code",
                "document_scope",
                "chapter",
                "section",
                "article",
                "paragraph",
                "subparagraph",
            )
        }
        inherited.update(address)
        prefix = parent.get(
            "legal_unit_id",
            f"{inherited['instrument_id']}:{inherited['document_scope']}",
        )
        # Article addresses do not depend on chapter/section placement.
        if kind == "article":
            prefix = f"{inherited['instrument_id']}:{inherited['document_scope']}"
        key = prefix if kind == "document" else f"{prefix}:{kind}:{label}"
        if key in self.ids:
            raise ValueError(f"Duplicate legal address: {key}")
        self.ids.add(key)
        address = dict(
            **inherited,
            legal_unit_id=key,
            parent_id=parent.get("legal_unit_id"),
            kind=kind,
            role="transitional"
            if role == "norm" and parent.get("role") == "transitional"
            else role,
            label=label,
        )
        address["citation_label"] = _citation(address)
        draft = _Draft(address)
        self.stack.append(draft)
        self.drafts.append(draft)
        return draft

    def _trim(self, kinds):
        while self.stack and self.stack[-1].address["kind"] in kinds:
            self.stack.pop()

    def _document(self, instrument, scope):
        self.stack = []
        self._start(
            "document",
            scope,
            role=scope,
            instrument_id=instrument,
            document_scope=scope,
            code="CIRS" if scope == "code" else None,
        )
        self.title_owner = None

    def _structural_heading(self, text: str):
        division = DIVISION.fullmatch(text)
        article = ARTICLE_HEADING.fullmatch(text)
        if division:
            kind = {
                "CAPÍTULO": "chapter",
                "SECÇÃO": "section",
                "SUBSECÇÃO": "subsection",
            }[division[1].upper()]
            self._trim(LEVELS[LEVELS.index(kind) :])
            self.title_owner = self._start(
                kind, division[2], **{kind: division[2]} if kind != "subsection" else {}
            )
        elif article:
            self._trim({"article", "paragraph", "subparagraph", "item", "note"})
            number = (
                "unico" if article[1].lower() == "único" else article_number(article[1])
            )
            self.title_owner = self._start("article", number, article=number)
        else:
            return False
        return True

    def _note(self, role, *, paragraph=False):
        self._trim({"subparagraph", "item", "note"})
        if not paragraph:
            self._trim({"paragraph"})
        parent = self.stack[-1].address["legal_unit_id"]
        number = 1 + sum(
            d.address["kind"] == "note" and d.address["parent_id"] == parent
            for d in self.drafts
        )
        self._start("note", str(number), role=role).resume = paragraph

    def _body_boundary(self, segment: _Segment):
        text = segment.block.text.strip()
        self._resume_note(segment)
        self._note_boundary(text)
        if (
            self.stack[-1].address["kind"] == "note"
            or self.stack[0].address["document_scope"] == "preamble"
        ):
            return
        if not self._wrapped(segment):
            self._numbered_boundary(text)

    def _resume_note(self, segment):
        leaf = self.stack[-1]
        if (
            leaf.resume
            and PARAGRAPH.match(segment.block.text)
            and any(round(size) >= 10 for size in segment.block.font_sizes)
        ):
            self.stack.pop()

    def _note_boundary(self, text):
        if re.match(
            r"^(?:\*+\s*)?Nota(?:\s+\d+|\s+adicional)?\s*:", text, re.IGNORECASE
        ):
            self._note("editorial_note")
        elif re.match(r"^(?:\(\*+\)|\*+\s)", text):
            self._note("editorial_note", paragraph=True)
        elif text.startswith("Visto e aprovado em Conselho de Ministros"):
            self._note("closing")

    def _wrapped(self, segment):
        return (
            self.previous is not None
            and self.previous.page.page_number == segment.page.page_number
            and segment.block.bbox[1] - self.previous.block.bbox[3] < 3
        )

    def _numbered_boundary(self, text):
        if match := PARAGRAPH.match(text):
            self._trim({"paragraph", "subparagraph", "item"})
            self._start("paragraph", match[1], paragraph=match[1])
        elif (match := ITEM.match(text)) and self._is_item(match[1]):
            # Numeric and Roman subitems are nested under their alínea.
            if match[1].isdigit():
                self._trim({"item"})
            elif (
                self.stack[-1].address["kind"] == "item"
                and not self.stack[-1].address["label"].isdigit()
            ):
                self.stack.pop()
            self._start("item", match[1])
        elif match := SUBPARAGRAPH.match(text):
            self._trim({"subparagraph", "item"})
            self._start("subparagraph", match[1].lower(), subparagraph=match[1].lower())

    def _is_item(self, label):
        if label.isdigit() or len(label) > 1:
            return True
        leaf = self.stack[-1]
        text = "\n".join(s.block.text for s in leaf.own_segments)
        return label.lower() == "i" and (
            leaf.address["kind"] == "item"
            or EDITORIAL.sub("", text).rstrip().endswith(":")
        )

    def _append(self, segment: _Segment):
        leaf = self.stack[-1]
        targets = [leaf] if leaf.address["kind"] == "note" else self.stack
        for target in targets:
            target.segments.append(segment)
        leaf.own_segments.append(segment)
        body = PARAGRAPH.sub("", segment.block.text)
        body = SUBPARAGRAPH.sub("", body)
        if re.match(r"^\(Revogad[oa]\b", body, re.IGNORECASE):
            leaf.status = "repealed"
        if leaf.address["kind"] == "note" and re.search(
            r"transitóri[oa]", segment.block.text, re.IGNORECASE
        ):
            leaf.address["role"] = "transitional"

    def consume(self, segment: _Segment):
        text = segment.block.text.strip()
        if re.search(r"\.{3,}\s*\d+\s*$", text):
            self.excluded += 1
            return
        heading = _heading(segment)
        if heading and (identity := _document_identity(text)):
            self._document(*identity)
            self._append(segment)
            return
        if not self.stack:
            self.excluded += 1
            return
        if heading and self._structural_heading(text):
            self._append(segment)
            return
        self._body(segment, heading)
        self._append(segment)

    def _body(self, segment, heading):
        text = segment.block.text.strip()
        if self.title_owner is not None and heading and not EDITORIAL.search(text):
            self.title_owner.title.append(text)
            if self.title_owner.address["kind"] == "article" and re.search(
                r"transitóri[oa]", text, re.IGNORECASE
            ):
                self.title_owner.address["role"] = "transitional"
        else:
            self.title_owner = None
            self._body_boundary(segment)


def parse_cirs(
    pages: list[ExtractedPage], tables: list[ExtractedTable]
) -> StructureResult:
    if not pages or len({p.snapshot_id for p in pages}) != 1:
        raise ValueError("Expected pages from one snapshot")
    parser = _Parser()
    for segment in _segments(pages, tables):
        parser.consume(segment)
        parser.previous = segment
    if not parser.drafts:
        raise ValueError("No recognized instrument boundary")
    texts = {
        d.address["legal_unit_id"]: "\n".join(s.block.text for s in d.own_segments)
        for d in parser.drafts
    }
    units = resolve_references([d.finish() for d in parser.drafts], texts)
    return StructureResult(
        units=units,
        tables=tables,
        excluded_block_count=parser.excluded,
        extraction_issues=[
            {"page_number": p.page_number, "issues": p.issues}
            for p in pages
            if p.issues
        ],
    )
