"""Bounded search parts referencing complete, unchanged legal units."""

import hashlib
from collections import defaultdict
from collections.abc import Callable

from atrag.ingestion.chunk_policy import candidates, context, embedding_text, intro
from atrag.ingestion.extraction import normalize_text
from atrag.ingestion.table_parts import (
    TableRowTooLarge,
    bounded_table_ranges,
    select_tables,
    table_rows,
    table_text,
)
from atrag.ingestion.text_parts import text_ranges
from atrag.schemas.ingestion.fragments import (
    ChunkParameters,
    ChunkResult,
    FragmentContext,
    FragmentOmission,
    SearchFragment,
)
from atrag.schemas.ingestion.sources import SourceDefinition
from atrag.schemas.ingestion.structure import LegalUnit, StructureResult


def _unique(values):
    return list({v.model_dump_json(): v for v in values}.values())


class _Structure:
    def __init__(self, result: StructureResult):
        self.by_id: dict[str, LegalUnit] = {}
        self.children: dict[str | None, list[LegalUnit]] = defaultdict(list)
        for unit in result.units:
            if unit.legal_unit_id in self.by_id:
                raise ValueError("Duplicate legal unit identity")
            if unit.parent_id is not None and unit.parent_id not in self.by_id:
                raise ValueError("Expected parents before their children")
            self.by_id[unit.legal_unit_id] = unit
            self.children[unit.parent_id].append(unit)
        if len({u.snapshot_id for u in result.units}) != 1:
            raise ValueError("Expected one nonempty snapshot")
        self.tables = {t.table_id: t for t in result.tables}
        if len(self.tables) != len(result.tables):
            raise ValueError("Duplicate table identity")
        for unit in result.units:
            if not set(unit.table_ids) <= self.tables.keys():
                raise ValueError("Unknown table identity")

    def embedded(self, unit):
        yield unit
        for child in self.children[unit.legal_unit_id]:
            if child.kind != "note":
                yield from self.embedded(child)

    def own_text(self, unit):
        text = unit.text_original
        for child in self.children[unit.legal_unit_id]:
            if child.kind == "note":
                continue  # The parser stores detached notes outside parent text.
            before, found, after = text.partition(child.text_original)
            if not found:
                raise ValueError("Child text is absent from its parent")
            text = before + after
        return text.strip()

    def context(self, unit):
        return context(self, unit)


def _fragment(unit, tree, source, parser_version, parameters, count_tokens, part, rows):
    index, total, start, end, intro = part
    context = tree.context(unit)
    text_original = unit.text_original[start:end]
    members = [
        unit,
        *[
            u
            for u in tree.embedded(unit)
            if u != unit and u.text_original in text_original
        ],
    ]
    evidence = {u.legal_unit_id: u for u in [*members, *(u for u, _ in context)]}
    locators = unit.locators
    references = _unique(r for u in tree.embedded(unit) for r in u.references)
    issues = sorted({i for u in evidence.values() for i in u.issues})
    if any(r.status == "unresolved" for r in references):
        issues.append("unresolved_reference")
    tables, table_ranges = select_tables(rows, tree.tables, start, end)
    text = embedding_text(unit, source, context, text_original, intro, tables)
    table_ids = [t.table_id for t in tables]
    fragment = SearchFragment(
        **unit.model_dump(
            exclude={
                "locators",
                "page_start",
                "page_end",
                "printed_page_labels",
                "editorial_notes",
                "references",
                "issues",
                "table_ids",
                "text_original",
                "text_normalized",
            }
        ),
        text_original=text_original,
        text_normalized=normalize_text(text_original),
        part_index=index,
        part_count=total,
        source_start=start,
        source_end=end,
        repeated_intro=intro,
        fragment_id="0" * 64,
        parser_version=parser_version,
        parameters=parameters,
        source=source,
        legal_unit_ids=[u.legal_unit_id for u in members],
        context_unit_ids=[u.legal_unit_id for u, _ in context],
        context=[
            FragmentContext(
                legal_unit_id=u.legal_unit_id, text_original=t, locators=u.locators
            )
            for u, t in context
        ],
        embedding_text=text,
        embedding_input_hash=hashlib.sha256(text.encode()).hexdigest(),
        token_count=count_tokens(text),
        locators=locators,
        page_start=min((l.page_number for l in locators), default=None),
        page_end=max((l.page_number for l in locators), default=None),
        printed_page_labels=list(
            dict.fromkeys(label for l in locators for label in l.printed_page_labels)
        ),
        editorial_notes=_unique(
            n for u in evidence.values() for n in u.editorial_notes
        ),
        references=references,
        table_ids=table_ids,
        tables=tables,
        table_ranges=table_ranges,
        issues=issues,
        review_status="needs_review" if issues else "ready",
    )
    identity = fragment.model_dump_json(exclude={"fragment_id", "source"})
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return fragment.model_copy(update={"fragment_id": digest})


def _chunks(unit, tree, source, parser_version, parameters, count_tokens):
    unsafe = [i for i in unit.table_ids if tree.tables[i].review_status != "verified"]
    if unsafe:
        yield FragmentOmission(
            legal_unit_id=unit.legal_unit_id,
            reason="unverified_table",
            token_count=None,
            table_ids=unsafe,
        )
        return
    context = tree.context(unit)
    repeated = intro(tree, unit, parameters, count_tokens)
    rows = table_rows(unit, tree.tables)

    def measure(start, end):
        tables, _ = select_tables(rows, tree.tables, start, end)
        text = embedding_text(
            unit,
            source,
            context,
            unit.text_original[start:end],
            repeated if start else "",
            tables,
        )
        return max(count_tokens(text), count_tokens(table_text(tables)))

    try:
        ranges = (
            bounded_table_ranges(
                unit.text_original,
                rows,
                measure,
                parameters.target_tokens,
                parameters.max_tokens,
            )
            if rows
            else text_ranges(
                unit.text_original,
                measure,
                parameters.target_tokens,
                parameters.max_tokens,
            )
        )
    except TableRowTooLarge as error:
        yield FragmentOmission(
            legal_unit_id=unit.legal_unit_id,
            reason="table_row_over_token_limit",
            token_count=error.token_count,
            table_ids=[error.table_id],
        )
        return
    for index, (start, end) in enumerate(ranges):
        yield _fragment(
            unit,
            tree,
            source,
            parser_version,
            parameters,
            count_tokens,
            (index, len(ranges), start, end, repeated if start else ""),
            rows,
        )


def chunk_structure(
    result: StructureResult,
    source: SourceDefinition,
    *,
    parser_version: str,
    parameters: ChunkParameters,
    count_tokens: Callable[[str], int],
) -> ChunkResult:
    tree = _Structure(result)
    pieces = [
        p
        for unit in candidates(tree)
        for p in _chunks(unit, tree, source, parser_version, parameters, count_tokens)
    ]
    return ChunkResult(
        fragments=[p for p in pieces if isinstance(p, SearchFragment)],
        omissions=[p for p in pieces if isinstance(p, FragmentOmission)],
    )
