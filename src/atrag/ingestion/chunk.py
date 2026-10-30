"""Pure structural chunking: preserve a legal block or report its omission."""

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable

from atrag.ingestion.extraction import normalize_text
from atrag.schemas.ingestion.fragments import (
    ChunkParameters,
    ChunkResult,
    FragmentContext,
    FragmentOmission,
    SearchFragment,
)
from atrag.schemas.ingestion.sources import SourceDefinition
from atrag.schemas.ingestion.structure import LegalUnit, StructureResult

EXCEPTION = re.compile(
    r"\b(salvo|exceto|excepto|exceção|excepção|sem prejuízo|ressalv\w*|"
    r"não se aplica|não [ée] aplic[áa]vel)\b",
    re.IGNORECASE,
)


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

    def own_unit(self, unit):
        children = [c for c in self.children[unit.legal_unit_id] if c.kind != "note"]
        excluded = {l.model_dump_json() for c in children for l in c.locators}
        tables = {i for c in children for i in c.table_ids}
        text = self.own_text(unit)
        return unit.model_copy(
            update={
                "text_original": text,
                "text_normalized": normalize_text(text),
                "locators": [
                    l for l in unit.locators if l.model_dump_json() not in excluded
                ],
                "table_ids": [i for i in unit.table_ids if i not in tables],
            }
        )

    def ancestry(self, unit):
        parents = []
        parent = self.by_id.get(unit.parent_id)
        while parent is not None and parent.kind != "document":
            parents.append(parent)
            parent = self.by_id.get(parent.parent_id)
        return [
            p
            for p in reversed(parents)
            if p.kind in ("article", "paragraph", "subparagraph", "item", "note")
        ]

    def context(self, unit):
        parents = self.ancestry(unit)
        context = [(self.own_unit(p), self.own_text(p)) for p in parents]
        for owner in [*parents, *self.embedded(unit)]:
            context.extend(
                (note, note.text_original)
                for note in self.children[owner.legal_unit_id]
                if note.kind == "note" and note.role != "closing"
            )
        context.extend(self.exceptions(unit))
        return list({u.legal_unit_id: (u, text) for u, text in context}.values())

    def exceptions(self, unit):
        for sibling in self.children[unit.parent_id]:
            if sibling == unit or sibling.kind == "note":
                continue
            if not EXCEPTION.search(sibling.text_normalized):
                continue
            targets = {i for r in sibling.references for i in r.target_ids}
            if unit.kind in ("subparagraph", "item") or unit.legal_unit_id in targets:
                yield sibling, sibling.text_original

    def candidates(self):
        for unit in self.by_id.values():
            if unit.kind == "article":
                paragraphs = [
                    c
                    for c in self.children[unit.legal_unit_id]
                    if c.kind == "paragraph"
                ]
                yield from paragraphs or [unit]
            elif unit.kind == "document" and unit.role != "code":
                yield self.own_unit(unit)
            elif unit.kind == "note" and unit.role == "closing":
                yield unit


def _embedding_text(unit, source, context):
    scope = {
        "code": "Código CIRS",
        "approving_decree": "Decreto-Lei n.º 442-A/88 — diploma de aprovação",
        "preamble": "Preâmbulo",
    }[unit.document_scope]
    return "\n\n".join(
        [
            source.document_title,
            scope,
            unit.citation_label,
            *(text for _, text in context if text),
            unit.text_original,
        ]
    )


def _fragment(unit, tree, source, parser_version, parameters, count_tokens):
    context = tree.context(unit)
    members = [unit] if unit.kind == "document" else list(tree.embedded(unit))
    evidence = {u.legal_unit_id: u for u in [*members, *(u for u, _ in context)]}
    locators = _unique(l for u in evidence.values() for l in u.locators)
    references = _unique(r for u in evidence.values() for r in u.references)
    issues = sorted({i for u in evidence.values() for i in u.issues})
    if any(r.status == "unresolved" for r in references):
        issues.append("unresolved_reference")
    text = _embedding_text(unit, source, context)
    table_ids = sorted({i for u in evidence.values() for i in u.table_ids})
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
            }
        ),
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
        page_start=min(l.page_number for l in locators),
        page_end=max(l.page_number for l in locators),
        printed_page_labels=list(
            dict.fromkeys(label for l in locators for label in l.printed_page_labels)
        ),
        editorial_notes=_unique(
            n for u in evidence.values() for n in u.editorial_notes
        ),
        references=references,
        table_ids=table_ids,
        tables=[tree.tables[i] for i in table_ids],
        issues=issues,
        review_status="needs_review" if issues else "ready",
    )
    identity = fragment.model_dump_json(exclude={"fragment_id", "source"})
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return fragment.model_copy(update={"fragment_id": digest})


def _chunks(unit, tree, source, parser_version, parameters, count_tokens):
    fragment = _fragment(unit, tree, source, parser_version, parameters, count_tokens)
    children = [
        c
        for c in tree.children[unit.legal_unit_id]
        if c.kind in ("subparagraph", "item")
    ]
    if fragment.token_count > parameters.target_tokens and children:
        for child in children:
            yield from _chunks(
                child, tree, source, parser_version, parameters, count_tokens
            )
        return
    unsafe = [t.table_id for t in fragment.tables if t.review_status != "verified"]
    if unsafe or fragment.token_count > parameters.max_tokens:
        yield FragmentOmission(
            legal_unit_id=unit.legal_unit_id,
            reason="unverified_table" if unsafe else "over_token_limit",
            token_count=fragment.token_count,
            table_ids=unsafe,
        )
    else:
        yield fragment


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
        for unit in tree.candidates()
        for p in _chunks(unit, tree, source, parser_version, parameters, count_tokens)
    ]
    return ChunkResult(
        fragments=[p for p in pieces if isinstance(p, SearchFragment)],
        omissions=[p for p in pieces if isinstance(p, FragmentOmission)],
    )
