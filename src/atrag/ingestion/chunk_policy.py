"""Source-specific selection and context; range packing stays shared."""

from atrag.ingestion.table_parts import table_header_text


def candidates(tree):
    for unit in tree.by_id.values():
        if unit.kind == "faq_entry":
            yield unit
        elif unit.kind == "article":
            paragraphs = [
                c for c in tree.children[unit.legal_unit_id] if c.kind == "paragraph"
            ]
            yield from paragraphs or [unit]
        elif unit.kind == "document" and unit.role != "code" or unit.kind == "note":
            yield unit


def context(tree, unit):
    if unit.document_scope == "faq":
        return []
    parents = []
    parent = tree.by_id.get(unit.parent_id)
    while parent is not None and parent.kind != "document":
        parents.append(parent)
        parent = tree.by_id.get(parent.parent_id)
    return [
        (p, tree.own_text(p))
        for p in reversed(parents)
        if p.kind in ("article", "paragraph", "subparagraph", "item", "note")
    ]


def intro(tree, unit, parameters, count_tokens):
    if unit.faq is not None:
        first, separator, _ = unit.text_original.partition("\n")
        return (
            first
            if separator and count_tokens(first) <= parameters.target_tokens // 4
            else ""
        )
    children = [
        c
        for c in tree.children[unit.legal_unit_id]
        if c.kind in ("subparagraph", "item")
    ]
    if not children:
        return ""
    return unit.text_original[
        : unit.text_original.index(children[0].text_original)
    ].strip()


def embedding_text(unit, source, context, text, repeated_intro="", tables=()):
    if unit.faq is not None:
        return "\n\n".join(
            value
            for value in (
                source.document_title,
                " > ".join(unit.faq.topic_path),
                unit.citation_label,
                unit.faq.question,
                repeated_intro,
                table_header_text(tables),
                text,
            )
            if value
        )
    scope = {
        "code": "Código CIRS",
        "approving_decree": "Decreto-Lei n.º 442-A/88 — diploma de aprovação",
        "preamble": "Preâmbulo",
    }[unit.document_scope]
    return "\n\n".join(
        text
        for text in [
            source.document_title,
            scope,
            unit.citation_label,
            *(text for _, text in context if text),
            repeated_intro,
            table_header_text(tables),
            text,
        ]
        if text
    )
