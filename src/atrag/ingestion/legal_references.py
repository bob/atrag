"""Conservative local reference resolution; external acts remain unresolved."""

import re

from atrag.schemas.ingestion.structure import LegalReference, LegalUnit

NUMBER = r"\d+(?:\.º)?(?:-[A-Z])?"
ARTICLE_WORD = r"(?:artigos?|art\.º|artº)"
ARTICLE = (
    rf"{ARTICLE_WORD}\s+(?:{NUMBER}(?:\s*(?:e|a|,)\s*{NUMBER})*|anterior|seguinte)"
)
PARAGRAPH = (
    r"(?:n\.(?:º|os)|nº|números?)\s+(?:\d+(?:\s*(?:e|a|,)\s*\d+)*|anterior|seguinte)"
)
SUBPARAGRAPH = r"alíneas?\s+(?:[a-z]\)|anterior|seguinte)"
REFERENCE = re.compile(
    rf"\b{SUBPARAGRAPH}(?:\s+do\s+{PARAGRAPH})?(?:\s+do\s+{ARTICLE})?|\b{PARAGRAPH}(?:\s+do\s+{ARTICLE})?|\b{ARTICLE}",
    re.IGNORECASE,
)
EXTERNAL = re.compile(
    r"^\s*,?\s*d[oa]\s+(?:Código\b|CIVA\b|CIRC\b|EBF\b|LGT\b|RGIT\b|Estatuto\b|Regulamento\b|Regime\b|Lei\b|Decreto|Constituição)",
    re.IGNORECASE,
)


def article_number(text: str) -> str:
    return text.replace(".º", "").upper()


def _article_id(unit: LegalUnit, number: str, *, cirs=False) -> str:
    prefix = "pt:cirs:code" if cirs else f"{unit.instrument_id}:{unit.document_scope}"
    return f"{prefix}:article:{number}"


def _context_address(unit: LegalUnit, kind: str) -> str:
    address = (
        _article_id(unit, unit.article) if kind == "article" else unit.legal_unit_id
    )
    if kind == "paragraph":
        address = f"{_article_id(unit, unit.article)}:paragraph:{unit.paragraph}"
    if kind == "subparagraph":
        address = f"{_article_id(unit, unit.article)}:paragraph:{unit.paragraph}:subparagraph:{unit.subparagraph}"
    return address


def _peers(current: LegalUnit, units: list[LegalUnit]) -> list[str]:
    if current.kind == "article":
        return [
            u.legal_unit_id
            for u in units
            if u.kind == "article"
            and (u.instrument_id, u.document_scope)
            == (current.instrument_id, current.document_scope)
        ]
    return [
        u.legal_unit_id
        for u in units
        if u.kind == current.kind and u.parent_id == current.parent_id
    ]


def _relative(
    unit: LegalUnit, kind: str, direction: str, units: list[LegalUnit]
) -> str | None:
    # Relative addresses follow the observed order, including letter suffixes.
    address = _context_address(unit, kind)
    current = next((u for u in units if u.legal_unit_id == address), None)
    if current is None:
        return None
    peers = _peers(current, units)
    position = peers.index(address) + (-1 if direction == "anterior" else 1)
    return peers[position] if 0 <= position < len(peers) else None


def _target(
    unit: LegalUnit, text: str, kind: str, units: list[LegalUnit], cirs: bool
) -> str | None:
    if re.search(r"\b(?:artigos|alíneas|números)\b", text, re.IGNORECASE):
        return None
    if (unit.kind == "note" or unit.document_scope == "preamble") and not cirs:
        return None
    relative = re.search(r"\b(?:anterior|seguinte)\b", text, re.IGNORECASE)
    if relative:
        # Compound relatives need their parent qualifiers; do not guess an anchor.
        if re.search(r"\s+do\s+", text, re.IGNORECASE):
            return None
        if cirs and (unit.instrument_id, unit.document_scope) != ("pt:cirs", "code"):
            return None
        return _relative(unit, kind, relative[0].lower(), units)
    return _exact_target(unit, text, kind, cirs)


def _exact_target(unit: LegalUnit, text: str, kind: str, cirs: bool) -> str | None:
    article = re.search(rf"{ARTICLE_WORD}\s+({NUMBER})", text, re.IGNORECASE)
    article_id = _article_id(
        unit, article_number(article[1]) if article else unit.article, cirs=cirs
    )
    if kind == "article":
        return article_id
    numbers = re.findall(
        r"\d+", re.split(rf"\s+do\s+{ARTICLE_WORD}", text, flags=re.IGNORECASE)[0]
    )
    if kind == "paragraph":
        return f"{article_id}:paragraph:{numbers[0]}" if len(numbers) == 1 else None
    letter = re.search(r"alínea\s+([a-z])\)", text, re.IGNORECASE)
    return (
        f"{article_id}:paragraph:{numbers[0] if len(numbers) == 1 else unit.paragraph}:subparagraph:{letter[1].lower()}"
        if letter
        else None
    )


def _reference(
    unit: LegalUnit, match, tail: str, units: list[LegalUnit], ids: set[str]
) -> LegalReference:
    text = match[0]
    lower = text.lower()
    kind = "article" if lower.startswith("art") else "paragraph"
    if lower.startswith("alínea"):
        kind = "subparagraph"
    cirs = bool(
        re.match(
            r"\s*,?\s*do (?:CIRS|Código (?:do IRS|do Imposto sobre o Rendimento das Pessoas Singulares))\b",
            tail,
            re.IGNORECASE,
        )
    )
    external = bool(EXTERNAL.match(tail)) and not cirs
    target = None if external else _target(unit, text, kind, units, cirs)
    resolved = target in ids
    return LegalReference(
        raw_text=text,
        kind=kind,
        status="resolved" if resolved else "unresolved",
        target_ids=[target] if resolved else [],
        reason=None
        if resolved
        else "external_instrument"
        if external
        else "missing_or_ambiguous_target",
    )


def resolve_references(
    units: list[LegalUnit], texts: dict[str, str]
) -> list[LegalUnit]:
    ids = {u.legal_unit_id for u in units}
    result = []
    for unit in units:
        text = " ".join(
            line
            for line in texts[unit.legal_unit_id].splitlines()
            if not re.fullmatch(rf"Artigo\s+(?:{NUMBER}|único)", line, re.IGNORECASE)
        )
        refs = [
            _reference(unit, m, text[m.end() :], units, ids)
            for m in REFERENCE.finditer(text)
        ]
        result.append(unit.model_copy(update={"references": refs}))
    return result
