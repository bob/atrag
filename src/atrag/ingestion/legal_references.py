"""Conservative local reference resolution; external acts remain unresolved."""

import re
from itertools import chain

from atrag.schemas.ingestion.structure import LegalReference, LegalUnit

NUMBER = r"\d+(?:\.º)?(?:-[A-Z])?"
ARTICLE_WORD = r"(?:artigos?|art\.º|artº)"
ARTICLE = (
    rf"{ARTICLE_WORD}\s+(?:{NUMBER}(?:\s*(?:e|a|,)\s*{NUMBER})*|anterior|seguinte)"
)
PARAGRAPH = rf"(?:n\.(?:º|os)|nº|números?)\s+(?:{NUMBER}(?:\s*(?:e|a|,)\s*{NUMBER})*|anterior(?:es)?|seguintes?)"
PARAGRAPH_QUALIFIER = rf"(?:{PARAGRAPH}|mesmo número)"
SUBPARAGRAPH = r"alíneas?\s+(?:[a-z]\)|anterior|seguinte)"
REFERENCE = re.compile(
    rf"\b{SUBPARAGRAPH}(?:\s+do\s+{PARAGRAPH_QUALIFIER})?(?:\s+do\s+{ARTICLE})?|\b{PARAGRAPH}(?:\s+do\s+{ARTICLE})?|\b{ARTICLE}",
    re.IGNORECASE,
)
EXTERNAL = re.compile(
    r"^\s*,?\s*d[oa]\s+(?:Código\b|CIVA\b|CIRC\b|EBF\b|LGT\b|RGIT\b|Estatuto\b|Regulamento\b|Regime\b|Lei\b|Decreto|Constituição)",
    re.IGNORECASE,
)
ACT_NUMBER = re.compile(
    r"\b(?:Lei(?:\s+Orgânica)?|Decreto(?:-Lei)?|Portaria|Despacho|Regulamento|Diretiva|Decisão)\s+$",
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
    if re.search(r"\b(?:artigos|alíneas|números|mesmo)\b", text, re.IGNORECASE):
        return None
    if unit.kind == "note" and not cirs:
        return None
    relative = re.search(r"\b(?:anterior|seguinte)\b", text, re.IGNORECASE)
    if relative:
        # Compound relatives need their parent qualifiers; do not guess an anchor.
        if re.search(r"\s+do\s+", text, re.IGNORECASE):
            return None
        if cirs and (unit.instrument_id, unit.document_scope) != ("pt:cirs", "code"):
            return None
        return _relative(unit, kind, relative[0].lower(), units)
    return _subparagraph_target(unit, text, cirs)


def _address_numbers(text):
    groups = []
    for part in re.split(r"\s*(?:,|\be\b)\s*", text, flags=re.IGNORECASE):
        bounds = [
            article_number(b) for b in re.split(r"\s+a\s+", part, flags=re.IGNORECASE)
        ]
        if len(bounds) == 1:
            groups.append(bounds)
        elif all(b.isdigit() for b in bounds) and int(bounds[0]) <= int(bounds[1]):
            groups.append(map(str, range(int(bounds[0]), int(bounds[1]) + 1)))
        else:
            return []
    return chain.from_iterable(groups)


def _targets(unit, text, kind, units, cirs):
    head, *qualifier = re.split(r"\s+do\s+", text, flags=re.IGNORECASE)
    if kind == "subparagraph" or not re.search(r"\d", head):
        target = _target(unit, text, kind, units, cirs)
        return [target] if target else []
    if unit.kind == "note" and not cirs:
        return []
    value = re.search(rf"{NUMBER}.*", head)[0]
    numbers = _address_numbers(value)
    if kind == "article":
        return (_article_id(unit, n, cirs=cirs) for n in numbers)
    return _paragraph_targets(unit, numbers, qualifier, cirs)


def _paragraph_targets(unit, numbers, qualifier, cirs):
    article = unit.article
    if qualifier:
        match = re.fullmatch(
            rf"{ARTICLE_WORD}\s+({NUMBER})", qualifier[0], re.IGNORECASE
        )
        if match is None:
            return []
        article = article_number(match[1])
    prefix = _article_id(unit, article, cirs=cirs)
    return (f"{prefix}:paragraph:{n}" for n in numbers)


def _subparagraph_target(unit: LegalUnit, text: str, cirs: bool) -> str | None:
    article = re.search(rf"{ARTICLE_WORD}\s+({NUMBER})", text, re.IGNORECASE)
    article_id = _article_id(
        unit, article_number(article[1]) if article else unit.article, cirs=cirs
    )
    numbers = re.findall(
        r"\d+", re.split(rf"\s+do\s+{ARTICLE_WORD}", text, flags=re.IGNORECASE)[0]
    )
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
    kind = _kind(text)
    cirs = bool(
        re.match(
            r"\s*,?\s*do (?:CIRS|Código (?:do IRS|do Imposto sobre o Rendimento das Pessoas Singulares))\b",
            tail,
            re.IGNORECASE,
        )
    )
    external = bool(EXTERNAL.match(tail)) and not cirs
    targets = _existing_targets(
        [] if external else _targets(unit, text, kind, units, cirs), ids
    )
    resolved = bool(targets)
    return LegalReference(
        raw_text=text,
        kind=kind,
        status="resolved" if resolved else "unresolved",
        target_ids=targets if resolved else [],
        reason=None
        if resolved
        else "external_instrument"
        if external
        else "missing_or_ambiguous_target",
    )


def _existing_targets(targets, ids):
    found = {}
    for target in targets:
        if target not in ids:
            return []
        found[target] = None
    return list(found)


def _kind(text):
    lower = text.lower()
    if lower.startswith("alínea"):
        return "subparagraph"
    return "article" if lower.startswith("art") else "paragraph"


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
            if not ACT_NUMBER.search(text[: m.start()])
        ]
        result.append(unit.model_copy(update={"references": refs}))
    return result
