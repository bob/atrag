"""Pure address normalization and conservative language hints."""

import re

from atrag.ingestion.legal_references import article_number
from atrag.schemas.retrieval import SearchFilters

CODE = r"(?:CIRS|CIVA|CIRC|EBF|LGT|RGIT)"
NUMBER = r"\d+(?:\.º)?(?:-[a-z])?"
ARTICLE = rf"(?:artigo|art\.)\s*({NUMBER})\s*(?:do\s+)?({CODE})\b"
CODE_FIRST = rf"\b({CODE})\s+({NUMBER})(?![\w-])"


def normalize_address(query):
    query = query.strip(" .?!\t\r\n")
    match = re.fullmatch(ARTICLE, query, re.IGNORECASE)
    if match:
        number, code = match.groups()
    else:
        match = re.fullmatch(CODE_FIRST, query, re.IGNORECASE)
        if match is None:
            return None
        code, number = match.groups()
    return SearchFilters(
        code=code.upper(), article=article_number(number), document_scope="code"
    )


def query_language(query, explicit):
    if explicit is not None:
        return explicit
    if normalize_address(query) is not None:
        raise ValueError("Set query_language to the interface language for an address")
    if re.search("[а-яё]", query, re.IGNORECASE):
        return "ru"
    if re.search(
        r"[ãõç]|\b(como|residência|rendimento|imposto|são|quais|fiscal|tributação|não)\b",
        query,
        re.IGNORECASE,
    ):
        return "pt"
    return "en"


def merge_filters(explicit, address):
    values = explicit.model_dump(exclude_none=True)
    if address is not None:
        for key, value in address.model_dump(exclude_none=True).items():
            if key in values and values[key] != value:
                raise ValueError("Query address conflicts with explicit filters")
            values[key] = value
    return SearchFilters(**values)
