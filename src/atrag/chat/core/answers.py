"""Resolve answer references exclusively against evidence returned by tools."""

import re


class InvalidAnswer(ValueError):
    """The model has not produced a complete answer with valid references."""


def answer_sources(answer: str, registry: list[dict]) -> list[dict]:
    if not answer.strip():
        raise InvalidAnswer("The model returned an empty answer")
    references = re.findall(r"\[([-\d][\d,\s–-]*)\]", answer)
    if registry and not references:
        raise InvalidAnswer("Answer omits citations to the retrieved evidence")
    sources = {str(source["citation_id"]): source for source in registry}
    if any(reference not in sources for reference in references):
        raise InvalidAnswer("Answer contains an unknown or malformed citation")
    return [sources[reference] for reference in dict.fromkeys(references)]
