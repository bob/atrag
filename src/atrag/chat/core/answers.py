"""Resolve answer references exclusively against evidence returned by tools."""

import re
from collections.abc import Callable


class InvalidAnswer(ValueError):
    """The model has not produced a complete answer with valid references."""


def render_citations(answer: str, registry: list[dict]) -> str:
    """Resolve descriptive evidence keys; the server owns numeric references."""
    if any(s.get("citation_key") for s in registry) and re.search(r"\[\d+\]", answer):
        raise InvalidAnswer("Use evidence keys; numeric citations are server-owned")
    sources = {s.get("citation_key"): s["citation_id"] for s in registry}

    def replace(match):
        if match[1] not in sources:
            raise InvalidAnswer("Answer contains an unknown evidence key")
        return f"[{sources[match[1]]}]"

    answer = re.sub(r"\[(source:[^\]\n]+)\]", replace, answer)
    if "[source:" in answer:
        raise InvalidAnswer("Answer contains an unfinished evidence key")
    return answer


class CitationStream:
    """Keep a key split across model chunks private until it can be resolved."""

    def __init__(self, sources: Callable[[], list[dict]]):
        self.sources = sources
        self.pending = ""

    def feed(self, text: str) -> str:
        text = self.pending + text
        split = text.rfind("[") if text.rfind("[") > text.rfind("]") else len(text)
        self.pending = text[split:]
        ready = text[:split]
        return render_citations(ready, self.sources()) if "[" in ready else ready

    def finish(self) -> str:
        text, self.pending = self.pending, ""
        return render_citations(text, self.sources())


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
