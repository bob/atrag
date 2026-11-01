"""Lossless, contiguous text ranges; formatting is measured by the caller."""

import re

BOUNDARIES = (
    re.compile(r"\n(?=(?:\d+(?:-[A-Z])?\s*[-–—]|[a-zivxlcdm]+\))\s)", re.IGNORECASE),
    re.compile(r"(?<=[.!?;])\s+"),
    re.compile(r"\s+"),
)


def _atoms(text, start, end, level, fits, hard_limit):
    if fits(start, end):
        yield start, end
        return
    if level == len(BOUNDARIES):
        if end - start == 1:
            if hard_limit(start, end):
                yield start, end
                return
            raise ValueError("Envelope leaves no room for a source character")
        middle = (start + end) // 2
        yield from _atoms(text, start, middle, level, fits, hard_limit)
        yield from _atoms(text, middle, end, level, fits, hard_limit)
        return
    cursor = start
    for match in BOUNDARIES[level].finditer(text, start, end):
        boundary = match.end()
        yield from _atoms(text, cursor, boundary, level + 1, fits, hard_limit)
        cursor = boundary
    if cursor != end:
        yield from _atoms(text, cursor, end, level + 1, fits, hard_limit)


def pack_ranges(atoms, fits):
    """Pack already safe atoms without losing their preferred boundaries."""
    start, end = next(atoms)
    for left, right in atoms:
        if fits(start, right):
            end = right
        else:
            yield start, end
            start, end = left, right
    yield start, end


def text_ranges(text, measure, target, maximum):
    # Repeated introductory text can change the envelope at each source offset.
    fits = lambda start, end: (
        measure(start, end) <= (maximum if measure(start, start) >= target else target)
    )
    atoms = _atoms(text, 0, len(text), 0, fits, lambda a, b: measure(a, b) <= maximum)
    return list(pack_ranges(atoms, fits))
