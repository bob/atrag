"""Only complete, unambiguous local addresses become resolved references."""

import json
from pathlib import Path

import pytest

from atrag.ingestion.legal_references import resolve_references
from atrag.schemas.ingestion.structure import LegalUnit


@pytest.fixture
def article16_units():
    return [
        LegalUnit.model_validate(u)
        for u in json.loads(
            Path("tests/fixtures/cirs/article16-units.json").read_bytes()
        )
    ]


@pytest.mark.parametrize(
    "text,numbers",
    [
        ("Ver os n.os 14 e 16.", [14, 16]),
        ("Ver os números 14, 15 e 16 do artigo 16.º.", [14, 15, 16]),
        ("Ver os n.os 14 a 16.", [14, 15, 16]),
        ("Ver os n.os 14 a 14.", [14]),
        ("Ver os N.OS 14 E 16.", [14, 16]),
        ("Ver os artigos 16.º a 16.º.", [None]),
        ("Ver os artigos 16.º e 16.º.", [None]),
    ],
)
def test_lists_and_ranges_resolve_every_address(article16_units, text, numbers):
    units = article16_units
    owner = next(u for u in units if u.paragraph == "4")
    texts = {u.legal_unit_id: "" for u in units}
    texts[owner.legal_unit_id] = text
    resolved = resolve_references(units, texts)
    ref = next(
        u for u in resolved if u.legal_unit_id == owner.legal_unit_id
    ).references[0]
    prefix = "pt:cirs:code:article:16"
    assert ref.status == "resolved"
    assert ref.target_ids == [
        prefix if number is None else f"{prefix}:paragraph:{number}"
        for number in numbers
    ]
    assert ref.reason is None


@pytest.mark.parametrize(
    "text",
    [
        "Ver os n.os 14 e 999.",
        "Ver os n.os 16 a 14.",
        "Ver os n.os 14 e 16 do Código do IRC.",
        "Ver os números anteriores.",
        "Ver a alínea b) do mesmo número.",
        "Ver o número anterior do artigo 16.º.",
        "Ver os artigos 16-A a 16-B.",
        "Ver os n.os 14, 16 a 14.",
        "Ver os n.os 0 a 1000000000.",
    ],
)
def test_ambiguous_missing_and_external_lists_stay_unresolved(article16_units, text):
    owner = next(u for u in article16_units if u.paragraph == "15")
    texts = {u.legal_unit_id: "" for u in article16_units}
    texts[owner.legal_unit_id] = text
    ref = next(
        u
        for u in resolve_references(article16_units, texts)
        if u.legal_unit_id == owner.legal_unit_id
    ).references[0]
    assert ref.status == "unresolved"
    assert ref.target_ids == []
    assert ref.raw_text in text


def test_legislative_identifiers_are_not_local_paragraph_references(article16_units):
    unit = next(u for u in article16_units if u.paragraph == "4")
    texts = {u.legal_unit_id: "" for u in article16_units}
    texts[unit.legal_unit_id] = "Redação da Lei n.º 14/2026. Ver o n.º 14."
    changed = resolve_references(article16_units, texts)
    refs = next(u for u in changed if u.legal_unit_id == unit.legal_unit_id).references
    assert [r.raw_text for r in refs] == ["n.º 14"]
    assert refs[0].target_ids == ["pt:cirs:code:article:16:paragraph:14"]


def test_detached_note_does_not_inherit_a_relative_paragraph_anchor(article16_units):
    parent = next(u for u in article16_units if u.paragraph == "15")
    note = parent.model_copy(
        update={
            "legal_unit_id": parent.legal_unit_id + ":note:1",
            "parent_id": parent.legal_unit_id,
            "kind": "note",
        }
    )
    units = [*article16_units, note]
    texts = {u.legal_unit_id: "" for u in units}
    texts[note.legal_unit_id] = "Ver o número anterior."
    ref = resolve_references(units, texts)[-1].references[0]
    assert ref.status == "unresolved"
    assert not ref.target_ids
