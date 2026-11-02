"""Prepared materials have typed locations independent of PDF/legal structure."""

import pytest
from pydantic import ValidationError

from atrag.schemas.materials import PreparedMaterial


def test_html_material_has_stable_version_and_no_pdf_address():
    content = {
        "source_id": "faq",
        "source_sha256": "a" * 64,
        "preparation_version": "html-1",
        "metadata": {"title": "FAQ"},
        "blocks": [
            {
                "block_id": "residency",
                "text": "Residence rules",
                "location": {
                    "kind": "html",
                    "section": "Residence",
                    "anchor": "residency",
                },
            }
        ],
    }
    material = PreparedMaterial(**content)
    assert material.blocks[0].location.kind == "html"
    assert material.snapshot_id == PreparedMaterial(**content).snapshot_id
    for change in (
        {"metadata": {"title": "New FAQ"}},
        {"preparation_version": "html-2"},
        {
            "blocks": [
                {
                    "block_id": "residency",
                    "text": "New rules",
                    "location": {"kind": "docx", "paragraph": 2},
                }
            ]
        },
    ):
        assert (
            PreparedMaterial.model_validate({**content, **change}).snapshot_id
            != material.snapshot_id
        )


@pytest.mark.parametrize(
    "blocks",
    [
        [],
        [{"block_id": "x", "text": "Text", "location": {"kind": "pdf", "page": 0}}],
        [{"block_id": "x", "text": "Text", "location": {"kind": "docx"}}],
        [{"block_id": "x", "text": "Text", "location": {"kind": "html", "anchor": "a"}}]
        * 2,
    ],
)
def test_invalid_or_duplicate_blocks_are_rejected(blocks):
    with pytest.raises(ValidationError):
        PreparedMaterial(
            source_id="faq",
            source_sha256="a" * 64,
            preparation_version="1",
            metadata={},
            blocks=blocks,
        )


def test_material_identity_is_independent_of_metadata_key_order():
    content = {
        "source_id": "faq",
        "source_sha256": "a" * 64,
        "preparation_version": "1",
        "metadata": {"title": "FAQ", "revision": "2"},
        "blocks": [
            {
                "block_id": "a",
                "text": "Text",
                "location": {"kind": "docx", "paragraph": 1},
            }
        ],
    }
    first = PreparedMaterial(**content)
    content["metadata"] = {"revision": "2", "title": "FAQ"}
    assert PreparedMaterial(**content).snapshot_id == first.snapshot_id
