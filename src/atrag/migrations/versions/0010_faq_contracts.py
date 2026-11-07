"""Persist explicit source formats and allow FAQ explanatory units."""

from alembic import op
from sqlalchemy import Column, Text, text

revision = "0010_faq_contracts"
down_revision = "0009_english_default"
branch_labels = None
depends_on = None


def _address_constraints(*, faq: bool) -> None:
    domains = (
        ("document_scope", ("code", "approving_decree", "preamble"), ("faq",)),
        (
            "kind",
            (
                "document",
                "chapter",
                "section",
                "subsection",
                "article",
                "paragraph",
                "subparagraph",
                "item",
                "note",
            ),
            ("topic", "faq_entry"),
        ),
        (
            "role",
            (
                "norm",
                "approving_decree",
                "code",
                "preamble",
                "editorial_note",
                "transitional",
                "closing",
            ),
            ("at_explanation",),
        ),
    )
    for table in ("legal_units", "search_fragments"):
        for column, legacy, additions in domains:
            name = f"{table}_{column}_check"
            values = legacy + (additions if faq else ())
            choices = ", ".join(repr(value) for value in values)
            op.drop_constraint(name, table, type_="check")
            op.create_check_constraint(name, table, f"{column} IN ({choices})")


def upgrade() -> None:
    op.add_column(
        "source_snapshots",
        Column("source_format", Text, nullable=False, server_default="pdf"),
    )
    op.create_check_constraint(
        "source_snapshots_source_format_check",
        "source_snapshots",
        "source_format IN ('pdf', 'html_collection')",
    )
    _address_constraints(faq=True)


def downgrade() -> None:
    if op.get_bind().scalar(
        text(
            "SELECT EXISTS (SELECT 1 FROM source_snapshots WHERE source_format <> 'pdf')"
        )
    ):
        raise ValueError("Cannot discard HTML source provenance")
    # Restoring the old domains also refuses to erase existing FAQ units.
    _address_constraints(faq=False)
    op.drop_column("source_snapshots", "source_format")
