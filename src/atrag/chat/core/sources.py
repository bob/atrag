"""Assign stable citation IDs to identical source slices."""

from urllib.parse import quote, urlsplit

from atrag.schemas.ingestion.sources import SourceDefinition, SourceFormat


def citation_url(source: SourceDefinition, locations: list[dict]) -> str | None:
    """Resolve a live official link from this snapshot's checked locations."""
    if source.source_format == SourceFormat.HTML_COLLECTION:
        location = next((p for p in locations if p.get("kind") == "html"), {})
        url = location.get("page_url")
        fragment = quote(location.get("anchor", ""), safe="")
    else:
        url = str(source.source_url) if source.source_url is not None else None
        page = next((p["page"] for p in locations if p.get("kind") == "pdf"), None)
        fragment = f"page={page}" if page is not None else ""
    return urlsplit(url)._replace(fragment=fragment).geturl() if url else None


def register_sources(registry, records):
    delivered = []
    for record in records:
        existing = next(
            (
                s
                for s in registry
                if {
                    k: v
                    for k, v in s.items()
                    if k not in ("citation_id", "citation_key")
                }
                == record
            ),
            None,
        )
        if existing is None:
            existing = {"citation_id": len(registry) + 1, **record}
            registry.append(existing)
        existing["citation_key"] = (
            f"source:{existing['citation_id']}:{record.get('document_type', 'document')}:"
            f"{record.get('citation_label', record['address']['block_id'])}"
        )
        delivered.append(existing)
    return delivered
