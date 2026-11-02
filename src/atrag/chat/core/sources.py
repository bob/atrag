"""Assign stable citation IDs to identical source slices."""


def register_sources(registry, records):
    delivered = []
    for record in records:
        existing = next(
            (
                s
                for s in registry
                if {k: v for k, v in s.items() if k != "citation_id"} == record
            ),
            None,
        )
        if existing is None:
            existing = {"citation_id": len(registry) + 1, **record}
            registry.append(existing)
        delivered.append(existing)
    return delivered
