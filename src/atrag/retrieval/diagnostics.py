"""Select source diagnostics for evidence actually returned by search."""


def _returned_scope(hits):
    unit_ids, pages, table_ids = set(), set(), set()
    for hit in hits:
        unit_ids.update(hit.fragment.legal_unit_ids)
        for unit in (hit.fragment, *hit.related_units):
            unit_ids.add(unit.legal_unit_id)
            pages.update(locator.page_number for locator in unit.locators)
            table_ids.update(unit.table_ids)
        for table in (*hit.fragment.tables, *hit.related_tables):
            table_ids.add(table.table_id)
            pages.update(part.page_number for part in table.parts)
    return unit_ids, pages, table_ids


def hit_diagnostics(hits, report):
    unit_ids, pages, table_ids = _returned_scope(hits)
    return scope_diagnostics(unit_ids, pages, table_ids, report)


def scope_diagnostics(unit_ids, pages, table_ids, report):
    selected = {
        key: [row for row in report.get(key, []) if row["legal_unit_id"] in unit_ids]
        for key in ("unresolved_references", "structure_issues", "omissions")
    }
    # An omitted table of a returned unit is a relevant gap, not delivered data.
    table_ids.update(
        identity for row in selected["omissions"] for identity in row["table_ids"]
    )
    selected.update(
        extraction_issues=[
            row
            for row in report.get("extraction_issues", [])
            if row["page_number"] in pages
        ],
        uncovered_unit_ids=[
            identity
            for identity in report.get("uncovered_unit_ids", [])
            if identity in unit_ids
        ],
        table_issues=[
            identity
            for identity in report.get("table_issues", [])
            if identity in table_ids
        ],
    )
    return selected
