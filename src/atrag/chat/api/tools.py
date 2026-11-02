"""Expose chat retrieval use cases as four validated LangChain tools."""

from functools import partial

from langchain_core.tools import StructuredTool

from atrag.chat.services.rag import RagService
from atrag.schemas.chat.tools import (
    ContextArguments,
    NeighborArguments,
    SearchArguments,
    UnitArguments,
)

TOOL_SCHEMAS = {
    "search_tax_documents": (
        SearchArguments,
        "Search the turn's tax documents. source_ids and legal filters only narrow the search. Cite returned citation_id values; follow available references and notes.",
    ),
    "load_unit_text": (
        UnitArguments,
        "Read original unit text by document/version/block address or a saved citation. Use start to continue an incomplete result.",
    ),
    "load_neighbor_parts": (
        NeighborArguments,
        "Read adjacent parts of a fragment or saved fragment citation, within the same document and version.",
    ),
    "load_context": (
        ContextArguments,
        "Read a unit and explicitly selected available references/notes. Dependency edges must exist in the source graph.",
    ),
}


class RagTools:
    """Bind model-facing tools to one authenticated, persisted chat turn."""

    def __init__(self, engine, user, conversation_id, turn_id, *, client=None):
        self.turn_id = turn_id
        self.service = RagService(engine, user, conversation_id, turn_id, client=client)
        self.tools = tuple(
            StructuredTool.from_function(
                func=partial(self._run, name),
                name=name,
                description=description,
                args_schema=schema,
                handle_validation_error='{"status":"invalid_arguments","reason":"Invalid tool arguments"}',
            )
            for name, (schema, description) in TOOL_SCHEMAS.items()
        )

    def _run(self, name, **values):
        schema, _ = TOOL_SCHEMAS[name]
        return self.service.run(name, schema.model_validate(values))
