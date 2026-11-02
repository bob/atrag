"""Four LangChain tools bound to one authenticated, persisted chat turn."""

from functools import partial

from langchain_core.tools import StructuredTool
from sqlalchemy import select, update

from atrag import corpus
from atrag.chat.rag_models import (
    BlockAddress,
    CitationAddress,
    ContextArguments,
    NeighborArguments,
    SearchArguments,
    UnitArguments,
)
from atrag.chat.service import _conversation, _required
from atrag.chat.sources import register_sources, source_record
from atrag.db_models import ChatTurnRow, FragmentRow
from atrag.ingestion.api.embeddings import EmbeddingRequestError
from atrag.retrieval.corpus import search_corpus
from atrag.retrieval.text import load_context, load_neighbor_parts, load_unit_text
from atrag.schemas.retrieval import (
    ContextRequest,
    SearchRequest,
    _compact_pdf_locations,
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


def _empty(status, reason):
    return {"status": status, "reason": reason, "sources": [], "context_tokens_used": 0}


def _new_usage():
    return {
        "requests": 0,
        "retries": 0,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def _add_usage(previous, current):
    return {
        "requests": previous.get("requests", 0) + current["requests"],
        "retries": previous.get("retries", 0) + current["retries"],
        "usage": {
            key: previous.get("usage", {}).get(key, 0) + value
            for key, value in current["usage"].items()
        },
    }


class RagTools:
    """Construct on the server with a User from ChatService.authenticate().

    All mutable evidence and budget state lives in the locked database row,
    so concurrent tools and reconstructed instances share the same limits.
    """

    def __init__(self, engine, user, conversation_id, turn_id, *, client=None):
        self.engine, self.user = engine, user
        self.conversation_id, self.turn_id = conversation_id, turn_id
        self.client = client
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
        args = schema.model_validate(values)
        with self.engine.begin() as connection:
            _required(
                connection.execute(_conversation(self.user, self.conversation_id))
            )
            turn = _required(
                connection.execute(
                    select(ChatTurnRow)
                    .where(
                        ChatTurnRow.id == self.turn_id,
                        ChatTurnRow.conversation_id == self.conversation_id,
                    )
                    .with_for_update()
                )
            )
            remaining = 24000 - turn["retrieval_tokens"]
            if remaining == 0:
                return _empty("budget_exhausted", "turn_budget_exhausted")
            args = args.model_copy(
                update={"context_tokens": min(args.context_tokens, remaining)}
            )
            usage = _new_usage()
            execute = {
                "search_tax_documents": self._search,
                "load_unit_text": self._unit,
                "load_neighbor_parts": self._neighbors,
                "load_context": self._context,
            }[name]
            try:
                result, records = execute(connection, turn, args, usage)
            except EmbeddingRequestError:
                raise
            except ValueError as error:
                return _empty("invalid_arguments", str(error))
            registry = list(turn["sources"])
            result["sources"] = register_sources(registry, records)
            connection.execute(
                update(ChatTurnRow)
                .where(ChatTurnRow.id == self.turn_id)
                .values(
                    sources=registry,
                    retrieval_tokens=turn["retrieval_tokens"]
                    + result["context_tokens_used"],
                    embedding_usage=_add_usage(turn["embedding_usage"], usage),
                )
            )
            return result

    def _search(self, connection, turn, args, usage):
        members = dict(corpus.resolve_turn_corpus(connection, self.turn_id).members)
        selected = set(members) if args.source_ids is None else set(args.source_ids)
        if not selected <= members.keys():
            raise ValueError("Source filter is outside the turn's corpus")
        snapshots = [members[source] for source in sorted(selected)]
        if not snapshots:
            return _empty("no_match", "empty_corpus"), []
        request = SearchRequest(
            query=args.query,
            limit=args.limit,
            snapshot_id=snapshots[0],
            query_language=turn["language"],
            filters=args.filters,
            context_tokens=args.context_tokens,
        )
        hits, used, limited = search_corpus(
            request, snapshots, engine=self.engine, client=self.client, usage=usage
        )
        records = [self._fragment_record(connection, hit.fragment) for hit in hits]
        return {
            "status": "ok" if hits else "budget_exhausted" if limited else "no_match",
            "reason": "context_budget_exceeded" if limited else None,
            "context_tokens_used": used,
            "hits": [h.model_dump(include={"score", "vector_rank"}) for h in hits],
            "limited": limited,
        }, records

    def _address(self, connection, argument):
        if isinstance(argument, CitationAddress):
            sources = connection.scalar(
                select(ChatTurnRow.sources).where(
                    ChatTurnRow.id == argument.turn_id,
                    ChatTurnRow.conversation_id == self.conversation_id,
                )
            )
            for source in sources or []:
                if source.get("citation_id") == argument.citation_id:
                    return BlockAddress.model_validate(source["address"])
            raise ValueError("Citation is absent from this conversation")
        members = corpus.resolve_turn_corpus(connection, self.turn_id).members
        if (argument.source_id, argument.snapshot_id) not in members:
            raise ValueError("Address is outside the turn's corpus")
        return argument

    def _unit_address(self, connection, argument):
        address = self._address(connection, argument)
        if address.kind == "fragment":
            unit_id = connection.scalar(
                select(FragmentRow.legal_unit_id).where(
                    FragmentRow.snapshot_id == address.snapshot_id,
                    FragmentRow.fragment_id == address.block_id,
                )
            )
            if unit_id is None:
                raise ValueError("Fragment is absent from the selected snapshot")
            return address.model_copy(update={"kind": "unit", "block_id": unit_id})
        return address

    def _fragment_record(self, connection, part):
        address = BlockAddress(
            kind="fragment",
            source_id=part.source.source_id,
            snapshot_id=part.snapshot_id,
            block_id=part.fragment_id,
        )
        return source_record(
            connection,
            address,
            part,
            complete=part.part_count == 1,
            start=part.source_start,
            end=part.source_end,
        )

    def _text_record(self, connection, address, text):
        address = address.model_copy(update={"block_id": text.legal_unit_id})
        return source_record(
            connection,
            address,
            text,
            complete=text.complete,
            start=text.source_start,
            end=text.source_end,
        )

    def _unit(self, connection, turn, args, usage):
        address = self._unit_address(connection, args.address)
        try:
            text = load_unit_text(
                address.block_id,
                snapshot_id=address.snapshot_id,
                engine=self.engine,
                start=args.start,
                context_tokens=args.context_tokens,
            )
        except ValueError as error:
            if str(error) == "Envelope leaves no room for a source character":
                return _empty("budget_exhausted", "context_budget_exceeded"), []
            raise
        return {
            "status": "ok",
            "context_tokens_used": text.context_tokens_used,
            "complete": text.complete,
            "next_start": text.next_start,
        }, [self._text_record(connection, address, text)]

    def _neighbors(self, connection, turn, args, usage):
        address = self._address(connection, args.address)
        if address.kind != "fragment":
            raise ValueError("Neighbor loading requires a fragment address")
        result = load_neighbor_parts(
            address.block_id,
            snapshot_id=address.snapshot_id,
            engine=self.engine,
            before=args.before,
            after=args.after,
            context_tokens=args.context_tokens,
        )
        return {
            "status": "ok" if result.parts else "budget_exhausted",
            "context_tokens_used": result.context_tokens_used,
            "complete": result.complete,
            "omitted_part_indices": result.omitted_part_indices,
        }, [self._fragment_record(connection, p) for p in result.parts]

    def _context(self, connection, turn, args, usage):
        address = self._unit_address(connection, args.address)
        result = load_context(
            ContextRequest(
                snapshot_id=address.snapshot_id,
                legal_unit_id=address.block_id,
                reference_ids=args.reference_ids,
                note_ids=args.note_ids,
                dependency_edges=args.dependency_edges,
                context_tokens=args.context_tokens,
            ),
            engine=self.engine,
        )
        return {
            "status": "ok" if result.units else "budget_exhausted",
            "context_tokens_used": result.context_tokens_used,
            "complete": result.complete,
            "omitted_unit_ids": result.omitted_unit_ids,
            "omitted_table_ids": result.omitted_table_ids,
            "tables": [
                _compact_pdf_locations(t.model_dump(mode="json")) for t in result.tables
            ],
        }, [self._text_record(connection, address, t) for t in result.units]
