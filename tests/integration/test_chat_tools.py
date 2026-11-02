"""The model sees only bounded evidence from an owned turn and saved citations."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fixtures.prepared import prepare_document, prepare_revision
from sqlalchemy import select, update

from atrag.chat.api.tools import RagTools
from atrag.chat.services.chat import ChatService
from atrag.corpus import publish_corpus
from atrag.db_models import ChatTurnRow, FragmentRow, LegalUnitRow, SnapshotRow
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package
from atrag.migrations import migrate_database

pytestmark = pytest.mark.integration


@pytest.fixture
def chat(empty_corpus_schema, engine):
    migrate_database(engine)
    service = ChatService(engine)
    user = service.authenticate(
        {
            "iss": "accounts.google.com",
            "sub": "owner",
            "email": "a@b",
            "email_verified": True,
        },
        ["a@b"],
    )
    conversation = service.create_conversation(user)
    return service, user, conversation.id


@pytest.fixture
def materials(chat, engine, tmp_path):
    versions = {}
    for source in ("first", "second"):
        prepared = prepare_document(
            tmp_path / source,
            source_id=source,
            second_paragraph=None
            if source == "first"
            else "2 - Exceção própria do segundo documento.",
        )
        package = build_package(
            *prepared[1:], prepared[0], tmp_path / (source + "-package")
        )
        versions[source] = index_package(
            package, engine, tmp_path / "server"
        ).snapshot_id
    with engine.begin() as connection:
        reports = connection.scalar(
            select(SnapshotRow.reports).where(
                SnapshotRow.snapshot_id == versions["second"]
            )
        )
        reports["chunk"]["extraction_issues"] = []
        connection.execute(
            update(SnapshotRow)
            .where(SnapshotRow.snapshot_id == versions["second"])
            .values(reports=reports)
        )
        for row in connection.execute(
            select(FragmentRow).where(
                FragmentRow.snapshot_id == versions["first"],
                FragmentRow.paragraph == "1",
            )
        ).mappings():
            payload = dict(
                row["payload"],
                issues=["inherited_context_issue"],
                editorial_notes=[
                    {
                        "kind": "amendment",
                        "raw_text": "Nota do contexto original.",
                        "locators": row["payload"]["locators"],
                    }
                ],
            )
            connection.execute(
                update(FragmentRow)
                .where(
                    FragmentRow.snapshot_id == versions["first"],
                    FragmentRow.fragment_id == row["fragment_id"],
                )
                .values(payload=payload)
            )
    publish_corpus(engine, {source: snapshot for source, snapshot in versions.items()})
    return versions


def start(chat, engine, client=None):
    service, user, conversation = chat
    turn = uuid4()
    service.start_turn(user, conversation, turn, "Residência?", "pt")
    return RagTools(engine, user, conversation, turn, client=client)


def invoke(rag, name, **arguments):
    return next(tool for tool in rag.tools if tool.name == name).invoke(arguments)


def test_global_search_and_all_readers_keep_document_and_citations(
    chat, materials, engine
):
    rag = start(chat, engine)
    result = invoke(rag, "search_tax_documents", query="CIRS 16", limit=4)
    assert result["status"] == "ok"
    assert len(result["sources"]) == 4
    assert {s["address"]["source_id"] for s in result["sources"]} == {"first", "second"}
    assert [s["citation_id"] for s in result["sources"]] == [1, 2, 3, 4]
    assert [hit["score"] for hit in result["hits"]] == [1.0, 1.0, 1.0, 1.0]
    assert all(s["complete"] for s in result["sources"])
    source = result["sources"][0]
    assert source["text_original"].startswith("1 - São residentes")
    assert source["locations"] == [{"kind": "pdf", "page": 1}]
    for item in result["sources"]:
        assert bool(item["diagnostics"]["extraction_issues"]) == (
            item["address"]["source_id"] == "first"
        )
        assert item["source_length"] == len(item["text_original"])
        if item["address"]["source_id"] == "first" and item["text_original"].startswith(
            "1 -"
        ):
            assert item["issues"] == ["inherited_context_issue"]
            assert (
                item["editorial_notes"][0]["raw_text"] == "Nota do contexto original."
            )
    assert source["consolidation_label"]
    assert source["revision_publication_date"] == "2026-06-03"
    assert source["available_reference_ids"] == ["pt:cirs:code:article:68-A"]
    repeated = invoke(rag, "search_tax_documents", query="CIRS 16", limit=4)
    assert repeated["sources"] == result["sources"]
    text = invoke(rag, "load_unit_text", address=source["address"])
    assert text["sources"][0]["text_original"] == source["text_original"]
    neighbors = invoke(rag, "load_neighbor_parts", address=source["address"])
    assert neighbors["sources"][0]["citation_id"] == source["citation_id"]
    context = invoke(
        rag,
        "load_context",
        address=source["address"],
        reference_ids=source["available_reference_ids"],
    )
    assert context["status"] == "ok"
    assert len(context["sources"]) == 2 and context["tables"][0]["rows"] == [
        ["1000", "2,5"]
    ]
    assert all(
        s["address"]["source_id"] == source["address"]["source_id"]
        for s in context["sources"]
    )
    saved = chat[0].load_history(chat[1], chat[2])[0]
    assert saved.sources and saved.retrieval_tokens > 0


@pytest.mark.parametrize("retry", [False, True])
def test_semantic_search_embeds_once_and_uses_global_top_k(
    chat, materials, engine, monkeypatch, retry, tmp_path
):
    prepared = prepare_document(tmp_path / "unpublished", source_id="unpublished")
    package = build_package(
        *prepared[1:], prepared[0], tmp_path / "unpublished-package"
    )
    index_package(package, engine, tmp_path / "server")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if retry and len(calls) % 2 == 1:
            import httpx
            from openai import APIConnectionError

            raise APIConnectionError(
                request=httpx.Request("POST", "https://example.com")
            )
        raw = {
            "model": kwargs["model"],
            "data": [{"index": 0, "embedding": [0.6, 0.8] + [0.0] * 1534}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
        return SimpleNamespace(model_dump=lambda: raw)

    monkeypatch.setattr("atrag.retrieval.search.time.sleep", lambda _: None)
    rag = start(
        chat, engine, SimpleNamespace(embeddings=SimpleNamespace(create=create))
    )
    result = invoke(rag, "search_tax_documents", query="residência", limit=2)
    assert len(calls) == (2 if retry else 1)
    assert [h["score"] for h in result["hits"]] == pytest.approx([0.8, 0.8])
    assert [h["vector_rank"] for h in result["hits"]] == [1, 2]
    assert {s["address"]["source_id"] for s in result["sources"]} == {"first", "second"}
    expected = {
        "requests": 2 if retry else 1,
        "retries": int(retry),
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    assert chat[0].load_history(chat[1], chat[2])[0].embedding_usage == expected
    narrowed = invoke(
        rag, "search_tax_documents", query="residência", source_ids=["first"], limit=1
    )
    assert {s["address"]["source_id"] for s in narrowed["sources"]} == {"first"}
    assert chat[0].load_history(chat[1], chat[2])[0].embedding_usage == {
        "requests": expected["requests"] * 2,
        "retries": expected["retries"] * 2,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


def test_old_citations_keep_the_original_version_after_publication(
    chat, materials, engine, tmp_path
):
    old = start(chat, engine)
    source = invoke(
        old, "search_tax_documents", query="CIRS 16", source_ids=["first"], limit=1
    )["sources"][0]
    prepared = prepare_document(tmp_path / "revision-input", source_id="first")
    package = prepare_revision(
        prepared,
        tmp_path / "revision",
        source_updates={"consolidation_label": "Nova redação"},
    )
    new = index_package(package, engine, tmp_path / "server").snapshot_id
    publish_corpus(engine, {"first": new, "second": None})
    again = invoke(old, "load_unit_text", address=source["address"])
    assert again["sources"][0]["address"]["snapshot_id"] == materials["first"]
    current = start(chat, engine)
    citation = {
        "kind": "citation",
        "turn_id": str(old.turn_id),
        "citation_id": source["citation_id"],
    }
    restored = invoke(current, "load_unit_text", address=citation)
    assert restored["sources"][0]["address"]["snapshot_id"] == materials["first"]
    # A direct old-version address cannot bypass the current release.
    assert (
        invoke(current, "load_unit_text", address=source["address"])["status"]
        == "invalid_arguments"
    )
    found = invoke(current, "search_tax_documents", query="CIRS 16")
    assert {s["address"]["snapshot_id"] for s in found["sources"]} == {new}
    assert all(s["consolidation_label"] == "Nova redação" for s in found["sources"])


def test_budget_is_shared_by_tools_instances_and_concurrent_calls(
    chat, materials, engine
):
    rag = start(chat, engine)
    first = invoke(rag, "search_tax_documents", query="CIRS 16", limit=4)
    cost = first["context_tokens_used"]
    assert 0 < cost <= 6000
    with engine.begin() as connection:
        connection.execute(
            update(ChatTurnRow)
            .where(ChatTurnRow.id == rag.turn_id)
            .values(retrieval_tokens=24000 - cost)
        )
    other = RagTools(engine, chat[1], chat[2], rag.turn_id)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(
                lambda r: invoke(r, "search_tax_documents", query="CIRS 16", limit=4),
                [rag, other],
            )
        )
    assert sorted(r["status"] for r in results) == ["budget_exhausted", "ok"]
    assert chat[0].load_history(chat[1], chat[2])[0].retrieval_tokens == 24000
    assert (
        invoke(other, "load_unit_text", address=first["sources"][0]["address"])[
            "status"
        ]
        == "budget_exhausted"
    )


def test_empty_search_and_exhaustion_do_not_invent_sources(chat, engine, materials):
    rag = start(chat, engine)
    for query in ("CIRS 999", "CIVA 53"):
        result = invoke(rag, "search_tax_documents", query=query)
        assert result["status"] == "no_match" and not result["sources"]
    assert (
        invoke(rag, "search_tax_documents", query="CIRS 16", source_ids=[])["reason"]
        == "empty_corpus"
    )
    for name, arguments in (
        ("search_tax_documents", {"query": "CIRS 16"}),
        (
            "load_unit_text",
            {
                "address": {
                    "kind": "unit",
                    "source_id": "first",
                    "snapshot_id": materials["first"],
                    "block_id": "pt:cirs:code:article:16",
                }
            },
        ),
        (
            "load_context",
            {
                "address": {
                    "kind": "unit",
                    "source_id": "first",
                    "snapshot_id": materials["first"],
                    "block_id": "pt:cirs:code:article:16",
                }
            },
        ),
    ):
        result = invoke(rag, name, **arguments, context_tokens=1)
        assert result["status"] == "budget_exhausted"
        assert result["context_tokens_used"] == 0 and not result["sources"]
    assert chat[0].load_history(chat[1], chat[2])[0].sources == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "x", "limit": 0},
        {"query": "x", "limit": True},
        {"query": "x", "context_tokens": 6001},
        {"query": "x", "context_tokens": 0},
        {"query": "x", "user_id": "intruder"},
        {"query": "x", "snapshot_id": "a" * 64},
        {"query": "x", "filters": {"source_id": "intruder"}},
    ],
)
def test_model_schema_rejects_invalid_values_and_server_context(
    chat, engine, arguments
):
    rag = start(chat, engine)
    result = json.loads(invoke(rag, "search_tax_documents", **arguments))
    assert result["status"] == "invalid_arguments"
    assert chat[0].load_history(chat[1], chat[2])[0].retrieval_tokens == 0


def test_outside_sources_addresses_and_references_are_rejected(chat, engine, materials):
    rag = start(chat, engine)
    source = invoke(rag, "search_tax_documents", query="CIRS 16", limit=1)["sources"][0]
    for arguments in ({"source_ids": ["outside"]}, {"filters": {"code": "CIVA"}}):
        assert (
            invoke(rag, "search_tax_documents", query="CIRS 16", **arguments)["status"]
            == "invalid_arguments"
        )
    for address in (
        {**source["address"], "source_id": "outside"},
        {**source["address"], "snapshot_id": "a" * 64},
        {**source["address"], "block_id": "absent"},
        {"kind": "citation", "turn_id": str(rag.turn_id), "citation_id": 999},
    ):
        assert (
            invoke(rag, "load_unit_text", address=address)["status"]
            == "invalid_arguments"
        )
    unit_address = {
        **source["address"],
        "kind": "unit",
        "block_id": source["legal_unit_id"],
    }
    assert (
        invoke(rag, "load_neighbor_parts", address=unit_address)["status"]
        == "invalid_arguments"
    )
    assert (
        invoke(rag, "load_unit_text", address=unit_address, start=999999)["status"]
        == "invalid_arguments"
    )
    assert (
        invoke(rag, "load_context", address=unit_address, note_ids=["invented"])[
            "status"
        ]
        == "invalid_arguments"
    )


def test_tools_recheck_owner_and_history_citations_never_cross_conversations(
    chat, engine, materials
):
    service, owner, conversation = chat
    rag = start(chat, engine)
    source = invoke(rag, "search_tax_documents", query="CIRS 16", limit=1)["sources"][0]
    stranger = service.authenticate(
        {
            "iss": owner.issuer,
            "sub": "stranger",
            "email": owner.email,
            "email_verified": True,
        },
        [owner.email],
    )
    for user in (
        stranger,
        replace(owner, subject="forged"),
        replace(owner, issuer="forged"),
    ):
        denied = RagTools(engine, user, conversation, rag.turn_id)
        with pytest.raises(PermissionError):
            invoke(denied, "search_tax_documents", query="CIRS 16")
    other_conversation = service.create_conversation(owner)
    other_turn = uuid4()
    service.start_turn(owner, other_conversation.id, other_turn, "Other?", "en")
    other = RagTools(engine, owner, other_conversation.id, other_turn)
    citation = {
        "kind": "citation",
        "turn_id": str(rag.turn_id),
        "citation_id": source["citation_id"],
    }
    assert (
        invoke(other, "load_unit_text", address=citation)["status"]
        == "invalid_arguments"
    )
    with pytest.raises(PermissionError):
        invoke(
            RagTools(engine, owner, conversation, other_turn),
            "search_tax_documents",
            query="CIRS 16",
        )


@pytest.mark.parametrize(
    "location",
    [
        {"kind": "html", "section": "Residency", "anchor": "resident"},
        {"kind": "docx", "section": "Residency", "paragraph": 3},
    ],
)
def test_non_pdf_material_passes_search_reading_and_citation(
    chat, engine, tmp_path, location
):
    prepared = prepare_document(tmp_path / "nonpdf", source_id="html-source")
    package = build_package(*prepared[1:], prepared[0], tmp_path / "package")
    snapshot = index_package(package, engine, tmp_path / "server").snapshot_id
    # Simulate a future loader using the same located-block storage contract.
    with engine.begin() as connection:
        for table in (LegalUnitRow, FragmentRow):
            for row in connection.execute(
                select(table).where(table.snapshot_id == snapshot)
            ).mappings():
                payload = dict(
                    row["payload"],
                    page_start=None,
                    page_end=None,
                    locators=[],
                    printed_page_labels=[],
                    location=location,
                )
                key = (
                    table.legal_unit_id if table is LegalUnitRow else table.fragment_id
                )
                connection.execute(
                    update(table)
                    .where(table.snapshot_id == snapshot, key == row[key.key])
                    .values(page_start=None, page_end=None, payload=payload)
                )
    publish_corpus(engine, {"html-source": snapshot})
    rag = start(chat, engine)
    found = invoke(rag, "search_tax_documents", query="CIRS 16", limit=1)["sources"][0]
    assert found["locations"] == [location]
    assert "page" not in found["locations"][0]
    for tool in ("load_unit_text", "load_neighbor_parts", "load_context"):
        result = invoke(rag, tool, address=found["address"])
        assert result["status"] == "ok"
        assert result["sources"][0]["locations"] == [location]
    assert chat[0].load_history(chat[1], chat[2])[0].sources[0]["locations"] == [
        location
    ]


def test_embedding_failure_is_operational_and_does_not_create_sources(
    chat, engine, materials
):
    from atrag.ingestion.api.embeddings import EmbeddingRequestError

    rag = start(chat, engine)
    with pytest.raises(EmbeddingRequestError, match="OPENAI_API_KEY"):
        invoke(rag, "search_tax_documents", query="residência")
    saved = chat[0].load_history(chat[1], chat[2])[0]
    assert saved.sources == [] and saved.retrieval_tokens == 0


def test_split_text_preserves_notes_issues_and_shared_budget(chat, engine, tmp_path):
    prepared = prepare_document(
        tmp_path / "split",
        second_paragraph="2 - " + "Condição fiscal sem isenção. " * 700,
    )
    package = build_package(*prepared[1:], prepared[0], tmp_path / "package")
    snapshot = index_package(package, engine, tmp_path / "server").snapshot_id
    with engine.begin() as connection:
        for table in (LegalUnitRow, FragmentRow):
            for row in connection.execute(
                select(table).where(
                    table.snapshot_id == snapshot, table.paragraph == "2"
                )
            ).mappings():
                payload = dict(
                    row["payload"],
                    issues=["incomplete_source"],
                    editorial_notes=[
                        {
                            "kind": "amendment",
                            "raw_text": "Redação transitória. " * 150,
                            "locators": row["payload"]["locators"],
                        }
                    ],
                )
                key = (
                    table.legal_unit_id if table is LegalUnitRow else table.fragment_id
                )
                connection.execute(
                    update(table)
                    .where(table.snapshot_id == snapshot, key == row[key.key])
                    .values(payload=payload)
                )
    publish_corpus(engine, {"pt-at-cirs": snapshot})
    rag = start(chat, engine)
    found = invoke(
        rag,
        "search_tax_documents",
        query="CIRS 16",
        filters={"paragraph": "2"},
        limit=1,
    )
    part = found["sources"][0]
    assert not part["complete"] and part["issues"] == ["incomplete_source"]
    assert part["editorial_notes"][0]["raw_text"].startswith("Redação transitória.")
    read = invoke(rag, "load_unit_text", address=part["address"], context_tokens=1000)
    assert read["status"] == "ok" and not read["complete"] and read["next_start"] > 0
    assert (
        read["context_tokens_used"] > 750
    )  # Editorial notes consume evidence budget too.
    next_read = invoke(
        rag,
        "load_unit_text",
        address=part["address"],
        start=read["next_start"],
        context_tokens=1000,
    )
    assert next_read["sources"][0]["source_start"] == read["next_start"]
    assert next_read["sources"][0]["citation_id"] != read["sources"][0]["citation_id"]
    neighbors = invoke(
        rag, "load_neighbor_parts", address=part["address"], after=2, context_tokens=1
    )
    assert neighbors["status"] == "budget_exhausted" and neighbors[
        "omitted_part_indices"
    ] == [0, 1, 2]


def test_empty_published_corpus_is_explicit(chat, engine):
    rag = start(chat, engine)
    result = invoke(rag, "search_tax_documents", query="CIRS 16")
    assert result == {
        "turn_id": str(rag.turn_id),
        "status": "no_match",
        "reason": "empty_corpus",
        "sources": [],
        "context_tokens_used": 0,
    }


def test_agent_reads_and_continues_with_original_citations_after_publication(
    chat, materials, engine, database_dsn
):
    from fixtures.chat import ScriptedModel, call
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.postgres import PostgresSaver

    from atrag.chat.api.agent import build_agent

    service, user, conversation = chat
    rag = start(chat, engine)
    config = {"configurable": {"thread_id": str(conversation)}}

    def script():
        yield call(args={"query": "CIRS 16", "source_ids": ["first"], "limit": 1})
        source = rag.service.sources()[0]
        yield call("load_unit_text", {"address": source["address"]}, "unit")
        yield call("load_neighbor_parts", {"address": source["address"]}, "neighbors")
        yield call(
            "load_context",
            {
                "address": source["address"],
                "reference_ids": source["available_reference_ids"],
            },
            "context",
        )
        yield AIMessage("São residentes, com as ressalvas indicadas [1] [2].")

    with PostgresSaver.from_conn_string(database_dsn) as saver:
        graph = build_agent(
            ScriptedModel(messages=script()), rag, "pt", checkpointer=saver
        )
        first = graph.invoke({"messages": [HumanMessage("Residência?")]}, config)
        assert len(first["sources"]) == 2
        assert first["model_usage"]["requests"] == 5
        assert {source["address"]["snapshot_id"] for source in first["sources"]} == {
            materials["first"]
        }
        history_tool = next(
            message for message in first["messages"] if isinstance(message, ToolMessage)
        )
        historic = json.loads(history_tool.content)
        assert historic["turn_id"] == str(rag.turn_id)
    with engine.begin() as connection:
        connection.execute(
            update(ChatTurnRow)
            .where(ChatTurnRow.id == rag.turn_id)
            .values(answer=first["answer"], status="completed")
        )
    publish_corpus(engine, {"first": None})
    next_turn = uuid4()
    service.start_turn(
        user, conversation, next_turn, "Explain the previous citation", "en"
    )
    next_rag = RagTools(engine, user, conversation, next_turn)
    model = ScriptedModel(
        messages=iter(
            [
                call(
                    "load_unit_text",
                    {
                        "address": {
                            "kind": "citation",
                            "turn_id": historic["turn_id"],
                            "citation_id": 1,
                        }
                    },
                    "historical",
                ),
                AIMessage("The previous material says [1]."),
            ]
        )
    )
    with PostgresSaver.from_conn_string(database_dsn) as saver:
        try:
            graph = build_agent(model, next_rag, "en", checkpointer=saver)
            second = graph.invoke(
                {"messages": [HumanMessage("Explain the previous citation")]}, config
            )
            assert second["sources"][0]["address"]["snapshot_id"] == materials["first"]
            assert second["model_usage"] == {
                "requests": 2,
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            }
            assert "Answer in en" in model.seen[0][0].text
            assert any(message.text == first["answer"] for message in model.seen[0])
            assert (
                len(graph.get_state(config).values["messages"])
                == len(first["messages"]) + 4
            )
            assert service.load_history(user, conversation)[0].answer == first["answer"]
        finally:
            saver.delete_thread(str(conversation))


def test_agent_rejects_foreign_turn_before_calling_model(chat, engine):
    from fixtures.chat import ScriptedModel

    from atrag.chat.api.agent import build_agent

    rag = start(chat, engine)
    stranger = chat[0].authenticate(
        {
            "iss": "accounts.google.com",
            "sub": "stranger",
            "email": "a@b",
            "email_verified": True,
        },
        ["a@b"],
    )
    model = ScriptedModel(messages=iter([]))
    with pytest.raises(PermissionError):
        build_agent(model, RagTools(engine, stranger, chat[2], rag.turn_id), "en")
    assert model.seen == []


def test_agent_reports_an_empty_corpus_without_fabricating_sources(chat, engine):
    from fixtures.chat import ScriptedModel, call
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from atrag.chat.api.agent import build_agent

    rag = start(chat, engine)
    model = ScriptedModel(
        messages=iter(
            [
                call(),
                AIMessage("No documents are published. Which source should be added?"),
            ]
        )
    )
    result = build_agent(model, rag, "en").invoke(
        {"messages": [HumanMessage("Tax question")]}
    )
    tool_result = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert json.loads(tool_result.content)["reason"] == "empty_corpus"
    assert result["sources"] == []
    assert result["answer"].startswith("No documents")


def test_agent_can_correct_invalid_tool_arguments(chat, materials, engine):
    from fixtures.chat import ScriptedModel, call
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from atrag.chat.api.agent import build_agent

    model = ScriptedModel(
        messages=iter(
            [
                call(args={"query": ""}),
                call(identifier="corrected"),
                AIMessage("Evidence [1]."),
            ]
        )
    )
    result = build_agent(model, start(chat, engine), "en").invoke(
        {"messages": [HumanMessage("Residência?")]}
    )
    statuses = [
        json.loads(message.content)["status"]
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert statuses == ["invalid_arguments", "ok"]
    assert result["answer"] == "Evidence [1]."
    assert result["model_usage"]["requests"] == 3


def test_remaining_budget_caps_a_partial_read(chat, materials, engine):
    rag = start(chat, engine)
    source = invoke(rag, "search_tax_documents", query="CIRS 16", limit=1)["sources"][0]
    with engine.begin() as connection:
        connection.execute(
            update(ChatTurnRow)
            .where(ChatTurnRow.id == rag.turn_id)
            .values(retrieval_tokens=23960)
        )
    result = invoke(rag, "load_unit_text", address=source["address"])
    assert result["status"] == "ok" and not result["complete"]
    assert 0 < result["context_tokens_used"] <= 40
    assert (
        result["sources"][0]["text_original"]
        == source["text_original"][: result["next_start"]]
    )
    assert chat[0].load_history(chat[1], chat[2])[0].retrieval_tokens <= 24000
