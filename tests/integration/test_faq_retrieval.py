"""HTML and PDF share publication, ranking, readers and pinned chat evidence."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fixtures.faq_pipeline import card, make_faq_package, make_faq_snapshot
from fixtures.prepared import refresh_manifest
from fixtures.reranking import reranking_response
from sqlalchemy import event

from atrag.chat.api.tools import RagTools
from atrag.chat.services.chat import ChatService
from atrag.corpus import publish_corpus, resolve_corpus
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package, load_package
from atrag.migrations import migrate_database
from atrag.retrieval.reranking import RerankingRequestError
from atrag.retrieval.search import search_tax_documents

pytestmark = pytest.mark.integration


@pytest.fixture
def materials(
    empty_corpus_schema, engine, prepared_document, faq_snapshot, tmp_path, request
):
    migrate_database(engine)
    data, parsed, chunked, embedded = prepared_document
    if vector := getattr(request, "param", {}).get("cirs_vector"):
        path = embedded.parent / "embeddings.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["embedding"] = vector + [0.0] * 1534
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        refresh_manifest(embedded)
    cirs = build_package(parsed, chunked, embedded, data, tmp_path / "cirs")
    cirs_id = index_package(cirs, engine, tmp_path / "server").snapshot_id
    before = publish_corpus(engine, {"pt-at-cirs": cirs_id})
    faq = make_faq_package(faq_snapshot, tmp_path, model="text-embedding-3-small")
    document = load_package(faq)
    index_package(faq, engine, tmp_path / "server")
    with engine.connect() as connection:
        assert resolve_corpus(connection) == before  # Indexing does not publish.
    return {"pt-at-cirs": cirs_id, "pt-at-faq": document.snapshot.snapshot_id}, document


@pytest.fixture
def session(engine, materials):
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
    conversation = service.create_conversation(user).id
    calls = []

    def create(**request):
        calls.append(request)
        return SimpleNamespace(
            model_dump=lambda: {
                "model": request["model"],
                "data": [{"index": 0, "embedding": [1.0] + [0.0] * 1535}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
        )

    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=create),
        responses=SimpleNamespace(create=reranking_response),
    )

    def start():
        turn = uuid4()
        service.start_turn(user, conversation, turn, "Residência?", "pt")
        return RagTools(engine, user, conversation, turn, client=client)

    return service, user, conversation, calls, start


def invoke(rag, name, **arguments):
    return next(tool for tool in rag.tools if tool.name == name).invoke(arguments)


@pytest.mark.parametrize(
    "query",
    ["Under CIRS 16, am I resident?", "Segundo o artigo 16 do CIRS, sou residente?"],
)
def test_article_mention_searches_both_formats_and_snapshot_path(
    engine, materials, session, query
):
    members, _ = materials
    publish_corpus(engine, {"pt-at-faq": members["pt-at-faq"]})
    _, _, _, calls, start = session
    rag = start()
    result = invoke(rag, "search_tax_documents", query=query)
    assert result["status"] == "ok"
    assert {s["address"]["source_id"] for s in result["sources"]} == set(members)
    assert len(calls) == 1  # One embedding for the whole corpus.
    assert [h["vector_rank"] for h in result["hits"]] == list(range(1, 5))
    assert result["context_tokens_used"] <= 6000
    narrowed = invoke(
        rag, "search_tax_documents", query=query, source_ids=["pt-at-faq"]
    )
    assert {s["address"]["source_id"] for s in narrowed["sources"]} == {"pt-at-faq"}
    single = search_tax_documents(
        query,
        snapshot_id=members["pt-at-faq"],
        engine=engine,
        client=rag.service.client,
        query_language="pt",
    )
    assert single.status == "ok" and single.hits[0].fragment.faq is not None
    assert len(calls) == 3


def test_citation_links_are_server_owned_and_survive_history(
    engine, materials, session
):
    from atrag.chat.services.chat import ChatService

    members, _ = materials
    publish_corpus(engine, {"pt-at-faq": members["pt-at-faq"]})
    service, user, conversation, _, start = session
    rag = start()
    found = invoke(rag, "search_tax_documents", query="residência")
    sources = {s["address"]["source_id"]: s for s in found["sources"]}
    faq = sources["pt-at-faq"]
    assert faq["url"].endswith("/faqs-00267.aspx#heading1")
    assert faq["document_type"] == "faq"
    assert faq["source_format"] == "html_collection"
    assert faq["faq_id"] == "42" and faq["question"] == "Question?"
    assert faq["topic_path"] == ["IRS", "Topic"]
    assert faq["source_url_status"] == "verified"
    law = sources["pt-at-cirs"]
    assert law["document_type"] == "law" and law["source_format"] == "pdf"
    assert law["source_url_status"] == "candidate"
    assert law["url"].endswith(f"/CIRS.pdf#page={law['locations'][0]['page']}")
    for name in ("load_unit_text", "load_context", "load_neighbor_parts"):
        read = invoke(rag, name, address=faq["address"])
        assert all(s["url"] == faq["url"] for s in read["sources"])
    saved = service.load_history(user, conversation)[0].sources
    publish_corpus(engine, {"pt-at-faq": None})
    assert ChatService(engine).load_history(user, conversation)[0].sources == saved


@pytest.mark.parametrize("language", ["en", "pt", "uk", "ru"])
def test_mixed_answer_reopens_with_both_links_in_the_real_app(
    engine, materials, session, monkeypatch, language
):
    import streamlit as st
    from fixtures.chat import ScriptedModel, call
    from langchain_core.messages import AIMessage
    from streamlit.testing.v1 import AppTest

    from atrag.chat.api.execution import LangGraphBackend
    from atrag.chat.services.turns import TurnService
    from atrag.interface import resources

    members, _ = materials
    publish_corpus(engine, {"pt-at-faq": members["pt-at-faq"]})
    service, user, _, _, start = session
    service.set_language(user, language)
    client = start().service.client
    conversation = service.create_conversation(user).id
    claims = {
        "is_logged_in": True,
        "iss": user.issuer,
        "sub": user.subject,
        "email": user.email,
        "email_verified": True,
    }
    monkeypatch.setattr(
        st, "user", SimpleNamespace(get=claims.get, to_dict=lambda: claims)
    )
    monkeypatch.setattr(resources, "get_engine", lambda: engine)

    def messages():
        yield call(args={"query": "residência"})
        saved = service.load_history(user, conversation)[-1].sources
        ids = {s["document_type"]: s["citation_key"] for s in saved}
        yield AIMessage(f"CIRS [{ids['law']}]; FAQ AT [{ids['faq']}].")

    model = ScriptedModel(messages=messages(), disable_streaming=True)
    turns = TurnService(engine, LangGraphBackend(engine, model, client=client))
    monkeypatch.setattr(resources, "get_turn_service", lambda: turns)
    app_path = Path(__file__).resolve().parents[2] / "streamlit_app.py"

    def open_app():
        app = AppTest.from_file(str(app_path), default_timeout=20)
        app.secrets["access"] = {"allowed_emails": [user.email]}
        app.query_params["conversation"] = str(conversation)
        return app.run()

    app = open_app()
    app.chat_input[0].set_value("Residência?").run()
    assert not app.exception
    turn = service.load_history(user, conversation)[0]
    assert turn.status == "completed" and turn.language == language
    assert {s["document_type"] for s in turn.sources} == {"law", "faq"}
    urls = [s["url"] for s in turn.sources]
    assert [b.proto.url for b in app.get("link_button")] == urls
    reopened = open_app()
    assert not reopened.exception
    assert [b.proto.url for b in reopened.get("link_button")] == urls
    assert service.load_history(user, conversation)[0] == turn


@pytest.mark.parametrize("query", ["CIRS 16", "artigo 16 do CIRS"])
def test_bare_address_stays_exact_without_embedding(engine, materials, session, query):
    members, _ = materials
    publish_corpus(engine, {"pt-at-faq": members["pt-at-faq"]})
    _, _, _, calls, start = session
    result = invoke(start(), "search_tax_documents", query=query)
    assert {s["address"]["source_id"] for s in result["sources"]} == {"pt-at-cirs"}
    single = search_tax_documents(
        query, snapshot_id=members["pt-at-cirs"], engine=engine, query_language="pt"
    )
    assert single.status == "ok" and all(h.score == 1 for h in single.hits)
    assert calls == []


@pytest.mark.parametrize(
    "faq_snapshot",
    [
        {
            "cards": "".join(
                card(str(i), answer=f"Detailed explanation {i}.", position=i)
                for i in range(42, 122)
            )
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("materials", [{"cirs_vector": [0.8, 0.6]}], indirect=True)
def test_reranking_recovers_a_provision_beyond_the_old_candidate_window(
    engine, materials, session
):
    members, _ = materials
    publish_corpus(engine, {"pt-at-faq": members["pt-at-faq"]})
    service, user, conversation, calls, start = session
    rag = start()
    submitted = []

    def select_evidence(**request):
        payload = json.loads(request["input"])
        submitted.append(payload)
        response = reranking_response(**request)
        preferred = [
            c["id"] for c in payload["candidates"] if "1 - São residentes" in c["text"]
        ]
        order = preferred + [
            c["id"] for c in payload["candidates"] if c["id"] not in preferred
        ]
        response.output_text = json.dumps(
            {str(identity): 100 if identity in preferred else 0 for identity in order}
        )
        return response

    rag.service.client.responses.create = select_evidence
    result = invoke(rag, "search_tax_documents", query="Residência?", limit=5)
    assert {s["address"]["source_id"] for s in result["sources"]} == set(members)
    position = next(
        i
        for i, s in enumerate(result["sources"])
        if s["legal_unit_id"] == "pt:cirs:code:article:16:paragraph:1"
    )
    assert result["hits"][position]["vector_rank"] > 50
    assert len(submitted[0]["candidates"]) == 83 and len(calls) == 1
    assert (
        service.load_history(user, conversation)[0].embedding_usage["reranking"][
            "requests"
        ]
        == 1
    )
    narrowed = invoke(
        rag, "search_tax_documents", query="Residência?", source_ids=["pt-at-faq"]
    )
    assert {s["address"]["source_id"] for s in narrowed["sources"]} == {"pt-at-faq"}
    assert all(c["role"] == "at_explanation" for c in submitted[1]["candidates"])


def test_invalid_reranking_does_not_save_invented_evidence(engine, materials, session):
    members, _ = materials
    publish_corpus(engine, {"pt-at-faq": members["pt-at-faq"]})
    service, user, conversation, _, start = session
    rag = start()

    def invalid(**request):
        response = reranking_response(**request)
        response.output_text = '{"999": 100, "998": 50}'
        return response

    rag.service.client.responses.create = invalid
    with pytest.raises(RerankingRequestError):
        invoke(rag, "search_tax_documents", query="Residência?", limit=1)
    saved = service.load_history(user, conversation)[0]
    assert saved.sources == [] and saved.retrieval_tokens == 0


@pytest.mark.parametrize(
    "faq_snapshot",
    [
        {
            "answer": "<p>Only if resident under CIRS 16:</p>"
            + "<p>Complete condition.</p>" * 300
        }
    ],
    indirect=True,
)
def test_faq_readers_preserve_full_answer_and_do_not_invent_legal_edges(
    engine, materials, session
):
    members, document = materials
    publish_corpus(engine, {"pt-at-faq": members["pt-at-faq"]})
    _, _, _, _, start = session
    rag = start()
    result = invoke(
        rag,
        "search_tax_documents",
        query="residência",
        source_ids=["pt-at-faq"],
        limit=1,
    )
    source = result["sources"][0]
    assert source["locations"][0]["kind"] == "html"
    assert source["available_reference_ids"] == source["available_note_ids"] == []
    entry = next(u for u in document.units if u.faq)
    assert len(document.fragments) > 1
    result = invoke(rag, "load_unit_text", address=source["address"])
    assert (
        result["complete"] and result["sources"][0]["text_original"] == entry.faq.answer
    )
    neighbors = invoke(
        rag, "load_neighbor_parts", address=source["address"], before=50, after=50
    )
    assert neighbors["complete"]
    parts = sorted(neighbors["sources"], key=lambda s: s["source_start"])
    assert "".join(s["text_original"] for s in parts) == entry.faq.answer
    context = invoke(rag, "load_context", address=source["address"])
    assert context["complete"] and context["tables"] == []
    assert len(context["sources"]) == 1
    assert context["sources"][0]["text_original"] == entry.faq.answer
    invalid = invoke(
        rag,
        "load_context",
        address=source["address"],
        reference_ids=["pt:cirs:code:article:16"],
    )
    assert invalid["status"] == "invalid_arguments"


def test_faq_update_rollback_and_restart_keep_old_turn_and_citations(
    engine,
    materials,
    session,
    tmp_path,
    database_dsn,
    monkeypatch,
    capsys,
):
    from atrag.cli import main

    members, _ = materials
    monkeypatch.setenv("DATABASE_URL", database_dsn)
    changes = tmp_path / "changes.json"
    changes.write_text(json.dumps({"pt-at-faq": members["pt-at-faq"]}))
    main(["corpus", "publish", "--changes", str(changes)])
    assert json.loads(capsys.readouterr().out)["members"] == members
    service, user, conversation, _, start = session
    old = start()
    result = invoke(
        old, "search_tax_documents", query="residência", source_ids=["pt-at-faq"]
    )
    saved = result["sources"][0]
    revision = tmp_path / "revision"
    revision.mkdir()
    revised = make_faq_snapshot(
        revision, {"cards": card(answer="New conditions under CIRS 16.", position=7)}
    )
    newer = make_faq_package(
        revised,
        revision,
        model="text-embedding-3-small",
    )
    next_id = index_package(newer, engine, tmp_path / "server").snapshot_id
    assert next_id != members["pt-at-faq"]
    updated = publish_corpus(engine, {"pt-at-faq": next_id})
    assert dict(updated.members) == {**members, "pt-at-faq": next_id}
    # Rebuilding the tool service and retrying a turn preserve its original release.
    service.start_turn(user, conversation, old.turn_id, "Residência?", "pt")
    rebuilt = RagTools(
        engine, user, conversation, old.turn_id, client=old.service.client
    )
    again = invoke(
        rebuilt, "search_tax_documents", query="residência", source_ids=["pt-at-faq"]
    )
    assert again["sources"] == result["sources"]
    new = start()
    fresh = invoke(
        new, "search_tax_documents", query="residência", source_ids=["pt-at-faq"]
    )
    assert fresh["sources"][0]["address"]["snapshot_id"] == next_id
    assert fresh["sources"][0]["text_original"] == "New conditions under CIRS 16."
    assert fresh["sources"][0]["locations"] != saved["locations"]
    assert fresh["sources"][0]["url"].endswith("#heading7")
    assert saved["url"].endswith("#heading1")
    restored = publish_corpus(engine, {"pt-at-faq": members["pt-at-faq"]})
    assert dict(restored.members) == members
    historical = invoke(
        new,
        "load_unit_text",
        address={
            "kind": "citation",
            "turn_id": str(old.turn_id),
            "citation_id": saved["citation_id"],
        },
    )
    assert historical["sources"][0]["text_original"] == saved["text_original"]
    assert historical["sources"][0]["address"]["snapshot_id"] == members["pt-at-faq"]
    assert service.load_history(user, conversation)[0].sources == result["sources"]

    def fail_pointer(connection, cursor, statement, parameters, context, many):
        if statement.startswith("UPDATE corpus_state"):
            with engine.connect() as reader:
                assert resolve_corpus(reader) == restored
            raise RuntimeError("injected publication failure")

    event.listen(engine, "before_cursor_execute", fail_pointer)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            publish_corpus(engine, {"pt-at-faq": next_id})
    finally:
        event.remove(engine, "before_cursor_execute", fail_pointer)
    with engine.connect() as connection:
        assert resolve_corpus(connection) == restored
