# ATRAG

ATRAG is an AI assistant for Portuguese tax research. It provides conversational
access to a curated document corpus for tax advisers, accountants and client
support teams using retrieval-augmented generation (RAG).

## Capabilities

- Questions and answers in English, Portuguese, Ukrainian and Russian, with
  quotations preserved in the source language.
- Answers with numbered citations, original excerpts, legal references and PDF
  page locations.
- Semantic search and exact article lookup, with retrieval of related provisions,
  tables, notes and surrounding context.
- Google sign-in, configurable email access restrictions and private conversation
  histories, with streamed responses and recovery of interrupted requests.
- Versioned document publication across a shared corpus. Previous answers retain
  references to the document versions used to produce them.
- CLI pipeline for PDF extraction, OCR, legal structure parsing, embedding
  generation with persistent caching, package export/import and indexing.
- Retrieval evaluation against reviewed queries and expected citations.

## Technical implementation

| Area | Technologies and mechanisms |
| --- | --- |
| Agentic RAG | LangChain and LangGraph orchestration; tool calling (function calling) with Pydantic and JSON Schema argument validation. Four tools handle search, legal unit reading, neighboring chunks and linked context. |
| Streaming | OpenAI Responses API with Server-Sent Events (SSE) for model output; LangGraph text deltas and tool execution events; Streamlit WebSockets and `st.write_stream` for browser updates. |
| Vector retrieval | OpenAI embeddings, 1,536-dimensional vectors and pgvector cosine similarity search, with SQL metadata filters, exact legal-reference lookup and context expansion. |
| Document ETL | pdfplumber PDF and table extraction, Tesseract OCR, legal structure parsing, token-aware chunking and preservation of source text locations. |
| Embedding pipeline | Batched embedding generation and a persistent PostgreSQL cache keyed by model, dimensions and input hash; portable JSON/JSONL document packages. |
| Data storage | PostgreSQL and JSONB, SQLAlchemy 2, Psycopg 3 and Alembic migrations; transactional indexing and atomic corpus publication. |
| Durable execution | LangGraph PostgreSQL checkpoints with synchronous persistence, explicit resume, UUID-based turn idempotency and PostgreSQL advisory locks per conversation. |
| Evidence provenance | SHA-256 fingerprints, immutable document snapshots, corpus versions pinned to each turn and validation of citation identifiers against retrieved evidence. |
| Authentication and authorization | Google OpenID Connect (OIDC), signed authentication cookies, configurable email allowlists and user ownership checks for conversations and agent state. |
| Runtime controls | tiktoken token counting, context budgets, history truncation by complete turns, model/tool call limits, retries, timeouts and token usage accounting. |
| Development and delivery | Python 3.14, uv lockfiles, pytest, PostgreSQL integration tests, Ruff and ty; GitHub Actions CI/CD, systemd and Caddy. |

## Current scope

The initial corpus covers CIRS, the Portuguese Personal Income Tax Code. Document
preparation and publication use the CLI; document upload is not available in the
chat interface. Additional document families require dedicated parsers.
Historical legal applicability by tax year is not implemented.
