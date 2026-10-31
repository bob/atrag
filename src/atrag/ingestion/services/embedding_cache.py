"""Durable PostgreSQL vectors with one paid-request writer per model space."""

import hashlib
from contextlib import contextmanager
from itertools import batched

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, select, text
from sqlalchemy.dialects.postgresql import insert

from atrag.schemas.ingestion.embeddings import EmbeddingRecord

EMBEDDINGS = Table(
    "embeddings",
    MetaData(),
    Column("model", Text, primary_key=True),
    Column("dimensions", Integer, primary_key=True),
    Column("input_hash", String(64), primary_key=True),
    Column("embedding", Vector(), nullable=False),
)


class EmbeddingStore:
    def __init__(self, connection, space):
        self.connection = connection
        self.space = space

    def get(self, hashes):
        result = {}
        for batch in batched(hashes, 1000):
            query = select(EMBEDDINGS).where(
                EMBEDDINGS.c.model == self.space.model,
                EMBEDDINGS.c.dimensions == self.space.dimensions,
                EMBEDDINGS.c.input_hash.in_(batch),
            )
            with self.connection.begin():
                for row in self.connection.execute(query).mappings():
                    record = EmbeddingRecord.model_validate(dict(row))
                    result[record.input_hash] = record
        return result

    def save(self, records):
        if any(
            (r.model, r.dimensions) != (self.space.model, self.space.dimensions)
            for r in records
        ):
            raise ValueError("Embedding records belong to another space")
        with self.connection.begin():
            self.connection.execute(
                insert(EMBEDDINGS).on_conflict_do_nothing(),
                [r.model_dump() for r in records],
            )


class PostgresEmbeddingCache:
    def __init__(self, engine):
        self.engine = engine

    @contextmanager
    def writer(self, space):
        identity = f"atrag:embeddings:{space.model}:{space.dimensions}".encode()
        key = int.from_bytes(hashlib.sha256(identity).digest()[:8], signed=True)
        with self.engine.connect() as connection:
            connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
            connection.commit()
            try:
                yield EmbeddingStore(connection, space)
            finally:
                connection.rollback()
                connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": key}
                )
                connection.commit()
