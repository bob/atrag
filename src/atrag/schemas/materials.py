"""Format independent, immutable prepared content; loaders own block identities."""

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveInt,
    model_validator,
)

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonempty = Annotated[str, Field(min_length=1, pattern=r"\S")]


class MaterialValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PDFLocation(MaterialValue):
    kind: Literal["pdf"]
    page: PositiveInt


class HTMLLocation(MaterialValue):
    kind: Literal["html"]
    section: Nonempty | None = None
    anchor: Nonempty


class DOCXLocation(MaterialValue):
    kind: Literal["docx"]
    section: Nonempty | None = None
    paragraph: PositiveInt


Location = Annotated[
    PDFLocation | HTMLLocation | DOCXLocation, Field(discriminator="kind")
]


class MaterialBlock(MaterialValue):
    block_id: Nonempty
    text: Nonempty
    location: Location
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PreparedMaterial(MaterialValue):
    source_id: Nonempty
    source_sha256: SHA256
    preparation_version: Nonempty
    metadata: dict[str, JsonValue]
    blocks: tuple[MaterialBlock, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_blocks(self) -> Self:
        ids = {block.block_id for block in self.blocks}
        if len(ids) != len(self.blocks):
            raise ValueError("Duplicate material block identity")
        return self

    @property
    def snapshot_id(self) -> str:
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
