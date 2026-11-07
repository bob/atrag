"""Format independent, immutable prepared content; loaders own block identities."""

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    PositiveInt,
    model_validator,
)

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonempty = Annotated[str, Field(min_length=1, pattern=r"\S")]


def _without_fragment(url: HttpUrl) -> HttpUrl:
    if url.fragment is not None:
        raise ValueError("Page URL must not include an anchor")
    return url


PageURL = Annotated[HttpUrl, AfterValidator(_without_fragment)]


class MaterialValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PDFLocation(MaterialValue):
    kind: Literal["pdf"]
    page: PositiveInt


class HTMLLocation(MaterialValue):
    kind: Literal["html"]
    section: Nonempty | None = None
    anchor: Nonempty
    page_url: PageURL | None = Field(default=None, exclude_if=lambda v: v is None)


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
