"""Published FAQ identity and original content at the HTML parser boundary."""

from pathlib import PurePosixPath
from typing import Annotated, Self
from urllib.parse import urlsplit

from pydantic import Field, HttpUrl, PositiveInt, model_validator

from atrag.schemas.materials import MaterialValue, Nonempty, PageURL


class FAQLink(MaterialValue):
    text: str
    href: Nonempty
    url: HttpUrl


class FAQEntry(MaterialValue):
    entry_id: Nonempty
    faq_id: Annotated[str, Field(pattern=r"^[0-9]+$")]
    question: Nonempty
    answer: Nonempty
    topic_path: tuple[Nonempty, ...] = Field(min_length=1)
    page_url: PageURL
    anchor: Nonempty
    links: tuple[FAQLink, ...] = ()
    ordinal: PositiveInt

    @model_validator(mode="after")
    def published_identity(self) -> Self:
        page_key = PurePosixPath(urlsplit(str(self.page_url)).path).stem
        if not page_key or self.entry_id != f"{page_key}:{self.faq_id}":
            raise ValueError(
                "FAQ identity must combine its page key and published number"
            )
        return self
