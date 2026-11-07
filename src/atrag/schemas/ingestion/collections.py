"""Complete HTML source collections; observations do not define byte identity."""

import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import AwareDatetime, Field, PositiveInt, field_validator, model_validator

from atrag.schemas.materials import SHA256, MaterialValue, Nonempty, PageURL


class HTMLPage(MaterialValue):
    url: PageURL
    local_path: Nonempty
    sha256: SHA256
    size_bytes: PositiveInt
    final_url: PageURL
    fetched_at: AwareDatetime
    http_status: Literal[200]
    headers: dict[str, str] = Field(default_factory=dict)
    navigation_urls: tuple[PageURL, ...] = ()

    @field_validator("local_path")
    @classmethod
    def relative_html_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or path.as_posix() != value
            or path.suffix != ".html"
        ):
            raise ValueError("Collection page requires a relative HTML file path")
        return value


class HTMLCollection(MaterialValue):
    schema_version: Literal[1] = 1
    catalog_url: PageURL
    pages: tuple[HTMLPage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_navigation(self) -> Self:
        urls = {page.url for page in self.pages}
        paths = {page.local_path for page in self.pages}
        if len(urls) != len(self.pages) or len(paths) != len(self.pages):
            raise ValueError("Collection URLs and file paths must be unique")
        if self.catalog_url not in urls:
            raise ValueError("Collection must include its catalog")
        targets = {url for page in self.pages for url in page.navigation_urls}
        if not targets <= urls:
            raise ValueError("Collection is missing required navigation pages")
        return self

    @property
    def source_sha256(self) -> str:
        content = sorted(
            ({"url": str(page.url), "sha256": page.sha256} for page in self.pages),
            key=lambda page: page["url"],
        )
        raw = json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def size_bytes(self) -> int:
        return sum(page.size_bytes for page in self.pages)
