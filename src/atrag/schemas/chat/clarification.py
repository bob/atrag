"""A clarification asks for missing input without asserting a tax conclusion."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)
]


class Clarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: Annotated[list[Question], Field(min_length=1, max_length=3)]
