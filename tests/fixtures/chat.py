"""A deterministic tool-capable model; no OpenAI transport is involved."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import Field


class ScriptedModel(GenericFakeChatModel):
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def call(name="search_tax_documents", args=None, identifier="call-1"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args if args is not None else {"query": "CIRS 16"},
                "id": identifier,
            }
        ],
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
