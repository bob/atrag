"""A substituted Responses API preserving the submitted candidate order."""

import json
from types import SimpleNamespace


def reranking_response(**request):
    payload = json.loads(request["input"])
    return SimpleNamespace(
        status="completed",
        output_text=json.dumps({"ids": list(range(payload["count"]))}),
        usage=SimpleNamespace(input_tokens=20, output_tokens=5, total_tokens=25),
    )
