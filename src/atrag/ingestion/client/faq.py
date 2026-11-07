"""Bounded HTTP acquisition; every redirect stays inside the official FAQ scope."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from atrag.ingestion.faq_navigation import normalize_faq_url


@dataclass(frozen=True)
class HTMLResponse:
    raw: bytes
    final_url: str
    fetched_at: datetime
    headers: dict[str, str]


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is None:
        return float(2**attempt)
    value = response.headers.get("retry-after", "")
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            seconds = float(2**attempt)
    return max(0.0, min(seconds, 30.0))


def _request(client: httpx.Client, url: str) -> httpx.Response:
    for _ in range(6):
        response = client.get(url, follow_redirects=False, timeout=30.0)
        if response.status_code not in (301, 302, 303, 307, 308):
            return response
        if "location" not in response.headers:
            raise ValueError("FAQ redirect is missing its destination")
        target = normalize_faq_url(url, response.headers.get("location", ""))
        if target is None:
            raise ValueError("FAQ redirect leaves the allowed scope")
        # Request the actual URL, while using normalization only to check scope.
        url = str(
            response.url.join(response.headers["location"]).copy_with(fragment=None)
        )
    raise ValueError("Too many FAQ redirects")


def fetch_html(
    client: httpx.Client,
    url: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> HTMLResponse:
    attempt = 0
    while True:
        response = None
        try:
            response = _request(client, url)
        except httpx.TransportError as error:
            if attempt == 2:
                raise ValueError(f"FAQ transport failed: {url}") from error
        else:
            if response.status_code != 429 and response.status_code < 500:
                return _html_response(response)
            if attempt == 2:
                raise ValueError(f"FAQ HTTP {response.status_code}: {url}")
        sleep(_retry_delay(response, attempt))
        attempt += 1


def _html_response(response: httpx.Response) -> HTMLResponse:
    content_type = (
        response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if response.status_code != 200 or content_type not in (
        "text/html",
        "application/xhtml+xml",
    ):
        raise ValueError(f"FAQ requires complete HTML, got HTTP {response.status_code}")
    return HTMLResponse(
        response.content, str(response.url), datetime.now(UTC), dict(response.headers)
    )
