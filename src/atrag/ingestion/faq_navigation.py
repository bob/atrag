"""Recognize the AT FAQ templates and their navigation, without parsing answers."""

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

HOST = "info.portaldasfinancas.gov.pt"
PREFIX = "/pt/apoio_contribuinte/questoes_frequentes/"
CONTENT_ID = "ctl00_PlaceHolderMain_ctl02__ControlWrapper_RichHtmlField"


def normalize_faq_url(base: str, href: str) -> str | None:
    """Normalize the confirmed Pages/pages and HTTP aliases; retain query bytes."""
    try:
        parts = urlsplit(urljoin(base, href))
        allowed = (
            parts.scheme in ("https", "http")
            and parts.hostname == HOST
            and parts.port in (None, 80, 443)
            and parts.username is None
        )
    except ValueError:
        return None
    if not allowed or not re.fullmatch(
        re.escape(PREFIX) + r"pages/faqs(?:-\d+)?\.aspx", parts.path, re.IGNORECASE
    ):
        return None
    path = parts.path.lower()
    if path == PREFIX + "pages/faqs.aspx":
        path = PREFIX + "Pages/faqs.aspx"
    return urlunsplit(("https", HOST, path, parts.query, ""))


@dataclass(frozen=True)
class HTMLInspection:
    template: str
    navigation_urls: tuple[str, ...]
    question_count: int


def _is_answer_link(link: Tag) -> bool:
    return link.find_parent(attrs={"data-bs-parent": "#faqAccordion"}) is not None


def _navigation(content: Tag, url: str) -> tuple[str, ...]:
    targets = set()
    for link in content.select("a[href]"):
        href = str(link["href"])
        if href.startswith("#") or _is_answer_link(link):
            continue
        target = normalize_faq_url(url, href)
        if target is not None:
            targets.add(target)
        elif link.get("rel") or "__doPostBack" in href:
            raise ValueError("Unsupported FAQ pagination/navigation control")
    return tuple(sorted(targets))


def inspect_html(raw: bytes, url: str) -> HTMLInspection:
    """Reject unknown/error pages and count all numbered headings, including spillover."""
    soup = BeautifulSoup(raw, "html.parser")
    content = soup.select_one(f"#{CONTENT_ID}")
    if content is None:
        raise ValueError("Unknown FAQ template: missing page content")
    headings = soup.select(".card-header a[aria-controls]")
    questions = [
        h
        for h in headings
        if re.fullmatch(r"\d+\s*-\s*.+", h.get_text(" ", strip=True))
    ]
    if soup.find(id="faqAccordion") is not None:
        if not questions or len(questions) != len(headings):
            raise ValueError("Unknown or empty FAQ accordion")
        template = "faq-accordion"
    elif content.select(".submenu-itens-sub a[href]"):
        template = "catalog-navigation"
    else:
        raise ValueError("Unknown FAQ navigation template")
    navigation = _navigation(content, url)
    if template == "catalog-navigation" and not navigation:
        raise ValueError("FAQ navigation has no in-scope targets")
    return HTMLInspection(template, navigation, len(questions))
