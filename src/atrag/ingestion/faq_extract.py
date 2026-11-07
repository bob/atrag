"""Pure parsing: bind each published question to its own answer across the DOM."""

import re
from collections import Counter
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Tag

from atrag.ingestion.faq_content import (
    answer_links,
    answer_tables,
    answer_text,
    list_items,
    plain_text,
)
from atrag.ingestion.legal_references import REFERENCE
from atrag.schemas.ingestion.faq import FAQEntry
from atrag.schemas.ingestion.faq_extraction import ExtractedFAQ, FAQIssue, FAQPage
from atrag.schemas.materials import HTMLLocation

PARSER_VERSION = "faq-1"
QUESTION = re.compile(r"^(\d+)\s*-\s*(\S.*)$", re.DOTALL)
UNSUPPORTED = (
    "img,svg,canvas,iframe,object,embed,video,audio,math,input,select,textarea"
)


def _answer(soup: BeautifulSoup, header: Tag) -> Tag:
    links = header.select("a[aria-controls]")
    if len(links) != 1 or not header.get("id"):
        raise ValueError("Heading must have an ID and one answer control")
    link = links[0]
    target = str(link["aria-controls"])
    answers = soup.find_all(id=target)
    if len(answers) != 1 or link.get("href") != f"#{target}":
        raise ValueError("Answer target must be unique and match href")
    answer = answers[0]
    if answer.get("aria-labelledby") != header["id"]:
        raise ValueError("Answer must refer back to its own heading")
    bodies = answer.select(".card-body")
    if len(bodies) != 1 or bodies[0].select_one(".card-header") is not None:
        raise ValueError("Answer must contain exactly one isolated body")
    return bodies[0]


def _issue(kind: str, ordinal: int | None, detail: str) -> FAQIssue:
    return FAQIssue(kind=kind, ordinal=ordinal, detail=detail)


def _content(body, entry_id, location, question, ordinal):
    issues = []
    try:
        tables = answer_tables(body, entry_id, location, question)
    except ValueError as error:
        tables = ()
        issues.append(_issue("unsupported_table", ordinal, str(error)))
    return tables, issues


def _parse_one(soup, header, page_url, topic_path, ordinal):
    match = QUESTION.fullmatch(plain_text(header))
    if match is None:
        return None, (), [_issue("invalid_question", ordinal, plain_text(header))]
    try:
        body = _answer(soup, header)
    except ValueError as error:
        return None, (), [_issue("answer_binding", ordinal, str(error))]
    answer = answer_text(body)
    content_issues = [
        _issue("unsupported_element", ordinal, str(node))
        for node in body.select(UNSUPPORTED)
    ]
    if not answer:
        return None, (), [*content_issues, _issue("empty_answer", ordinal, match[1])]
    page_key = PurePosixPath(urlsplit(page_url).path).stem
    entry_id = f"{page_key}:{match[1]}"
    location = HTMLLocation(
        kind="html",
        page_url=page_url,
        section=" > ".join(topic_path),
        anchor=str(header["id"]),
    )
    tables, table_issues = _content(body, entry_id, location, match[2], ordinal)
    issues = [*content_issues, *table_issues]
    entry = FAQEntry(
        entry_id=entry_id,
        faq_id=match[1],
        question=match[2],
        answer=answer,
        topic_path=topic_path,
        page_url=page_url,
        anchor=location.anchor,
        links=answer_links(body, page_url),
        ordinal=ordinal,
    )
    record = ExtractedFAQ(
        entry=entry,
        question_normalized=" ".join(entry.question.split()),
        answer_normalized=" ".join(answer.split()),
        table_ids=tuple(t.table_id for t in tables),
        list_items=list_items(body),
        references=tuple(
            m[0] for m in REFERENCE.finditer(f"{entry.question}\n{answer}")
        ),
    )
    return record, tables, issues


def _duplicates(soup, headers):
    counts = Counter(str(n["id"]) for n in soup.select("[id]"))
    return [
        _issue("duplicate_dom_id", ordinal, str(header.get("id")))
        for ordinal, header in enumerate(headers, 1)
        if counts[str(header.get("id"))] > 1
    ]


def parse_faq_page(raw: bytes, page_url: str, topic_path: tuple[str, ...]) -> FAQPage:
    soup = BeautifulSoup(raw, "html.parser")
    headers = soup.select(".card-header")
    if not headers:
        return FAQPage(
            page_url=page_url,
            detected_count=0,
            issues=(_issue("unknown_template", None, "No FAQ headings"),),
        )
    issues = _duplicates(soup, headers)
    answer_count = len(soup.select(".card-body"))
    if answer_count != len(headers):
        issues.append(
            _issue(
                "answer_count_mismatch",
                None,
                f"{len(headers)} headings, {answer_count} answer bodies",
            )
        )
    faqs, tables, rejected, seen = [], [], [], set()
    for ordinal, header in enumerate(headers, 1):
        try:
            record, found_tables, found_issues = _parse_one(
                soup, header, page_url, topic_path, ordinal
            )
        except ValueError as error:
            record, found_tables, found_issues = (
                None,
                (),
                [_issue("invalid_content", ordinal, str(error))],
            )
        issues.extend(found_issues)
        if record is None:
            rejected.append(ordinal)
            continue
        if record.entry.entry_id in seen:
            issues.append(_issue("duplicate_faq_id", ordinal, record.entry.entry_id))
            rejected.append(ordinal)
            continue
        seen.add(record.entry.entry_id)
        faqs.append(record)
        tables.extend(found_tables)
    return FAQPage(
        page_url=page_url,
        detected_count=len(headers),
        answer_count=answer_count,
        detected_table_count=len(soup.select(".card-body table")),
        faqs=tuple(faqs),
        tables=tuple(tables),
        issues=tuple(issues),
        rejected_ordinals=tuple(rejected),
    )
