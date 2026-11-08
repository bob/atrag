# Локальная приёмка FAQ ingestion

08.11.2026: Шаги 0–6 выполнены локально. Итоговые отчёты:
[Шаг 4 — пакет](STEP4.md), [Шаг 5 — общий поиск и публикация](STEP5.md),
[Шаг 6 — ответы и ссылки](STEP6.md).
Production-доставка в эту приёмку не входит и не выполнялась.

## Воспроизведение проверок кода

Из корня проекта, с установленным окружением:

```sh
ATRAG_TEST_DATABASE_URL=postgresql://localhost/atrag_test_context_20261101 \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check src/atrag/chat src/atrag/interface src/atrag/schemas/settings.py \
  tests/test_citation_links.py tests/test_chat_answers.py tests/test_chat_sources.py \
  tests/test_chat_agent.py tests/test_interface.py tests/integration/test_faq_retrieval.py \
  tests/integration/test_chat_tools.py tests/integration/test_chat_turns.py \
  tests/integration/test_interface_embeddings.py
.venv/bin/ruff format --check src/atrag/chat src/atrag/interface src/atrag/schemas/settings.py \
  tests/test_citation_links.py tests/test_chat_answers.py tests/test_chat_sources.py \
  tests/test_chat_agent.py tests/test_interface.py tests/integration/test_faq_retrieval.py \
  tests/integration/test_chat_tools.py tests/integration/test_chat_turns.py \
  tests/integration/test_interface_embeddings.py
```

Использовать существующую `atrag_test_context_20261101`: fixtures очищают
тестовые таблицы. Базу не пересоздавать и не удалять. Тесты блокируют живой
HTTP; платные API для этого прогона не нужны. Итог: **1134 passed**.

## Настоящие данные и ответы

Локально выбран release
`46f8e0989a5cf9cb11cc5030fc342e0a7771adfe6219c9b4a9ba9744202abde2`:

| Источник | Снимок |
| --- | --- |
| pt-at-cirs | `1f72f3e5480ebf498fd119293d6d8a8dca31415f374e388c4dbed73565853cd5` |
| pt-at-faq | `ed0ffdd17578885d8b7f040c15ec7a6c341b16e00d425b43179d388da12d4040` |

Неизменённый [эталон](benchmark.json): общий поиск FAQ 48/48,
прежний CIRS 36/36 (Шаг 5); смешанные живые ответы 16/16 и повторное
открытие через AppTest 16/16 (Шаг 6).

Полный пакет, исходники, embeddings и API receipts находятся вне Git.
Команды получения, подготовки, импорта, изменения публикации и отката
с конкретными путями и снимками сохранены в [STEP2.md](STEP2.md),
[STEP4.md](STEP4.md) и [STEP5.md](STEP5.md). Перед любым изменением указателя
сохранить текущий состав `atrag corpus show`; для отката публиковать именно
сохранённые пары source_id/snapshot_id.

Скрипты Шага 6 сохранены вместе с внешними артефактами:

```sh
# Требует локальный корпус и API-аудиты Шага 5; новые ответы используют API.
.venv/bin/python data/faq-step6-live/faq-step6-live.py
# Проверяет реальные официальные HTML-ссылки; скачанный CIRS сохранён рядом.
.venv/bin/python data/faq-step6-live/faq-step6-links-check.py
```

Скрипт живых ответов возобновляет проверку по отсутствующим результатам.
Для нового измерения сохранить прежний каталог и явно выбрать новый OUT.
Для UI-повтора скопировать сохранённый `faq-step6-ui-smoke.py` в `.metrics/`,
временно запустить его на localhost:8765 и выполнить сохранённый
`faq-step6-ui-check.py`. Fixture подставляет только локальную тестовую
личность и открывает настоящий `streamlit_app.py`; в production её не
использовать. После проверки остановить свой сервер. Обычный пользовательский
запуск по-прежнему выполняется через `streamlit run streamlit_app.py` с OIDC.

Официальные URL живые. PDF CIRS сейчас отличается от цитируемого снимка:
интерфейс показывает соответствующую отметку происхождения и сохранённую
версию. Legacy-история без ссылки остаётся без ссылки.

Перед отдельной production-доставкой нужны полный проверенный пакет,
резервный состав корпуса и проверка нового хода после публикации: оба
релевантных источника, корректные номера, ссылки и повторное открытие истории.
При провале восстановить прежний состав по инструкции Шага 5.
