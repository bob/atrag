# ATRAG

Подготовка португальских налоговых документов и поиск с вопросами на русском,
английском и португальском. Текущий MVP работает с одним снимком CIRS.pdf:
извлечение PDF/OCR, юридическая структура, части текста, embeddings с кешем,
переносимый пакет, атомарная индексация в PostgreSQL и проверка поиска.

Смысловые вопросы используют pgvector; статьи и пункты выбираются SQL-фильтрами.
Чат Streamlit и инструмент LangGraph относятся к следующей части проекта.
Исторические редакции и применимость по налоговым годам пока не реализованы.

Ниже — общий порядок запуска от новой базы до поиска. Все команды выполняются
из корня проекта. Если пользователь PostgreSQL, база и `.env` уже существуют,
пропусти их создание и сохрани текущие настройки. Для подготовленного документа
вместо полного цикла можно выбрать [импорт готового пакета](#5-импорт-уже-готового-пакета).

## 1. Окружение и настройки

Нужны Python 3.14, `uv`, запущенный PostgreSQL с установленным серверным
pgvector и, для полного извлечения PDF, Tesseract. Python-пакет `pgvector`
не устанавливает серверное расширение PostgreSQL. Инструкция предполагает,
что эти системные зависимости установлены. Серверный pgvector должен соответствовать
версии PostgreSQL; установка описана в [официальной инструкции](https://github.com/pgvector/pgvector#installation).

```sh
command -v uv psql createuser createdb tesseract
uv sync --locked

# Только если .env ещё не существует:
if [ ! -f .env ]; then
  cp .env.example .env
fi
```

Заполни `.env`, если это новая установка:

```dotenv
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=1536
DATABASE_URL=postgresql://atrag:YOUR_DATABASE_PASSWORD@localhost:5432/atrag
ATRAG_DATA_DIR=./data
```

Пароль в `DATABASE_URL` должен соответствовать паролю пользователя PostgreSQL;
спецсимволы в части URL с паролем нужно кодировать. Указанные выше значения
ключа и пароля — места для заполнения. Переменные окружения процесса имеют
приоритет над `.env`.

```sh
uv run atrag config check
```

Для полного цикла проверь в ответе `database_configured: true`,
`openai_configured: true`, модель large и 1536 измерений. Эта команда проверяет
настройки и реестр источника; она не подключается к БД и не создаёт таблицы.
Для импорта готового пакета и точного поиска ключ OpenAI не требуется.

## 2. Пользователь, база, vector и таблицы

Укажи существующего администратора PostgreSQL. На текущем компьютере это
`postgres`; для другой установки замени значение:

```sh
ATRAG_PG_ADMIN=postgres
```

**Только для новой установки, если пользователь и база отсутствуют:**

```sh
# Запросит новый пароль интерактивно. Тот же пароль укажи в .env.
createuser -h localhost -U "$ATRAG_PG_ADMIN" --login --pwprompt atrag
createdb -h localhost -U "$ATRAG_PG_ADMIN" --owner=atrag atrag
```

**В целевой базе однократно включи расширение от имени администратора:**

```sh
psql -h localhost -U "$ATRAG_PG_ADMIN" -d atrag -v ON_ERROR_STOP=1 \
  -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

Если PostgreSQL сообщает, что расширение `vector` недоступно, сначала установи
серверный pgvector для версии своего PostgreSQL по
[официальной инструкции](https://github.com/pgvector/pgvector#installation).

**Затем создай таблицы от имени пользователя из `.env`:**

```sh
uv run atrag db migrate
```

Успешная команда возвращает `status: succeeded`, `action: migrate`.
Она создаёт кеш embeddings и таблицы корпуса, применяя миграции до
`0003_vector_search`. Повторный запуск применяет только недостающие миграции;
база и корпус не пересоздаются. FTS в текущем MVP нет.

Далее выбери один вариант: разделы 3–4 для полной подготовки либо раздел 5
для уже подготовленного пакета. Тестовую БД `atrag_test_context_20261101`
для рабочего корпуса не используй.

## 3. PDF, OCR и сохранённый снимок

Для текущего CIRS нужны:

- исходный PDF: `docs/CIRS.pdf`;
- реестр: `data/sources/cirs.json`;
- проверенные таблицы этого снимка: `data/sources/cirs-extraction-review.json`;
- португальская модель OCR: `data/ocr/tessdata/por.traineddata`.

PDF, OCR-модель и рабочие артефакты доставляются отдельно от Git. Реестр и
ручная сверка хранятся в репозитории. До первого запуска проверь наличие
перечисленных файлов.
Если отсутствует только OCR-модель:

```sh
mkdir -p data/ocr/tessdata
curl --fail --location --max-time 60 \
  https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/por.traineddata \
  -o data/ocr/tessdata/por.traineddata
```

Проверь OCR и предварительно загрузи словарь токенизатора:

```sh
tesseract --tessdata-dir data/ocr/tessdata --list-langs
uv run python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base")'
```

В списке языков должен быть `por`. Первая загрузка словаря требует сети;
она не вызывает платный API OpenAI.

PDF должен совпадать с `expected_sha256` в реестре. Для другого снимка CIRS
нужно отдельно обновить реестр и сверку таблиц; старую ручную сверку нельзя
автоматически переносить на изменившийся PDF. Другие семейства документов
потребуют собственного парсера.

## 4. Полный цикл: PDF → части → embeddings → пакет → индекс

Этот блок предназначен для `bash` и останавливается при ошибке. Пути манифестов
передаются автоматически из JSON предыдущей команды; вручную копировать UUID
не требуется. `inspect` сам создаёт или проверяет неизменяемый снимок PDF,
поэтому отдельный `ingest snapshot` для этого сценария необязателен.

Этап `embed` вызывает платный API для отсутствующих в кеше текстов. Свежая БД
имеет пустой кеш; если нужны уже оплаченные векторы, используй раздел 5.
Остальные этапы этого блока работают без API OpenAI.

```sh
bash <<'INGEST'
set -euo pipefail

atrag_manifest_path() {
  uv run python -c 'import json, sys; from pathlib import Path; print(Path(json.load(sys.stdin)["directory"]) / "manifest.json")'
}

# 1. Снимок, текст, таблицы, OCR и ручная сверка текущего CIRS.
ATRAG_EXTRACTED_MANIFEST="$(
  uv run atrag ingest inspect --source docs/CIRS.pdf \
    --tessdata-dir data/ocr/tessdata \
    --review data/sources/cirs-extraction-review.json |
    atrag_manifest_path
)"

# 2. Юридические единицы и ссылки, без повторного PDF/OCR.
ATRAG_PARSED_MANIFEST="$(
  uv run atrag ingest parse --extracted "$ATRAG_EXTRACTED_MANIFEST" |
    atrag_manifest_path
)"

# 3. Поисковые части с точными диапазонами исходного текста.
ATRAG_CHUNKED_MANIFEST="$(
  uv run atrag ingest chunk --parsed "$ATRAG_PARSED_MANIFEST" |
    atrag_manifest_path
)"

# 4. Embeddings через OpenAI с постоянным кешем PostgreSQL.
ATRAG_EMBEDDED_MANIFEST="$(
  uv run atrag ingest embed --chunked "$ATRAG_CHUNKED_MANIFEST" |
    atrag_manifest_path
)"

# 5. Новый каталог переносимого пакета для этого запуска.
ATRAG_PACKAGE_DIR="data/packages/cirs-$(date -u +%Y%m%dT%H%M%SZ)"
uv run atrag ingest package \
  --parsed "$ATRAG_PARSED_MANIFEST" \
  --chunked "$ATRAG_CHUNKED_MANIFEST" \
  --embedded "$ATRAG_EMBEDDED_MANIFEST" \
  --review data/sources/cirs-extraction-review.json \
  --output "$ATRAG_PACKAGE_DIR"

# 6. Атомарная загрузка готовых данных в БД из DATABASE_URL.
uv run atrag ingest index --package "$ATRAG_PACKAGE_DIR/manifest.json"

printf '\nГотовый пакет: %s/manifest.json\n' "$ATRAG_PACKAGE_DIR"
INGEST
```

Каждый этап пишет отдельный каталог `ATRAG_DATA_DIR/runs/<run_id>/` с
`manifest.json`, временем, статусом и контрольными суммами. Входные файлы
сохраняются. Ошибка останавливает блок; завершённые этапы можно использовать
повторно через команды из раздела [«Повторный запуск и отчёты»](#7-повторный-запуск-и-отчёты).
После сбоя `embed` успешно сохранённые пакеты векторов остаются в кеше.
Ошибка индексации сохраняет предыдущий полный индекс.

Для проверенного текущего снимка ожидаются 1789 юридических единиц и 1032
части. `needs_review` остаётся допустимым статусом: таблица статьи 100 и
400 неразрешённых ссылок явно отражены в отчётах, а не объявлены исправленными.

## 5. Импорт уже готового пакета

Для текущего компьютера настоящий пакет large / 1536 уже подготовлен:

```sh
uv run atrag ingest index \
  --package target/live-acceptance-large-20261101/package/manifest.json
```

Выполни это после разделов 1–2. Команда загрузит PDF, 1789 единиц, 1032 части
и сохранённые векторы в новую БД `atrag`. Она не вызывает OpenAI и не требует
повторного OCR. Повторный импорт не создаёт дубликаты.

На другой машине передай весь каталог пакета и укажи путь к его `manifest.json`.
Пакет содержит PDF, реестр снимка и необходимые подготовленные данные;
Tesseract и исходный каталог `data/runs/` для импорта не нужны.

## 6. Проверка поиска и эталон

Для текущего CIRS snapshot_id:
`e8671caafbcda885e5c87d451d8d2ef121b36bc7ab63eb2eb1ab474970b15a72`.
Он совпадает с SHA-256 PDF и полем `snapshot_id` пакета.

Точная проверка без OpenAI:

```sh
uv run atrag search 'CIRS 16' --language ru \
  --snapshot e8671caafbcda885e5c87d451d8d2ef121b36bc7ab63eb2eb1ab474970b15a72
```

Смысловой поиск вызывает API для одного embedding вопроса:

```sh
uv run atrag search 'Как определяется налоговое резидентство в Португалии?' \
  --language ru \
  --snapshot e8671caafbcda885e5c87d451d8d2ef121b36bc7ab63eb2eb1ab474970b15a72
```

Ответ `search` содержит текст, цитаты и страницы без координат PDF:
`bbox` и `cell_bboxes` исключены, в том числе из вложенного контекста,
примечаний и таблиц. После удаления координат одинаковые записи в каждом
массиве `locators` объединяются с сохранением порядка страниц и разных
печатных меток. Геометрия сохраняется в подготовленных данных;
пересборка пакета и повторная индексация не нужны.

Полный эталон запускается на уже импортированном пакете. Для готового локального
пакета команда такая; для полного цикла подставь напечатанный в конце раздела 4
путь к пакету:

```sh
uv run atrag evaluate \
  --package target/live-acceptance-large-20261101/package/manifest.json \
  --benchmark docs/document-ingestion/benchmark-v2.json \
  --context-tokens 6000
```

Эталон выполняет 36 поисковых случаев и 18 расширений, включая 18 платных
query embeddings. Он не векторизует корпус повторно. В ответе CLI будет путь
к `report.json`. Проверь для `vector:ru`, `vector:en`, `vector:pt`:
`semantic_hit_at_5`, `exact_pass_rate`, `citations_correct` и `failed_case_ids`,
а также `passed` всех `expansion_records`. Успешное завершение команды означает,
что измерение завершилось; сами результаты оцениваются по отчёту.

Подтверждённая приёмка и ограничения: [векторный MVP](docs/document-ingestion/VECTOR_MVP.md).


## 7. Повторный запуск и отчёты

При остановке полного цикла найди последний успешный `manifest.json` в
`ATRAG_DATA_DIR/runs/`. Статус должен быть `succeeded`; манифесты `failed`
и `running` не являются завершёнными входами. Передай путь следующей команде:

```sh
# Примеры: подставь фактические каталоги завершённых этапов.
uv run atrag ingest parse --extracted data/runs/EXTRACTION_RUN_ID/manifest.json
uv run atrag ingest chunk --parsed data/runs/PARSE_RUN_ID/manifest.json
uv run atrag ingest embed --chunked data/runs/CHUNK_RUN_ID/manifest.json
```

Повтор `embed` использует постоянный PostgreSQL-кеш по модели, размерности
и хешу входного текста. Успешные пакеты сохраняются отдельно, поэтому ошибка
не требует заново оплачивать все embeddings. Полностью заполненный кеш
работает без API-ключа. Смена модели создаёт отдельное пространство векторов.

Для каждого этапа сохраняются манифест, входные/выходные SHA-256 и отчёты.
Извлечение оставляет `pages.jsonl`, таблицы, координаты и OCR; разбор —
`legal_units.jsonl`; разбиение — `fragments.jsonl`; embeddings —
`embeddings.jsonl`, usage и попадания кеша. Оригинальный PDF сохраняется
неизменяемым в `ATRAG_DATA_DIR/raw/<sha256>/source.pdf`.

Пакет переносится целиком. Индексация проверяет его до записи и транзакционно
заменяет только выбранный снимок. PDF/OCR, создание embeddings и проверка
юридических периодов при импорте не выполняются. Source URL, происхождение,
страницы и известные замечания сопровождают результаты поиска.

## 8. Поисковый сервис и ограничения MVP

Общий сервис для CLI и будущего инструмента:

```python
def search_tax_documents(
    query,
    limit=5,
    query_language=None,
    *,
    snapshot_id,
    engine,
    client=None,
    filters=None,
    context_tokens=6000,
):
    ...
```

Модель и размерность берутся из метаданных проиндексированного снимка.
Смысловой `score` равен косинусному сходству, `vector_rank` — позиции среди
50 кандидатов. Для точного адреса `score=1.0`, `vector_rank=null`, части
следуют в исходном порядке. HNSW пока не нужен по выполненным измерениям.

Язык вопроса не фильтрует язык оригинального текста. `--language` принимает
ru/en/pt; для короткого адреса он обязателен. Для полного вопроса язык можно
определять эвристически; явный выбор предпочтителен. Фильтры CLI:
`--code`, `--article`, `--scope`.

Поиск возвращает португальские части, цитаты, источник, страницы, таблицы,
доступные адреса ссылок и ограничения. Полный оригинал, продолжения, примечания
и выбранные зависимости загружаются отдельными сервисами с явным бюджетом.
Контракты: [части](docs/document-ingestion/PARTS.md) и
[контекст](docs/document-ingestion/CONTEXT.md).

Замечания в ответе поиска относятся только к фактически возвращённым частям,
их единицам, страницам и таблицам после применения `limit` и бюджета.
При пустой выдаче списки замечаний пустые; причина `no_match` и ограничение
бюджета сохраняются. Полная диагностика корпуса остаётся в отчётах подготовки,
`chunk/report.json` пакета и данных снимка в БД. Повторно импортировать корпус
для этого поведения не нужно. Подробности — в
[контракте замечаний](docs/document-ingestion/CONTEXT.md#замечания-в-результатах-поиска).

Профиль CIRS — `text-embedding-3-large` / 1536. При отсутствии настройки
встроенное значение модели — small; в `.env.example` указан прошедший приёмку
large. Small пропускает один русский смысловой случай эталона. Векторный MVP
даёт 6/6 на каждом языке, 18/18 точных случаев и 18/18 расширений; после
упрощения выдача проверена на сохранённых настоящих векторах без новых API-вызовов.
Шесть смысловых вопросов на язык не доказывают качество на любых новых вопросах.

Таблица статьи 100 и 400 неразрешённых ссылок остаются явными ограничениями
корпуса. FTS исключён из реализации MVP; его возможное возвращение описано
в [планах расширения](docs/document-ingestion/README.md#10-будущие-расширения).

## 9. Проверки разработки

```sh
uv run pytest -m 'not integration'
uv run ruff check src tests
uv run ruff format --check src tests
```

Для PostgreSQL-тестов используй отдельную сохранённую базу, а не рабочую `atrag`.
Только если тестовой базы ещё нет, создай её и включи расширение:

```sh
createdb -h localhost -U "$ATRAG_PG_ADMIN" atrag_test_context_20261101
psql -h localhost -U "$ATRAG_PG_ADMIN" -d atrag_test_context_20261101 \
  -v ON_ERROR_STOP=1 -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

Последующие прогоны используют ту же БД; тестовые таблицы очищаются, сама база
не удаляется и не пересоздаётся:

```sh
ATRAG_TEST_DATABASE_URL=postgresql://localhost/atrag_test_context_20261101 uv run pytest
```

Без `ATRAG_TEST_DATABASE_URL` интеграционные тесты пропускаются. Тесты проверяют
миграции, ограничения, кеш, атомарность, смену модели, повторный импорт,
изоляцию снимков, косинусное ранжирование и точные адреса. Живой HTTP в обычных
тестах заблокирован. Последний полный прогон: 488 тестов и 100% покрытия;
Ruff check/format и синтаксис Python 3.13 проходят. Runtime
остаётся Python 3.14, Ruff target — `py313`.

## Документация

- [План document-ingestion](docs/document-ingestion/README.md).
- [Приёмка текущего векторного MVP](docs/document-ingestion/VECTOR_MVP.md).
- [Первоначальные живые измерения и стоимость](docs/document-ingestion/LIVE_ACCEPTANCE.md).
- [Ограничения корпуса](docs/document-ingestion/CORPUS_REVIEW.md).
- [Журнал действий и решений](docs/document-ingestion/ACTION_LOG.md).
