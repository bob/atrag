# ATRAG

RAG для португальских налоговых документов с вопросами и ответами на русском, английском и португальском.

Стек: Python, LangGraph, Streamlit, OpenAI API, PostgreSQL и расширение pgvector. Настройки подключения и API-ключи загружаются из корневого `.env`.

Решение от 30.10.2026: PostgreSQL выбран основным хранилищем источников, редакций норм, фрагментов, метаданных и embeddings. Векторный поиск выполняется через pgvector, полнотекстовый - средствами PostgreSQL; результаты объединяются в гибридном поиске. Исходные PDF хранятся в файлах или объектном хранилище, с проверяемыми ссылками из базы.

Согласованный доступ к БД: SQLAlchemy 2.x, Psycopg 3, пакет `pgvector`, Alembic и `langgraph-checkpoint-postgres`; настройки читает `python-dotenv`. Первая версия использует синхронные подключения, пул соединений и отдельную сессию для каждой операции. Подробности - в [плане взаимодействия с БД](docs/document-ingestion/README.md#библиотеки-и-взаимодействие-с-бд).

Реализован Шаг 1 плана `document-ingestion`: типизированные настройки, реестр CIRS и настройка синхронного подключения SQLAlchemy. Следующий шаг — неизменяемый снимок PDF. Первая часть проекта — подготовка документов и поискового корпуса; вторая — чат с поисковым инструментом.

Для запуска нужен Python 3.14 и `uv`:

```sh
uv sync --locked
cp .env.example .env
uv run atrag config check
```

В `.env` укажите `DATABASE_URL` для своей базы на уже запущенном локальном PostgreSQL. Формат — `postgresql://user:password@localhost:5432/database`; для локальной аутентификации через сокет подходит `postgresql:///database`. Пароль из шаблона — заглушка. Ключ OpenAI понадобится на этапе embeddings.

`atrag config check` проверяет настройки и `data/sources/cirs.json`, выводит безопасный JSON без API-ключа и DSN. Команда работает без базы и ключа OpenAI; признаки `database_configured` и `openai_configured` показывают наличие настроек. Переменные процесса имеют приоритет над `.env`, относительный `ATRAG_DATA_DIR` отсчитывается от корня проекта. Загрузчик `atrag.settings.load_settings()` общий для CLI и будущего Streamlit-приложения.

Реестр содержит португальский оригинал, URL со статусом `candidate` и известную отметку редакции сборника. Неизвестные даты получения и изменения нормы остаются `null`. Реестр хранится в Git; `.env`, промежуточные данные и отчёты исключены.

Модульные проверки:

```sh
uv run pytest -m 'not integration'
uv run ruff check src tests
uv run ruff format --check src tests
```

Интеграционные проверки используют отдельную временную базу с именем `atrag_test_*` на локальном сервере. В ней должен быть включён `vector`; расширение устанавливается отдельно от Python-пакета `pgvector` по [официальной инструкции](https://github.com/pgvector/pgvector#installation).

```sh
createdb atrag_test_ingestion
psql -d atrag_test_ingestion -c 'CREATE EXTENSION vector'
ATRAG_TEST_DATABASE_URL=postgresql:///atrag_test_ingestion uv run pytest
dropdb atrag_test_ingestion
```

Без `ATRAG_TEST_DATABASE_URL` интеграционные тесты пропускаются. Они проверяют векторный поиск и откат через SQLAlchemy/Psycopg, восстановление соединения и отдельный Psycopg-пул checkpointer. Проверено на Python 3.14.6, PostgreSQL 16.13 и pgvector 0.8.7; версии Python-зависимостей закреплены в `uv.lock`.

- [План подготовки документов на примере CIRS.pdf](docs/document-ingestion/README.md)
- [Журнал действий и решений](docs/document-ingestion/ACTION_LOG.md)
