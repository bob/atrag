# Project conventions

- Keep Python syntax compatible with Python 3.13 so Pylance does not report syntax errors for Python 3.14-only constructs. The project's runtime requirement remains Python 3.14.
- Always parenthesize multiple exception types: `except (ValueError, OSError):`.
- Preserve Ruff's `target-version = "py313"` setting and check changed Python files with `ruff check` and `ruff format --check`.
- Reuse the local PostgreSQL test database `atrag_test_context_20261101`. Clean test tables between runs; do not recreate or drop the database after each task. Create it only if it is missing.
