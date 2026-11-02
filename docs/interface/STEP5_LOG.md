# Шаг 5 — выполнение хода и восстановление

Дата: 02.11.2026. Режим: `ub-mode`.

| Действие | Причина и результат |
| --- | --- |
| Прочитаны AGENTS.md, ub-mode, PLAN.md и контракты chat | Граница работы — выполнение и восстановление; Streamlit относится к шагу 6 |
| Проверены status, история и worktree | Работа ведётся в master; прежние изменения README.md, docs/interface/README.md, CHAT_REFACTOR_LOG.md и corpus-changes.json сохраняются отдельно |
| Изучены ChatStore, RagService, агент, миграции и фикстуры PostgreSQL | Переиспользуются допуск, реестр источников и сохранённый состав; миграции не вызываются при исполнении |
| Прочитаны langgraph-fundamentals, persistence и human-in-the-loop, проверена официальная документация и установленные API | Восстановление после ошибки использует None с прежним thread_id; sync durability сохраняет границы узлов |
| Написаны интеграционные сценарии до реализации | Ожидаемый ModuleNotFoundError сохранён в `.metrics/interface-step5-red.txt` |
| Добавлены TurnService, ExecutionStore и LangGraphBackend с контрактами событий | Сервис не импортирует LangChain; адаптер использует существующие агент и RAG, SQL остаётся в db |
| Запущены PostgreSQL-проверки | Sandbox запрещает localhost; повтор с разрешением использует существующую atrag_test_context_20261101, без пересоздания БД |
| Исправлена обработка пустых middleware updates и идентификатор блока фикстуры | Первые проверки выявили `None` в update и неверный адрес синтетического материала; ошибки разобраны по серверному журналу |
| Разобрано восстановление после закрытия генератора | `get_state()` может применить pending writes узла при пустом next; отсутствие проверенного answer требует stream(None), а не повторного вопроса |
| Очищены поля результата при добавлении нового вопроса в checkpoint | Результат предыдущего хода не может стать результатом нового при сбое до before_agent |
| Добавлены проверки допуска до backend, ошибок открытия, некорректных ссылок, reasoning и настоящих токенов | 12 сценариев проходят; `.metrics/interface-step5-targeted.txt`; ошибки UI содержат только стабильные коды |
| Добавлена очистка checkpoint-таблиц между сценариями | Тесты переиспользуют миграции и очищают данные; база не удаляется |
| Выполнен исходный полный прогон | 713 тестов прошли; два защитных участка дополнены поведенческими сценариями незавершённого backend и чужого pending checkpoint |
| Выполнен первый mutator | 25/29 мутаций уничтожены; оставшиеся относятся к имени инструмента, последнему публичному сообщению, пустым/служебным событиям и избыточной ветке восстановления |
| Усилены тесты по мутациям и удалена избыточная ветка | Проверяются имена/ID/порядок событий, отсутствие выдачи tool result в текст, сохранение текста после повторной ошибки и видимость checkpoint до возврата модели; завершённый граф также восстанавливается через stream(None) |
| Воспроизведены два дополнительных сбоя до исправления | До нового checkpoint прежний ответ/usage могли попасть в новый ход; KeyboardInterrupt при открытии backend оставлял running. Два ожидаемых падения — `.metrics/interface-step5-boundaries-red.txt` |
| Привязано чтение результата checkpoint к UUID текущего хода и централизованы статусы | Прежний результат не копируется; interruption обрабатывается и до открытия backend; завершённое событие выдаётся после закрытия backend |
| Выполнен итоговый полный прогон после исправлений | 721 тест прошёл; 3744 строки приложения покрыты на 100%; `.metrics/interface-step5-tests.txt` и `target/coverage/python/lcov.info` |
| Выполнены crapper и dryer по всему приложению | Максимальный CRAP — 9, дублирования от 0.8 нет; `.metrics/interface-step5-crapper.txt`, `.metrics/interface-step5-dryer.txt` |
| Проверены Ruff, ty и синтаксис изменённого Python | Все 6 Python-файлов разбираются как Python 3.13; runtime >=3.14 и Ruff py313 сохранены; `.metrics/interface-step5-types.txt`, `.metrics/interface-step5-syntax.txt` |
| Описаны контракт выполнения и интеграция, обновлён статус плана | EXECUTION.md содержит события, статусы, ошибки, правила закрытия генератора и явного продолжения; следующий шаг — Streamlit |
| Проверена origin/master | Удалённая master совпадает с исходной 978173c; пользовательские изменения остаются вне коммитов шага 5 |
| Повторён mutator по изменённым функциям и выжившим мутациям | Последняя мутация требовала проверки timeout первого вызова без ответа: уточнены тип ошибки, пустой текст и нулевой usage. Все 29/29 мутаций уничтожены, выживших и непокрытых sites нет; `.metrics/interface-step5-mutator-final.txt` |
| Проверены временные исполнители и worktree | Worktree один, mutator удалил временные workers; прежняя незакоммиченная работа сохранена |
| Сохранена реализация коммитом 834e112 | Только код, тесты и новый контракт шага 5; sandbox-отказ записи индекса устранён разрешённым повтором git add |

Доказательства: `.metrics/interface-step5-red.txt`,
`.metrics/interface-step5-boundaries-red.txt`,
`.metrics/interface-step5-targeted.txt`, `.metrics/interface-step5-tests.txt`,
`.metrics/interface-step5-types.txt`, `.metrics/interface-step5-syntax.txt`,
`.metrics/interface-step5-crapper.txt`, `.metrics/interface-step5-dryer.txt`,
`.metrics/interface-step5-mutator.txt`, `.metrics/interface-step5-mutator-final.txt`,
`target/coverage/python/lcov.info`.
Коммиты, отправка в master и сверка remote фиксируются в
`.metrics/interface-step5-delivery.txt` после коммита документации.
