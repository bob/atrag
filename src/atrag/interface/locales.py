"""Presentation labels in Portuguese, English, Ukrainian and Russian."""

from atrag.schemas.settings import SupportedLanguage

LANGUAGES: tuple[SupportedLanguage, ...] = ("pt", "en", "uk", "ru")
LABELS = {
    "subtitle": (
        "Perguntas sobre documentos",
        "Questions about documents",
        "Питання щодо документів",
        "Вопросы по документам",
    ),
    "language": ("Idioma", "Language", "Мова", "Язык"),
    "login": (
        "Entrar com Google",
        "Sign in with Google",
        "Увійти через Google",
        "Войти через Google",
    ),
    "login_hint": (
        "Entre para abrir as suas conversas.",
        "Sign in to open your conversations.",
        "Увійдіть, щоб відкрити свої діалоги.",
        "Войдите, чтобы открыть свои диалоги.",
    ),
    "logout": ("Sair", "Sign out", "Вийти", "Выйти"),
    "denied": (
        "Acesso negado",
        "Access denied",
        "Доступ заборонено",
        "Доступ запрещён",
    ),
    "new": ("Nova conversa", "New conversation", "Новий діалог", "Новый диалог"),
    "conversations": ("Conversas", "Conversations", "Діалоги", "Диалоги"),
    "untitled": ("Sem título", "Untitled", "Без назви", "Без названия"),
    "choose": (
        "Crie ou selecione uma conversa.",
        "Create or select a conversation.",
        "Створіть або виберіть діалог.",
        "Создайте или выберите диалог.",
    ),
    "question": (
        "Faça uma pergunta",
        "Ask a question",
        "Поставте запитання",
        "Задайте вопрос",
    ),
    "corpus": (
        "Documentos publicados",
        "Published documents",
        "Опубліковані документи",
        "Опубликованные документы",
    ),
    "ingested_at": (
        "Data de ingestão",
        "Ingestion date",
        "Дата завантаження",
        "Дата загрузки",
    ),
    "empty_corpus": (
        "Ainda não há documentos publicados.",
        "No documents have been published yet.",
        "У корпусі ще немає документів.",
        "В корпусе пока нет документов.",
    ),
    "sources": ("Fontes", "Sources", "Джерела", "Источники"),
    "law": ("Lei", "Law", "Закон", "Закон"),
    "faq": (
        "Esclarecimento da AT",
        "AT explanation",
        "Роз’яснення AT",
        "Разъяснение AT",
    ),
    "at_instruction": (
        "Instrução da AT",
        "AT instruction",
        "Інструкція AT",
        "Инструкция AT",
    ),
    "individual_ruling": (
        "Informação vinculativa",
        "Individual ruling",
        "Індивідуальне роз’яснення",
        "Индивидуальное разъяснение",
    ),
    "open_source": (
        "Abrir fonte oficial",
        "Open official source",
        "Відкрити офіційне джерело",
        "Открыть официальный источник",
    ),
    "candidate_source": (
        "O PDF oficial pode ter outra redação. A citação corresponde à versão guardada abaixo; a identidade do ficheiro remoto não está confirmada.",
        "The official PDF may have a different revision. The quotation belongs to the saved version below; the remote file's identity is unconfirmed.",
        "Офіційний PDF може мати іншу редакцію. Цитата належить збереженій нижче версії; ідентичність віддаленого файлу не підтверджено.",
        "Официальный PDF может иметь другую редакцию. Цитата относится к сохранённой ниже версии; идентичность удалённого файла не подтверждена.",
    ),
    "live_source": (
        "A página oficial pode mudar. A citação e a versão guardadas abaixo são preservadas no histórico.",
        "The official page can change. The quotation and version saved below remain in the history.",
        "Офіційна сторінка може змінюватися. Збережені нижче цитата та версія залишаються в історії.",
        "Официальная страница может меняться. Сохранённые ниже цитата и версия остаются в истории.",
    ),
    "location": ("Localização", "Location", "Розташування", "Местоположение"),
    "pdf": ("Página", "Page", "Сторінка", "Страница"),
    "section": ("Secção", "Section", "Розділ", "Раздел"),
    "anchor": ("Âncora", "Anchor", "Якір", "Якорь"),
    "paragraph": ("Parágrafo", "Paragraph", "Абзац", "Абзац"),
    "revision": ("Redação", "Revision", "Редакція", "Редакция"),
    "notes": ("Notas", "Notes", "Зауваження", "Замечания"),
    "incomplete": (
        "O texto obtido está incompleto.",
        "The retrieved text is incomplete.",
        "Отримано неповний текст.",
        "Получен неполный текст.",
    ),
    "preparing": (
        "A preparar a resposta",
        "Preparing an answer",
        "Підготовка відповіді",
        "Подготовка ответа",
    ),
    "search": (
        "A pesquisar documentos",
        "Searching documents",
        "Пошук документів",
        "Поиск документов",
    ),
    "reading": (
        "A ler o documento",
        "Reading material",
        "Читання матеріалу",
        "Чтение материала",
    ),
    "done": ("Concluído", "Done", "Готово", "Готово"),
    "resume": ("Continuar", "Continue", "Продовжити", "Продолжить"),
    "execution_failed": (
        "A resposta ficou incompleta. Continue a partir do estado guardado.",
        "The answer is unfinished. Continue from the saved state.",
        "Відповідь не завершено. Продовжте зі збереженого стану.",
        "Ответ не завершён. Продолжите с сохранённого состояния.",
    ),
    "conversation_busy": (
        "Já está em curso um pedido nesta conversa. Atualize a página mais tarde.",
        "A request is already running in this conversation. Refresh the page later.",
        "У цьому діалозі вже виконується запит. Оновіть сторінку пізніше.",
        "В этом диалоге уже выполняется запрос. Обновите страницу позже.",
    ),
    "resume_required": (
        "Selecione Continuar para retomar a resposta incompleta.",
        "Select Continue to resume the unfinished answer.",
        "Щоб продовжити незавершену відповідь, натисніть «Продовжити».",
        "Для незавершённого ответа нажмите «Продолжить».",
    ),
    "unfinished_turn": (
        "Continue primeiro a resposta incompleta.",
        "Continue the unfinished answer first.",
        "Спочатку продовжте незавершену відповідь.",
        "Сначала продолжите незавершённый ответ.",
    ),
    "turn_not_found": (
        "O pedido não foi encontrado. Atualize a página.",
        "The request was not found. Refresh the page.",
        "Запит не знайдено. Оновіть сторінку.",
        "Запрос не найден. Обновите страницу.",
    ),
    "unavailable": (
        "O serviço está indisponível. Verifique a configuração e a ligação.",
        "The service is unavailable. Check settings and connectivity.",
        "Сервіс недоступний. Перевірте налаштування та підключення.",
        "Сервис недоступен. Проверьте настройки и подключение.",
    ),
    "auth_setup": (
        "Configure o Google OIDC em .streamlit/secrets.toml.",
        "Configure Google OIDC in .streamlit/secrets.toml.",
        "Налаштуйте Google OIDC у .streamlit/secrets.toml.",
        "Настройте Google OIDC в .streamlit/secrets.toml.",
    ),
}


def tr(key: str, language: SupportedLanguage) -> str:
    return LABELS[key][LANGUAGES.index(language)]
