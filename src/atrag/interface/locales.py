"""Presentation labels in Portuguese, English, Ukrainian and Russian."""

from atrag.schemas.settings import SupportedLanguage

LANGUAGES: tuple[SupportedLanguage, ...] = ("pt", "en", "uk", "ru")
LANGUAGE_NAMES = {
    "pt": "Português",
    "en": "English",
    "uk": "Українська",
    "ru": "Русский",
}
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
    "release": (
        "Versão do corpus",
        "Corpus release",
        "Склад корпусу",
        "Состав корпуса",
    ),
    "version": ("Versão", "Version", "Версія", "Версия"),
    "empty_corpus": (
        "Ainda não há documentos publicados.",
        "No documents have been published yet.",
        "У корпусі ще немає документів.",
        "В корпусе пока нет документов.",
    ),
    "sources": ("Fontes", "Sources", "Джерела", "Источники"),
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
