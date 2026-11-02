"""Presentation labels in Russian, English and Portuguese; evidence is unchanged."""

from atrag.schemas.settings import SupportedLanguage

LANGUAGES: tuple[SupportedLanguage, ...] = ("ru", "en", "pt")
LANGUAGE_NAMES = {"ru": "Русский", "en": "English", "pt": "Português"}
LABELS = {
    "subtitle": (
        "Вопросы по документам",
        "Questions about documents",
        "Perguntas sobre documentos",
    ),
    "language": ("Язык", "Language", "Idioma"),
    "login": ("Войти через Google", "Sign in with Google", "Entrar com Google"),
    "login_hint": (
        "Войдите, чтобы открыть свои диалоги.",
        "Sign in to open your conversations.",
        "Entre para abrir as suas conversas.",
    ),
    "logout": ("Выйти", "Sign out", "Sair"),
    "denied": ("Доступ запрещён", "Access denied", "Acesso negado"),
    "new": ("Новый диалог", "New conversation", "Nova conversa"),
    "conversations": ("Диалоги", "Conversations", "Conversas"),
    "untitled": ("Без названия", "Untitled", "Sem título"),
    "choose": (
        "Создайте или выберите диалог.",
        "Create or select a conversation.",
        "Crie ou selecione uma conversa.",
    ),
    "question": ("Задайте вопрос", "Ask a question", "Faça uma pergunta"),
    "corpus": (
        "Опубликованные документы",
        "Published documents",
        "Documentos publicados",
    ),
    "release": ("Состав корпуса", "Corpus release", "Versão do corpus"),
    "version": ("Версия", "Version", "Versão"),
    "empty_corpus": (
        "В корпусе пока нет документов.",
        "No documents have been published yet.",
        "Ainda não há documentos publicados.",
    ),
    "sources": ("Источники", "Sources", "Fontes"),
    "location": ("Местоположение", "Location", "Localização"),
    "pdf": ("Страница", "Page", "Página"),
    "section": ("Раздел", "Section", "Secção"),
    "anchor": ("Якорь", "Anchor", "Âncora"),
    "paragraph": ("Абзац", "Paragraph", "Parágrafo"),
    "revision": ("Редакция", "Revision", "Redação"),
    "notes": ("Замечания", "Notes", "Notas"),
    "incomplete": (
        "Получен неполный текст.",
        "The retrieved text is incomplete.",
        "O texto obtido está incompleto.",
    ),
    "preparing": ("Подготовка ответа", "Preparing an answer", "A preparar a resposta"),
    "search": ("Поиск документов", "Searching documents", "A pesquisar documentos"),
    "reading": ("Чтение материала", "Reading material", "A ler o documento"),
    "done": ("Готово", "Done", "Concluído"),
    "resume": ("Продолжить", "Continue", "Continuar"),
    "execution_failed": (
        "Ответ не завершён. Продолжите с сохранённого состояния.",
        "The answer is unfinished. Continue from the saved state.",
        "A resposta ficou incompleta. Continue a partir do estado guardado.",
    ),
    "conversation_busy": (
        "В этом диалоге уже выполняется запрос. Обновите страницу позже.",
        "A request is already running in this conversation. Refresh the page later.",
        "Já está em curso um pedido nesta conversa. Atualize a página mais tarde.",
    ),
    "resume_required": (
        "Для незавершённого ответа нажмите «Продолжить».",
        "Select Continue to resume the unfinished answer.",
        "Selecione Continuar para retomar a resposta incompleta.",
    ),
    "unfinished_turn": (
        "Сначала продолжите незавершённый ответ.",
        "Continue the unfinished answer first.",
        "Continue primeiro a resposta incompleta.",
    ),
    "turn_not_found": (
        "Запрос не найден. Обновите страницу.",
        "The request was not found. Refresh the page.",
        "O pedido não foi encontrado. Atualize a página.",
    ),
    "unavailable": (
        "Сервис недоступен. Проверьте настройки и подключение.",
        "The service is unavailable. Check settings and connectivity.",
        "O serviço está indisponível. Verifique a configuração e a ligação.",
    ),
    "auth_setup": (
        "Настройте Google OIDC в .streamlit/secrets.toml.",
        "Configure Google OIDC in .streamlit/secrets.toml.",
        "Configure o Google OIDC em .streamlit/secrets.toml.",
    ),
}


def tr(key: str, language: SupportedLanguage) -> str:
    return LABELS[key][LANGUAGES.index(language)]
