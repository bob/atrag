"""Administrative presentation; all authorization and data live in services."""

from datetime import UTC, datetime

import streamlit as st

from atrag.admin.schemas import DocumentVersion, UserSummary
from atrag.admin.service import AdminService
from atrag.interface.locales import tr
from atrag.schemas.chat.conversations import User
from atrag.schemas.settings import SupportedLanguage


def date_label(value: datetime | None, language: SupportedLanguage) -> str:
    if value is None:
        return tr("no_data", language)
    return f"{value.astimezone(UTC):%Y-%m-%d %H:%M} UTC"


def size_label(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def users(service: AdminService, actor: User) -> None:
    language = actor.language
    # Authorize before showing even an empty administrative page.
    with service.authorized(actor):
        pass
    st.subheader("Users")
    st.caption(tr("recorded_tokens", language))
    search = st.text_input(tr("email_search", language), key="admin_email_search")
    status = st.segmented_control(
        tr("status", language),
        ("all", "active", "inactive"),
        default="all",
        key="admin_status",
        format_func=lambda value: tr(value, language),
    )
    filters = (search, status)
    if st.session_state.get("admin_user_filters") != filters:
        st.session_state["admin_user_filters"] = filters
        st.session_state["admin_user_page"] = 1
    page = int(
        st.number_input(
            tr("page", language), min_value=1, step=1, key="admin_user_page"
        )
    )
    rows = service.list_users(
        actor,
        search=search,
        active={"active": True, "inactive": False}.get(status),
        limit=25,
        offset=(page - 1) * 25,
    )
    data = [
        {
            "Email": row.email,
            tr("registered", language): date_label(row.created_at, language),
            tr("last_login", language): date_label(row.last_login_at, language),
            tr("status", language): tr(
                "active" if row.is_active else "inactive", language
            ),
            "LLM input": row.input_tokens,
            "LLM output": row.output_tokens,
            "LLM total": row.total_tokens,
            "Actions": tr("deactivate" if row.is_active else "activate", language),
        }
        for row in rows
    ]
    st.dataframe(
        data,
        hide_index=True,
        alt=tr("users_table", language),
        column_config={
            "Actions": st.column_config.ButtonColumn(
                "Actions",
                key="admin_user_action",
                on_click=select_user_action,
                args=(rows,),
                width="medium",
            )
        },
    )
    if not rows:
        st.info(tr("no_data", language))
        return
    if pending := st.session_state.pop("admin_pending_user", None):
        show_user_confirmation(service, actor, pending)


def select_user_action(rows: tuple[UserSummary, ...]) -> None:
    click = st.session_state["admin_user_action"]
    st.session_state["admin_pending_user"] = rows[click.row]


def show_user_confirmation(
    service: AdminService, actor: User, row: UserSummary
) -> None:
    st.dialog(tr("confirm_action", actor.language))(confirm_user_status)(
        service, actor, row
    )


def confirm_user_status(service: AdminService, actor: User, row: UserSummary) -> None:
    language = actor.language
    action = "deactivate" if row.is_active else "activate"
    st.text(f"{tr(action, language)}: {row.email}?")
    own_account = row.id == actor.id
    if own_account:
        st.info(tr("cannot_deactivate_self", language))
    with st.container(horizontal=True):
        if st.button(
            tr("confirm", language),
            key="admin_confirm_status",
            type="primary",
            disabled=own_account,
        ):
            try:
                service.set_user_active(actor, row.id, not row.is_active)
            except PermissionError:
                st.error(tr("denied", language))
                return
            st.rerun()
        if st.button(tr("cancel", language), key="admin_cancel_status"):
            st.rerun()


def documents(service: AdminService, actor: User) -> None:
    rows = service.list_documents(actor)
    language = actor.language
    st.subheader("Documents")
    st.dataframe(
        [
            {
                tr("document", language): row.document_title,
                tr("uploaded", language): date_label(row.indexed_at, language),
                tr("format", language): row.source_format.replace(
                    "html_collection", "HTML"
                ).replace("pdf", "PDF"),
                tr("open_source", language): row.versions[0].source_url,
                tr("source_size", language): size_label(row.size_bytes),
                tr("published", language): row.published,
            }
            for row in rows
        ],
        hide_index=True,
        alt=tr("documents_table", language),
        column_config={
            tr("open_source", language): st.column_config.LinkColumn(width="small"),
            tr("document", language): st.column_config.TextColumn(width="medium"),
        },
    )
    if not rows:
        st.info(tr("no_data", language))
        return
    by_id = {row.source_id: row for row in rows}
    selected = st.selectbox(
        tr("document", language),
        tuple(by_id),
        format_func=lambda key: by_id[key].document_title,
        key="admin_document",
    )
    for version in by_id[selected].versions:
        version_view(service, actor, version)


def version_view(service: AdminService, actor: User, version: DocumentVersion) -> None:
    language = actor.language
    with st.expander(
        f"{date_label(version.indexed_at, language)} · {version.snapshot_id[:12]}"
    ):
        st.text(version.snapshot_id)
        st.caption(
            f"{tr('prepared', language)}: {date_label(version.ingested_at, language)}"
        )
        st.caption(
            f"{tr('uploaded', language)}: {date_label(version.indexed_at, language)}"
        )
        st.caption(f"{version.source_format} · {size_label(version.size_bytes)}")
        st.text(tr("published" if version.published else "unpublished", language))
        if version.source_url:
            st.link_button(tr("open_source", language), version.source_url)
        if not version.available:
            st.warning(tr("source_missing", language))
            return
        is_pdf = version.source_format == "pdf"
        st.download_button(
            tr("download_source", language),
            data=lambda: service.download_source(actor, version.snapshot_id).data,
            file_name=f"{version.source_sha256}.{'pdf' if is_pdf else 'zip'}",
            mime="application/pdf" if is_pdf else "application/zip",
            key=f"source_download_{version.snapshot_id}",
            on_click="ignore",
        )
