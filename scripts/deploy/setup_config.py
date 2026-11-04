"""One-time root-only provisioning; credentials arrive as JSON over SSH stdin."""

import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

SHARED = Path("/opt/atrag/shared")


def sql(query, database="postgres"):
    return subprocess.check_output(
        [
            "runuser",
            "-u",
            "postgres",
            "--",
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-At",
            "-d",
            database,
        ],
        input=query,
        text=True,
    ).strip()


def main():
    if os.geteuid() != 0:
        raise RuntimeError("Run as root")
    shared = SHARED
    # Retries preserve the original DB password and OAuth cookie secret.
    if any(
        (shared / name).exists() for name in (".env", "secrets.toml", "service.env")
    ):
        raise RuntimeError(
            "Production configuration already exists; inspect rather than overwrite"
        )
    values = json.load(sys.stdin)
    for key in ("openai_api_key", "client_id", "client_secret", "public_host"):
        value = values.get(key)
        if not isinstance(value, str) or not value or any(c in value for c in "\r\n"):
            raise ValueError("Missing or invalid configuration value")
    host = values["public_host"]
    if len(host) > 253 or not re.fullmatch(
        r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}", host
    ):
        raise ValueError("public_host must be a DNS hostname without scheme or path")
    if sql("SELECT 1 FROM pg_roles WHERE rolname='atrag_prod'") or sql(
        "SELECT 1 FROM pg_database WHERE datname='atrag_prod'"
    ):
        raise RuntimeError(
            "ATRAG role/database already exists; inspect rather than overwrite"
        )
    password = secrets.token_hex(32)
    # Write the recoverable configuration before creating the role/database.
    environment = (
        f"DATABASE_URL=postgresql://atrag_prod:{password}@127.0.0.1:5432/atrag_prod\n"
        "ATRAG_DATA_DIR=/opt/atrag/shared/data\n"
        f"OPENAI_API_KEY={values['openai_api_key']}\n"
        "OPENAI_EMBEDDING_MODEL=text-embedding-3-large\n"
        "OPENAI_EMBEDDING_DIMENSIONS=1536\n"
        "ATRAG_CHAT_MODEL=gpt-6-luna\n"
    )
    q = json.dumps
    auth = (
        "[access]\ncheck_allowed_emails = false\nallowed_emails = []\n\n"
        f'[auth]\nredirect_uri = "https://{host}/oauth2callback"\n'
        f"cookie_secret = {q(secrets.token_urlsafe(48))}\n\n"
        "[auth.google]\n"
        f"client_id = {q(values['client_id'])}\n"
        f"client_secret = {q(values['client_secret'])}\n"
        'server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"\n'
    )
    import grp

    for name, content in (
        (".env", environment),
        ("secrets.toml", auth),
        ("service.env", f"STREAMLIT_BROWSER_SERVER_ADDRESS={host}\n"),
    ):
        fd = os.open(shared / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640)
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            os.fchown(handle.fileno(), 0, grp.getgrnam("atrag").gr_gid)
    sql(
        f"CREATE ROLE atrag_prod LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD '{password}';"
    )
    sql("CREATE DATABASE atrag_prod OWNER atrag_prod;")
    sql("CREATE EXTENSION IF NOT EXISTS vector;", "atrag_prod")
    # Restrict only the new role, preserving other applications' HBA rules.
    hba = Path(sql("SHOW hba_file"))
    rules = (
        "# ATRAG: allow its own database, reject access to other databases.\n"
        "host atrag_prod atrag_prod 127.0.0.1/32 scram-sha-256\n"
        "host atrag_prod atrag_prod ::1/128 scram-sha-256\n"
        "host all atrag_prod 0.0.0.0/0 reject\n"
        "host all atrag_prod ::/0 reject\n"
    )
    existing = hba.read_text()
    hba.write_text(rules + existing)
    errors = sql("SELECT count(*) FROM pg_hba_file_rules WHERE error IS NOT NULL")
    if errors != "0":
        hba.write_text(existing)
        raise RuntimeError("Invalid HBA configuration; original file restored")
    sql("SELECT pg_reload_conf()")
    print("Production configuration and isolated PostgreSQL role are ready")


if __name__ == "__main__":
    main()
