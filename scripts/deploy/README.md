# Deployment tooling

The GitHub Actions workflow checks, builds and deploys the selected commit.
Deployment requires `ATRAG_DEPLOY_ENABLED=true` and a configured `production`
environment. SSH destinations and credentials come from GitHub variables and
secrets; production configuration is kept outside the repository.

Set `LIGHTSAIL_HOST`, `LIGHTSAIL_SSH_KEY` and `LIGHTSAIL_KNOWN_HOSTS` as secrets
in the `production` environment. Keep `LIGHTSAIL_SSH_USER`, `LIGHTSAIL_SSH_PORT`
and `ATRAG_DEPLOY_PATH` as environment variables. The host is a secret so GitHub
Actions masks its value in logs.

- `bootstrap.sh` installs the runtime, systemd unit and privileged control scripts
  on Ubuntu 24.04 with PostgreSQL 16.
- `setup_config.py` provisions the initial database and private configuration.
  It reads a JSON object from stdin with `openai_api_key`, `client_id`,
  `client_secret` and `public_host` (a DNS hostname, such as `tax.example.com`).
  The hostname sets both the Google OAuth callback and the Streamlit public
  address in `/opt/atrag/shared/service.env`.
- `configure_caddy.sh PUBLIC_HOST [EXISTING_URL EXPECTED_STATUS ...]` adds the
  reverse proxy configuration and verifies any supplied existing sites.
- `release.py`, `send.sh` and `server.py` package, deliver and activate checked
  commits, including database backups, health checks and compatible rollback.

For existing installations, retain the private database and OAuth configuration.
Before reinstalling the systemd unit, create a root-owned, mode `0640`
`/opt/atrag/shared/service.env` containing
`STREAMLIT_BROWSER_SERVER_ADDRESS=YOUR_PUBLIC_HOST`.
The root-owned server and control scripts are updated manually; CI updates
application releases only.
