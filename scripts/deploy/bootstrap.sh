#!/bin/bash
# Run once as root from a reviewed copy of these files, on Ubuntu 24.04 / PG16.
# Does not modify Caddy, OAuth, DNS, application secrets or existing databases.
set -euo pipefail
export PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l
[[ $(id -u) == 0 ]] || { echo 'Run as root' >&2; exit 1; }
source /etc/os-release
[[ $VERSION_ID == 24.04 && $(uname -m) == x86_64 ]] || exit 1
[[ $(runuser -u postgres -- psql -Atc 'SHOW server_version_num') == 16* ]] || exit 1
id deploy > /dev/null
if ss -lnt '( sport = :8501 )' | tail -n +2 | grep -q .; then
  systemctl is-active --quiet atrag.service || { echo 'Port 8501 is occupied' >&2; exit 1; }
fi
source_dir=$(cd -- "$(dirname -- "$0")/../.." && pwd)
apt-get update -qq
apt-get install -y postgresql-16-pgvector curl ca-certificates
id atrag > /dev/null 2>&1 || useradd --system --user-group --home-dir /opt/atrag/shared/data --shell /usr/sbin/nologin atrag
install -d -o deploy -g atrag -m 2755 /opt/atrag /opt/atrag/releases /opt/atrag/incoming /opt/atrag/python
install -d -o root -g atrag -m 0750 /opt/atrag/shared /opt/atrag/shared/cache /opt/atrag/shared/packages
install -d -o atrag -g atrag -m 0750 /opt/atrag/shared/data /opt/atrag/shared/cache/tiktoken
install -d -o root -g atrag -m 0750 /var/backups/atrag
for name in corpus-changes.json corpus-expected.json; do
  if [[ ! -e /opt/atrag/shared/$name ]]; then
    install -o atrag -g atrag -m 0640 /dev/null "/opt/atrag/shared/$name"
  fi
done
temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
if [[ ! -x /usr/local/bin/uv ]] || [[ $(/usr/local/bin/uv --version | cut -d ' ' -f 2) != 0.12.1 ]]; then
  cd "$temporary"
  asset=uv-x86_64-unknown-linux-gnu.tar.gz
  curl --fail --silent --show-error --location --retry 3 -o "$asset" "https://github.com/astral-sh/uv/releases/download/0.12.1/$asset"
  curl --fail --silent --show-error --location --retry 3 -o "$asset.sha256" "https://github.com/astral-sh/uv/releases/download/0.12.1/$asset.sha256"
  sha256sum --check "$asset.sha256"
  tar -xzf "$asset"
  install -o root -g root -m 0755 uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv
fi
cd /
runuser -u deploy -- env UV_PYTHON_INSTALL_DIR=/opt/atrag/python /usr/local/bin/uv python install 3.14.6
install -o root -g root -m 0755 "$source_dir/scripts/deploy/atrag-control" /usr/local/sbin/atrag-control
install -o root -g root -m 0644 "$source_dir/docs/deploy/atrag.service" /etc/systemd/system/atrag.service
install -d -o root -g root -m 0755 /usr/local/lib/atrag
for name in server.py release.py receive.py; do
  install -o root -g root -m 0644 "$source_dir/scripts/deploy/$name" "/usr/local/lib/atrag/$name"
done
install -o root -g root -m 0755 "$source_dir/scripts/deploy/atrag-ssh" /usr/local/bin/atrag-ssh
printf '%s\n' 'deploy ALL=(root) NOPASSWD: /usr/local/sbin/atrag-control *' > "$temporary/sudoers"
visudo -cf "$temporary/sudoers"
install -o root -g root -m 0440 "$temporary/sudoers" /etc/sudoers.d/atrag
systemctl daemon-reload
systemctl enable atrag.service
echo 'Runtime ready. Configure secrets and atrag_prod before the first deployment.'
