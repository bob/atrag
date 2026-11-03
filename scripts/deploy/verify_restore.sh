#!/bin/bash
# Explicit one-time restore drill. The verification DB is created only if absent.
set -euo pipefail
export PATH=/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
[[ $(id -u) == 0 ]] || exit 1
sha=$(basename "$(readlink -f /opt/atrag/current)")
[[ $sha =~ ^[0-9a-f]{40}$ ]] || exit 64
database=atrag_restore_check_20261103
[[ $(runuser -u postgres -- psql -Atc "SELECT count(*) FROM pg_database WHERE datname='$database'") == 0 ]] || exit 1
created=false
cleanup() {
  if [[ $created == true ]]; then
    runuser -u postgres -- dropdb "$database"
  fi
}
trap cleanup EXIT
atrag-control backup "$sha"
runuser -u postgres -- createdb --template=template0 --owner=atrag_prod "$database"
created=true
runuser -u postgres -- pg_restore --exit-on-error --dbname="$database" < "/var/backups/atrag/$sha.dump"
[[ $(runuser -u postgres -- psql -At -d "$database" -c 'SELECT count(*) FROM legal_units') == 1789 ]]
[[ $(runuser -u postgres -- psql -At -d "$database" -c 'SELECT count(*) FROM search_fragments') == 1032 ]]
[[ $(runuser -u postgres -- psql -At -d "$database" -c 'SELECT max(v) FROM checkpoint_migrations') == 9 ]]
atrag-control discard-backup "$sha"
echo 'Restore verified: 1789 units, 1032 fragments, checkpoints v9; temporary dump removed.'
