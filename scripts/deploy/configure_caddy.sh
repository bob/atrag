#!/bin/bash
# One-time root-only addition of the ATRAG site; preserve both existing sites.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ $(id -u) == 0 ]] || exit 1
file=/etc/caddy/Caddyfile
if grep -q 'tax.example.com' "$file"; then
  echo 'ATRAG site already exists; inspect instead of overwriting' >&2
  exit 1
fi
temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
cp -p "$file" "$temporary/original"
cp "$file" "$temporary/candidate"
cat >> "$temporary/candidate" <<'CADDY'

tax.example.com {
    reverse_proxy 127.0.0.1:8501
}
CADDY
caddy validate --config "$temporary/candidate" --adapter caddyfile
restore() {
  cp -p "$temporary/original" "$file"
  systemctl reload caddy
}
cp "$temporary/candidate" "$file"
if ! systemctl reload caddy; then
  restore
  exit 1
fi
if [[ $(curl --silent --show-error --output /dev/null --write-out '%{http_code}' --max-time 20 https://example.com) != 200 ]] ||
   [[ $(curl --silent --show-error --output /dev/null --write-out '%{http_code}' --max-time 20 https://ai.example.com) != 401 ]]; then
  restore
  echo 'Existing-site check failed; original Caddy configuration restored' >&2
  exit 1
fi
echo 'Caddy updated; existing sites passed checks. ATRAG TLS awaits DNS.'
