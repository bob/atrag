#!/bin/bash
# Usage: configure_caddy.sh PUBLIC_HOST [EXISTING_URL EXPECTED_STATUS ...]
# One-time root-only addition of the ATRAG site; preserve existing sites.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ $(id -u) == 0 ]] || exit 1
domain=${1:?Pass the public DNS hostname}
shift
[[ $domain =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]] || exit 64
(( $# % 2 == 0 )) || exit 64
checks=("$@")
for ((i=0; i<${#checks[@]}; i+=2)); do
  [[ ${checks[i]} =~ ^https?:// && ${checks[i+1]} =~ ^[1-5][0-9]{2}$ ]] || exit 64
done
file=/etc/caddy/Caddyfile
if grep -Fq -- "$domain" "$file"; then
  echo 'ATRAG site already exists; inspect instead of overwriting' >&2
  exit 1
fi
temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
cp -p "$file" "$temporary/original"
cp "$file" "$temporary/candidate"
cat >> "$temporary/candidate" <<CADDY

$domain {
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
for ((i=0; i<${#checks[@]}; i+=2)); do
  if [[ $(curl --silent --show-error --output /dev/null --write-out '%{http_code}' --max-time 20 "${checks[i]}") != "${checks[i+1]}" ]]; then
    restore
    echo 'Existing-site check failed; original Caddy configuration restored' >&2
    exit 1
  fi
done
echo 'Caddy updated; existing sites passed checks. ATRAG TLS awaits DNS.'
