#!/bin/bash
# GitHub runner transport. All configuration is supplied through environment.
set -euo pipefail
umask 077
: "${LIGHTSAIL_HOST:?}" "${LIGHTSAIL_SSH_USER:?}" "${LIGHTSAIL_SSH_PORT:?}"
: "${LIGHTSAIL_SSH_KEY:?}" "${LIGHTSAIL_KNOWN_HOSTS:?}" "${GH_TOKEN:?}" "${GITHUB_SHA:?}"
[[ ${ATRAG_DEPLOY_PATH:?} == /opt/atrag ]] || exit 64
[[ $GITHUB_SHA =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ $LIGHTSAIL_HOST =~ ^[a-zA-Z0-9.-]+$ ]] || exit 64
[[ $LIGHTSAIL_SSH_USER == deploy ]] || exit 64
[[ $LIGHTSAIL_SSH_PORT =~ ^[0-9]{1,5}$ ]] || exit 64
temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
printf '%s\n' "$LIGHTSAIL_SSH_KEY" > "$temporary/key"
printf '%s\n' "$LIGHTSAIL_KNOWN_HOSTS" > "$temporary/known_hosts"
ssh_options=(-o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4
  -o "UserKnownHostsFile=$temporary/known_hosts" -i "$temporary/key")
destination=$LIGHTSAIL_SSH_USER@$LIGHTSAIL_HOST
tar -cf - -C artifact release.tar.gz release.sha256 |
  ssh "${ssh_options[@]}" -p "$LIGHTSAIL_SSH_PORT" "$destination" "upload $GITHUB_SHA"
if [[ ${ATRAG_TRANSPORT_ONLY:-false} == true ]]; then
  printf 'Delivered %s; activation was not requested.\n' "$GITHUB_SHA"
  exit 0
fi
# Token travels over encrypted stdin, never in process arguments or server files.
python -c 'import json,os; print(json.dumps({"token":os.environ["GH_TOKEN"]}))' |
  ssh "${ssh_options[@]}" -p "$LIGHTSAIL_SSH_PORT" "$destination" \
    "deploy $GITHUB_SHA"
