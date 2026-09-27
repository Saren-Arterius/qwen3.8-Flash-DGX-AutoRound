#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")"
case "${1:---print}" in
  --print) mode=--print ;;
  --run) mode=--run ;;
  *) echo 'use --print or --run' >&2; exit 64 ;;
esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }
[ -r env ] && [ -r image ] && [ -r draft-vocab-ids-K65536.txt.gz ] || exit 65
image="$(tr -d '[:space:]' < image)"
case "$image" in *:latest|latest|*:|'') echo 'a fixed image tag is required' >&2; exit 66 ;; *:*) ;; *) exit 66 ;; esac
vocab="${HOME}/.cache/qwen38-v16b/draft-vocab-ids-K65536.txt"
if [ "$mode" = --run ]; then
  docker image inspect "$image" >/dev/null || { echo 'the measured image is unavailable; see repository docs/BUILD.md' >&2; exit 66; }
  mkdir -p "$(dirname "$vocab")"
  temp="$(mktemp "${vocab}.XXXXXX")"
  trap 'rm -f "$temp"' EXIT
  gzip -dc draft-vocab-ids-K65536.txt.gz > "$temp"
  [ "$(wc -l < "$temp")" -eq 65536 ] || { echo 'draft vocabulary has the wrong length' >&2; exit 65; }
  mv "$temp" "$vocab"
  trap - EXIT
fi
if [ "$mode" = --run ]; then set --; else set -- --print; fi
exec env -i HOME="$HOME" PATH="$PATH" IMAGE="$image" MTP_DRAFT_VOCAB="$vocab" bash -c '
  set -euo pipefail
  set -a; . ./env; set +a
  exec bash ./serve.sh "$@"
' launch.sh "$@"
