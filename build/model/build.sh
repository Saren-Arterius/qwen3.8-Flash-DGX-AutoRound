#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reproduce the T80 dense-MTP g32 directory from a pinned public base.
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")" && pwd)"
models_root="$(realpath "${MODELS_ROOT:-$HOME/models}")"
model_dir="$(realpath "${MODEL_DIR:-$models_root/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid}")"
out_dir="$(realpath -m "${OUT_DIR:-$models_root/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32}")"
image="${IMAGE:-qwen38-flash-dgx:iter6d-20260910}"

[ -d "$model_dir" ] || { echo "missing base checkpoint: $model_dir" >&2; exit 66; }
[ "$(dirname "$model_dir")" = "$models_root" ] || { echo 'MODEL_DIR must be a direct child of MODELS_ROOT' >&2; exit 64; }
[ "$(dirname "$out_dir")" = "$models_root" ] || { echo 'OUT_DIR must be a direct child of MODELS_ROOT' >&2; exit 64; }
[ "$model_dir" != "$out_dir" ] || { echo 'source and output must differ' >&2; exit 64; }

base="$(basename "$model_dir")"
output="$(basename "$out_dir")"
docker_common=(docker run --rm --network none --memory 8g --ipc=host
  --user "$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1
  -v "$here:/work:ro" -v "$models_root:/models" -w /work
  --entrypoint python3 "$image")
build_args=(/work/build_int4side_model_dir.py --tier drafter-dense --group-size 32
  --model-dir "/models/$base" --out-dir "/models/$output")
verify_args=(/work/build_int4side_model_dir.py --tier drafter-dense --group-size 32
  --verify-only --model-dir "/models/$base" --out-dir "/models/$output")

if [ "$mode" = --print ]; then
  printf 'Build: '; printf '%q ' "${docker_common[@]}" "${build_args[@]}"; printf '\n'
  printf 'Verify: '; printf '%q ' "${docker_common[@]}" "${verify_args[@]}"; printf '\n'
  exit 0
fi

[ ! -e "$out_dir" ] || { echo "output already exists: $out_dir" >&2; exit 66; }
docker image inspect "$image" >/dev/null || { echo "missing helper image: $image" >&2; exit 66; }
[ -f "$model_dir/config.json" ] && [ -f "$model_dir/model.safetensors.index.json" ] && \
  [ -f "$model_dir/model_extra_tensors.safetensors" ] || {
  echo 'the base checkpoint is incomplete' >&2; exit 66;
}
printf '%s  %s\n' \
  '4da5d411d90f4b2d89d4e13fdf201516f6f877cb70a06aec3c1d4fd61509571f' \
  "$model_dir/model.safetensors.index.json" \
  'e9e4786a8ef584c9cb112b0a2ce7063cb72db8e002694888d2a36d440edad83d' \
  "$model_dir/model_extra_tensors.safetensors" | sha256sum --check --status || {
  echo 'base checkpoint does not match the pinned public inputs' >&2; exit 65;
}

"${docker_common[@]}" /work/test_rtn_int4_gptq.py
"${docker_common[@]}" /work/test_build_int4side.py
"${docker_common[@]}" /work/test_build_mtpdense.py
"${docker_common[@]}" "${build_args[@]}"
"${docker_common[@]}" "${verify_args[@]}"

if find "$out_dir" -maxdepth 1 -type l -print -quit | grep -q .; then
  echo 'output has symlinks; the serving bind mount requires hardlinks' >&2
  exit 1
fi
python3 - "$out_dir/dense-mtp-build-report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['tier'] == 'drafter-dense' and r['int4_tier'] == ''
assert r['mtp_tier'] == 'drafter-dense' and r['group_size'] == 32
assert len(r['mtp_modules']) == 9 and not r['modules']
assert r['mtp_totals']['degenerate_groups'] == 0
print('Dense-MTP report: nine drafter modules at g32; zero target modules')
PY
echo "Verified dense-MTP g32 directory: $out_dir"
