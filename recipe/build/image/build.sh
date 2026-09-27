#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the measured image lineage from the pinned public parent image.
set -euo pipefail

mode="${1:---print}"
case "$mode" in --print|--run) ;; *) echo 'use --print or --run' >&2; exit 64 ;; esac
[ "$#" -le 1 ] || { echo 'too many arguments' >&2; exit 64; }

here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/Dockerfile.iter6c" ] && [ -f "$here/Dockerfile.iter6d" ] || {
  echo 'missing image Dockerfile' >&2; exit 66;
}
parent='vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8'
iter6c='qwen38-flash-dgx:iter6c-20260909'
iter6d='qwen38-flash-dgx:iter6d-20260910'

# The original iter6c build context referenced these files. Fail before
# pulling or building anything if the exact snapshot is not present.
missing=()
while IFS= read -r name; do
  [ -f "$here/src/$name" ] || missing+=("$name")
done < <(sed -nE 's/^COPY src\/([^ ]+) .*/\1/p' "$here/Dockerfile.iter6c" "$here/Dockerfile.iter6d" | sort -u)

if ((${#missing[@]})); then
  printf 'Missing iter6c source input(s):\n' >&2
  printf '  recipe/build/image/src/%s\n' "${missing[@]}" >&2
  echo 'The measured iter6d image cannot be rebuilt from this release yet.' >&2
  exit 66
fi
for name in test_draft_vocab_cpu.py test_block_drop_cpu.py test_iter6_patches_image.py blockdrop_base_sha256.json; do
  [ -f "$here/tests/$name" ] || { echo "missing image check: recipe/build/image/tests/$name" >&2; exit 66; }
done

# Pinned release inputs: ten exact clone copies and one comment-sanitized copy.
# Later image patches are checked by their Dockerfile anchors and CPU gates.
if ! (cd "$here" && md5sum --check --status <<'MD5'
dc14044033fb18da264967699289c547  src/draft_vocab_common.py
c862ebecc496cdd4587461079c3d6f9b  src/vllm_mtp_draft_vocab.py
c9f6b5ad6558f185a412e90a0f96cc19  src/patch_short_conv_async_h2d.py
60797774c6cf7803d6fabaf2ae8d6151  src/vllm_ple_mmap.py
9bb9d0cebc756fc6be7daab184579c09  src/vllm_fp8_hybrid.py
d4cfbfd3f30768d45e2b3a87e5b95b05  src/patch_never_evict.py
3edc482ca8f18e0d94a7ad82d23db867  src/mamba_utils_guarded.py
852f3404ed5578e2e55cfd830223dc3a  src/patch_hit_debug.py
f2fddcf38ce88c0f0535f51a1858d8c9  src/patch_mamba_align_split.py
37927ebc50d9989c91d95f4734f0902b  src/patch_step_profile.py
f9952df34874a2422ff8e291a772c2a9  src/patch_measure_draft_mass.py
MD5
); then
  echo 'release image source hash mismatch; refusing image build' >&2
  exit 65
fi
if ! (cd "$here" && sha256sum --check --status <<'SHA256'
9fc2a904fded1d03589e88fcb9e33abe27443a5ad51c5420562d538249412140  src/vllm_inproj_ba_gemv.py
82cb112d4c4a85919132c60c3a7206687e935a9a198a9bdec55cd332cb2ff83c  tests/blockdrop_base_sha256.json
SHA256
); then
  echo 'release image SHA-256 mismatch; refusing image build' >&2
  exit 65
fi

commands=(
  "docker pull '$parent'"
  "docker build --network none --pull=false -f '$here/Dockerfile.iter6c' -t '$iter6c' '$here'"
  "docker build --network none --pull=false -f '$here/Dockerfile.iter6d' -t '$iter6d' '$here'"
)
if [ "$mode" = --print ]; then
  printf '%s\n' "${commands[@]}"
  exit 0
fi

docker pull "$parent"
docker build --network none --pull=false -f "$here/Dockerfile.iter6c" -t "$iter6c" "$here"
docker build --network none --pull=false -f "$here/Dockerfile.iter6d" -t "$iter6d" "$here"

docker run --rm --network none --memory 8g \
  -v "$here/tests/test_draft_vocab_cpu.py:/tmp/t.py:ro" \
  --entrypoint python3 "$iter6d" /tmp/t.py
for knob in unset set; do
  extra=()
  [ "$knob" = set ] && extra=(-e VLLM_KEEP_DRAFT_BLOCKS=1)
  docker run --rm --network none --memory 8g "${extra[@]}" \
    -v "$here/tests/test_block_drop_cpu.py:/tmp/t.py:ro" \
    -v "$here/src/patch_block_drop.py:/tmp/patch_block_drop.py:ro" \
    -v "$here/tests/blockdrop_base_sha256.json:/tmp/base.json:ro" \
    --entrypoint python3 "$iter6d" /tmp/t.py
done
docker run --rm --network none --memory 8g \
  -v "$here/tests/test_iter6_patches_image.py:/tmp/t.py:ro" \
  -v "$here/src:/selftest/src:ro" \
  --entrypoint python3 "$iter6d" /tmp/t.py
docker run --rm --network none --memory 8g --entrypoint python3 "$iter6d" \
  -c 'import vllm_inproj_ba_gemv as g; assert g.logger.name == "vllm.inproj_ba_gemv"; assert not g.is_enabled()'
docker run --rm --network none --memory 8g -e VLLM_INPROJ_BA_GEMV=1 \
  --entrypoint python3 "$iter6d" \
  -c 'import vllm_inproj_ba_gemv as g; assert g.logger.name == "vllm.inproj_ba_gemv"; assert g.is_enabled()'
docker image inspect "$iter6d" --format '{{.Id}}'
