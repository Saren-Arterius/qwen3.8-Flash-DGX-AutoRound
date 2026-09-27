<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Build the v16b inputs

The promoted v16b launch uses the **T80 dense-MTP g32** checkpoint and the
`iter6d-20260910` image. Every Docker `COPY` input for the iter6c and iter6d
chain is included. The build starts from a pinned public image and public
model and table downloads.

## 1. Download the public checkpoints

Use a GB10 with ARM64 Linux, Docker with NVIDIA runtime, `hf`, Python 3, `jq`,
GNU `md5sum` and `sha256sum`, and NVMe storage for the downloads, Docker layers
and another 4.8 GiB for the T80 directory. The promoted memory configuration
is in [CONFIGURATION.md](CONFIGURATION.md). Check the downloaded model licenses before use.

```bash
mkdir -p "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid" "$HOME/models/ple-table-fp8"
hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid \
  --revision 8b82f0b7abe3d1150a7827d298c75e86267636ae \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid"
hf download Saren/Qwen3.8-Flash-Next-ple-table-fp8 \
  --revision 50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --local-dir "$HOME/models/ple-table-fp8"
```

The historical base checkpoint's `model.safetensors.index.json` is
23,813,217 bytes, SHA-256
`4da5d411d90f4b2d89d4e13fdf201516f6f877cb70a06aec3c1d4fd61509571f`.
Its `model_extra_tensors.safetensors` is 5,214,492,152 bytes, SHA-256
`e9e4786a8ef584c9cb112b0a2ce7063cb72db8e002694888d2a36d440edad83d`.
The T80 build wrapper checks both hashes before conversion. The checkpoint
contains 67 ordinary safetensors shards plus that extra shard. The PLE table
is downloaded separately and mounted read-only at launch.

## 2. Build the image

`build/image/Dockerfile.iter6c` follows the supplied full measured parent
Dockerfile, starting from the digest-pinned public vLLM preview image.
`build/image/Dockerfile.iter6d` adds the block-retention and R7 logger
patches. Their 53 and 5 active instruction lines, respectively, match the
supplied Dockerfiles. All 18 Docker `COPY` source files are present. Eleven
base-clone source files are checked by MD5 before the script pulls anything.
Ten are byte-identical to the staged clone. The eleventh,
`patch_measure_draft_mass.py`, has only comments and docstrings sanitized; its
release MD5 is `f9952df34874a2422ff8e291a772c2a9`, while its executable
Python AST, including the embedded runtime module, matches the staged original.
The byte-identical inputs include
`draft_vocab_common.py` (`dc14044033fb18da264967699289c547`),
`vllm_mtp_draft_vocab.py` (`c862ebecc496cdd4587461079c3d6f9b`) and
`patch_short_conv_async_h2d.py` (`c9f6b5ad6558f185a412e90a0f96cc19`).
Both Dockerfiles and their vLLM-derived patches remain Apache-2.0.

```bash
bash build/image/build.sh --print
bash build/image/build.sh --run
```

`--print` verifies the required source files and release hashes, then displays
the pull and two builds. `--run` executes them and runs the draft-vocabulary,
block-retention and iteration-6 CPU image gates. The pre-R7 GEMV module and
block-retention base manifest are also checked against
release SHA-256 values before Docker work. The GEMV module's comments and
docstrings were sanitized; its release SHA-256 is
`9fc2a904fded1d03589e88fcb9e33abe27443a5ad51c5420562d538249412140`
and its executable Python AST matches the staged original. The original
iter6c image was a local parent with ID beginning `sha256:302a7fd154c7`;
the measured iter6d image ID was
`sha256:fcb8086ccff2f2bb44ab4ff51881d30b13266d0eff6b9768e0d16201156a6716`.
Those IDs identify the recorded builds. Source comments and docstrings were
edited for publication; the executable Python AST was preserved.
The original Dockerfile chain includes the default-off measurement hook
`patch_measure_draft_mass.py` and step profiler; they remain because removing
those layers would change the measured lineage. The image uses the listed Docker `COPY` inputs.

The iter6d build report records a 14:01–14:04 UTC build-and-gate window,
88/88 block-retention CPU checks with the knob off and again with it on,
88/88 iteration-6 patch checks, and eight changed image paths versus iter6c.
It recorded SHA-256 `74d3d571b30353e0bd06a4fdcfe3268f158d317d925e75b15bc6145c0056a792`
for the original `patch_block_drop.py`, `a85ba0dc4b1258034415205e9f377b15a690507cdcaa0cb5135bbed87639ffba`
for its CPU test, and `82cb112d4c4a85919132c60c3a7206687e935a9a198a9bdec55cd332cb2ff83c`
for the base-file hash manifest. The release copies of these later patch and
test scripts have publication-edited docstrings, so their whole-file hashes differ while executable Python AST is
unchanged. The release pins above apply to the publication bytes.
`patch_ple_rendezvous.py` is covered by the AST comparison and image tests.

## 3. Build and verify the T80 directory

The T80 builder is present and its CPU tests pass in this checkout. It uses
the newly built iter6d image as its CPU helper. The source and destination must be
siblings under the same `MODELS_ROOT` so unchanged shards can be hardlinked.

```bash
bash build/model/build.sh --print
bash build/model/build.sh --run
```

The wrapper invokes `--tier drafter-dense --group-size 32`. Do not use only
`--mtp-tier drafter-dense`: that composes the drafter conversion with the
default target-side tier and changes the served checkpoint. The builder
converts nine bf16 drafter modules in `model_extra_tensors.safetensors`,
retains target routing, verifies the resulting safetensors/index, and links
the unchanged shards. The generated report is `dense-mtp-build-report.json`
inside the output directory. The historical reference report is
[reference-build-report.json](../build/model/reference-build-report.json).

| Recorded T80 check | Reference value |
|---|---:|
| Shards rewritten / linked | 1 / 67 |
| Extra shard before / after | 5,214,492,152 / 5,118,048,680 bytes |
| New allocated directory bytes | 5,118,048,680 bytes |
| Converted drafter / target modules | 9 / 0 |
| Conversion wall time in report | 4.626 s |
| Verification | `VERIFY OK`; 68 shards, 340 copied tensors byte-identical, zero degenerate groups |

The wrapper checks the two pinned input hashes and the report invariants;
`--verify-only` checks the produced directory.

## 4. Launch and test

With both builds complete:

```bash
bash config/v16b/launch.sh --print
bash config/v16b/launch.sh --run
curl -fsS http://localhost:8000/v1/models
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":512}'
```

The launcher uses the promoted `GPU_MEM=0.01`, explicit `KV_BYTES=16g`,
`SEQS=8`, `CTX=262144`, MTP depth 3 and prefix caching. See
[BENCHMARKS.md](BENCHMARKS.md) and [EVALS.md](EVALS.md) for the measured results.
