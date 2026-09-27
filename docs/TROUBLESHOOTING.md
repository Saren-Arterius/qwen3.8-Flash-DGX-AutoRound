<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Troubleshooting

| Symptom | Check |
|---|---|
| `launch.sh --run` says image unavailable | Follow [BUILD.md](BUILD.md) to build the iter6c and iter6d images, then inspect the first failed build or CPU gate. The root base image alone is not the promoted image. |
| Model or table mount missing | Edit `config/v16b/env`, inspect `launch.sh --print`, and confirm both host directories exist and are readable. |
| Draft vocabulary missing | `--run` expands the bundled gzip into the user's cache and checks for 65,536 ids. Confirm `gzip` is installed and cache storage is writable. |
| `SPEC_EXTRA` JSON fails | Install `jq`; the value must be a JSON object. `--print` fails before any Docker mutation. |
| GPU or CUDA load failure | Check the NVIDIA container runtime, `nvidia-smi`, the digest-pinned base and the container logs. The source does not define a minimum driver version. |
| Memory pressure or out-of-memory | Restore the tested KV, sequence and prefill settings; stop competing GPU jobs; inspect `MemAvailable` and driver logs. A larger KV pool or prefill rail costs real unified memory. |
| Cold TTFT much higher than warm | Check cache hits and the PLE table's storage latency. A cold 64k prefill is a different workload from a cached coding turn. |
| Prefix cache behaves unexpectedly | Keep the fp8 hybrid flag and split-op list aligned with the pinned image; compare served logs and image tag before changing scheduling. |

Open a bug report with hardware, driver, image digest, sanitized command output and relevant log excerpt. Do not attach credentials or private prompts.
