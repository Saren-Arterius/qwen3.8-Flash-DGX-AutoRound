<!-- SPDX-License-Identifier: Apache-2.0 -->
# Instructions for a coding agent: set up this recipe on the user's GB10

You are helping a person install the Qwen3.8 Flash DGX UltraFast recipe on their own NVIDIA GB10 machine (DGX Spark, ASUS Ascent GX10 or similar). Follow this file exactly. The person stays in control: you do the typing, they make every decision that changes their system.

## Ground rules

1. **Read-only first.** Phase 0 changes nothing. Do not install, download, build, start, stop or delete anything until the person approves the phase that does it.
2. **One approval per phase.** Before each phase from 1 on, show a short summary: the exact commands, what they change, the disk, network and time they cost, and how to undo them. Then stop and wait for a clear yes. An approval covers only that phase.
3. **Use only the commands in this file** and the scripts in this repository. Run every `--print` mode and show its output before the matching `--run`. Do not improvise other fixes.
4. **Never touch what you did not create.** Do not stop, remove, restart or kill any container, process or service that was already running. If something is in the way, tell the person and let them decide.
5. **No `sudo`, no system changes.** Do not change drivers, kernel settings, firewall rules, GPU clocks or power settings. If a step truly needs elevated rights, show the exact command and ask.
6. **Do not edit** the repository's scripts, `env` file, launch flags or checksums. The recipe is measured as shipped.
7. **Never handle secrets.** If a download asks for a token, ask the person to set it up themselves.
8. **On any failure, stop.** Show the exact error, say what you think it means, and ask. Do not retry in a loop and do not work around a failed check, especially a failed hash check.
9. **Delete nothing.** If cleanup is needed, list the commands and let the person run them.

## Phase 0: read-only checks (no approval needed)

Run these and keep the output:

```bash
uname -m                                   # expect aarch64
nvidia-smi                                 # the GPU name should contain GB10
free -g                                    # unified memory and what is available
df -h "$HOME"                              # free disk where the downloads will go
docker --version && docker info 2>&1 | grep -i -E 'runtimes|nvidia'
docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
ss -ltn | grep ':8000' || echo 'port 8000 is free'
command -v python3 jq md5sum sha256sum hf git
```

Stop and tell the person, without going further, if any of these holds:

- The machine is not ARM64 or the GPU is not a GB10. This recipe targets GB10 only.
- Docker or the NVIDIA container runtime is missing.
- Another model server or large process is already using the GPU or unified memory. With this recipe running, only about 16 GiB of the 121 GB stays free, so nothing else large can run beside it.
- A container named `qwen38-flash` already exists, or port 8000 is in use. The launcher removes a container with that name and publishes port 8000.
- Under about 150 GB of disk is free. The downloads are about 130 GB, plus about 5 GB for the drafter directory and room for Docker layers.

Then give the person a plan in plain language: what you found, which tools are missing, and the phases below with rough sizes. Ask where to put the repository (default `$HOME/qwen3.8-Flash-DGX-UltraFast`) and where to put the models (default `$HOME/models`). Wait for their answer.

## Phase 1: get the repository and tools (approval)

```bash
git clone https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast.git
cd qwen3.8-Flash-DGX-UltraFast
python3 -m venv recipe/.venv
. recipe/.venv/bin/activate
pip install --upgrade huggingface_hub
```

Install any other missing tool from `jq`, `git` or the GNU coreutils only after asking, and say which package manager command you would use.

## Phase 2: download the public model files (approval)

Tell the person these are about 130 GB from Hugging Face, and that the model licenses are theirs to check before use. Then run the two `hf download` commands from [docs/BUILD.md](docs/BUILD.md#1-download-the-public-checkpoints) with the pinned revisions, into the models directory they chose. If the download stops, ask before resuming.

## Phase 3: build the image (approval)

```bash
bash recipe/build/image/build.sh --print
```

Show the output, which lists the source-file and hash checks and the Docker commands. Tell the person it pulls a large public vLLM image and runs two local Docker builds. After approval:

```bash
bash recipe/build/image/build.sh --run
```

It ends with CPU test gates. If any gate or hash check fails, stop.

## Phase 4: build the drafter directory (approval)

```bash
bash recipe/build/model/build.sh --print
bash recipe/build/model/build.sh --run
```

This writes about 4.8 GiB of new data next to the downloaded checkpoint and links the unchanged shards. It must finish with `VERIFY OK`. Do not run the builder with other options.

## Phase 5: launch and test (approval)

```bash
bash recipe/config/v16b/launch.sh --print
```

Show the command, then tell the person plainly what `--run` will do:

- Start a container named `qwen38-flash`, removing any container with that name first.
- Use about 71 GiB of memory for the model plus a 16 GB KV cache.
- Publish an OpenAI-compatible API on port 8000 on all network interfaces, with no authentication, so other machines on the network can reach it.
- Set the container to restart automatically after a reboot.

After approval:

```bash
bash recipe/config/v16b/launch.sh --run
docker logs -f qwen38-flash        # startup takes a while; stop following once the server is ready
until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 10; done
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":512}'
```

A working setup answers the test request with a normal completion. If startup fails or the server never becomes ready, stop, show the last part of `docker logs qwen38-flash`, and ask.

## Finish

Report in plain language: what was installed and where, the disk it uses, the server address, and how to stop or remove it:

```bash
docker stop qwen38-flash                  # stop the server
docker update --restart=no qwen38-flash   # do not restart it on reboot
docker rm -f qwen38-flash                 # remove the container
```

Mention that the benchmark scripts in `recipe/benchmarks/` reproduce the published numbers and that [docs/CONFIGURATION.md](docs/CONFIGURATION.md) explains every launch setting. Offer to run a benchmark, but do not start one without asking.
