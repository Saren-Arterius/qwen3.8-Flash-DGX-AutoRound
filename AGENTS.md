<!-- SPDX-License-Identifier: Apache-2.0 -->
# Instructions for a coding agent: set up this recipe on the user's GB10

You are helping a person install the Qwen3.8 Flash DGX UltraFast recipe on their NVIDIA GB10 machine (DGX Spark, ASUS Ascent GX10 or similar). You are most likely running on their everyday computer and reaching the GB10 over SSH. Follow this file exactly. The person stays in control: you do the typing, they make every decision that changes their system.

## Ground rules

1. **Read-only first.** Phase 0 changes nothing. Do not install, download, build, start, stop or delete anything until the person approves the phase that does it.
2. **One approval per phase.** Before each phase from 1 on, show a short summary: the exact commands, what they change, the disk, network and time they cost, and how to undo them. Then stop and wait for a clear yes. An approval covers only that phase.
3. **Use only the commands in this file** and the scripts in this repository. Run every `--print` mode and show its output before the matching `--run`. Do not improvise other fixes.
4. **Never touch what you did not create.** Do not stop, remove, restart or kill any container, process or service that was already running. If something is in the way, tell the person and let them decide.
5. **No `sudo`, no system changes.** Do not change drivers, kernel settings, firewall rules, SSH settings, GPU clocks or power settings. If a step truly needs elevated rights, show the exact command and ask the person to run it.
6. **Do not edit** the repository's scripts, `env` file, launch flags or checksums. The recipe is measured as shipped.
7. **Never handle secrets.** Do not ask for, type or store passwords, SSH keys or tokens. If SSH or a download needs one, ask the person to set it up themselves.
8. **On any failure, stop.** Show the exact error, say what you think it means, and ask. Do not retry in a loop and do not work around a failed check, especially a failed hash check.
9. **Delete nothing.** If cleanup is needed, list the commands and let the person run them.

## How to run commands on the GB10

Ask the person how to reach the GB10, for example `ssh user@hostname` or an SSH config alias such as `ssh gb10`. Use it as `GB10` below. The person must have key-based SSH login working and must have connected once themselves to accept the host key. Use `ssh -o BatchMode=yes GB10 '...'`, so a password prompt fails instead of waiting. If that fails, stop and ask the person to fix SSH access.

If you are running directly in a terminal on the GB10 itself, run the same commands locally without `ssh`.

Each SSH call starts a fresh shell. The working directory and an activated virtual environment do not carry over. So:

- Start every repository command with `cd "$HOME/qwen3.8-Flash-DGX-UltraFast" &&`.
- Call the Hugging Face CLI by its full path, `recipe/.venv/bin/hf`.
- Run the long phases (2, 3 and 4) detached on the GB10, so a dropped connection or a tool timeout cannot kill them. Write the phase's commands to a small script on the GB10, then start it in the background with a log:

  ```bash
  ssh -o BatchMode=yes GB10 'mkdir -p ~/ultrafast-setup && cat > ~/ultrafast-setup/phase2.sh' <<'EOF'
  set -euo pipefail
  trap 'echo "EXIT=$?"' EXIT
  # the phase's commands go here, exactly as written below
  EOF
  ssh -o BatchMode=yes GB10 'cd "$HOME/qwen3.8-Flash-DGX-UltraFast" && nohup bash ~/ultrafast-setup/phase2.sh > ~/ultrafast-setup/phase2.log 2>&1 < /dev/null &'
  ```

  Then check progress every few minutes with `tail -n 20 ~/ultrafast-setup/phase2.log`. The phase is finished when the log ends with `EXIT=0`. Any other exit code is a failure. Never start the next phase while one is still running. Show the person the script before you start it.

## Phase 0: read-only checks (no approval needed)

Run these on the GB10 and keep the output:

```bash
hostname; uname -m                          # expect aarch64
nvidia-smi                                  # the GPU name should contain GB10
free -g                                     # unified memory and what is available
df -h "$HOME"                               # free disk for the model files
docker --version
docker info 2>&1 | grep -i -E 'runtimes|docker root dir|permission denied'
docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
ls -ld "$HOME/models" "$HOME/qwen3.8-Flash-DGX-UltraFast" 2>&1
(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep ':8000 ' || echo 'port 8000 looks free'
for t in git python3 jq md5sum sha256sum curl; do command -v "$t" >/dev/null || echo "missing: $t"; done
python3 -m venv --help >/dev/null 2>&1 || echo 'missing: python3 venv module'
```

Stop and tell the person, without going further, if any of these holds:

- The machine is not ARM64, or the GPU is not a GB10. This recipe targets GB10 only.
- Docker is missing, the NVIDIA runtime is not listed, or `docker` needs `sudo` (a permission denied error). The person has to fix Docker access first.
- Another model server or large process is already using the GPU or unified memory. With this recipe running, only about 16 GiB of the 121 GB stays free, so nothing else large can run beside it.
- A container named `qwen38-flash` already exists, or port 8000 is in use. The launcher replaces a container with that name and publishes port 8000.
- Under about 150 GB of disk is free for `$HOME/models`, or the Docker root directory is short on space. The downloads are about 130 GB, plus about 5 GB for the drafter directory and room for Docker images.
- `$HOME/models` or the repository folder already exists with other content. Ask before using it.

The model files must live in `$HOME/models`, because the launch settings point there. If `$HOME` lacks the space, the person can make `$HOME/models` a link to a larger disk themselves. Do not change the settings instead.

Then give the person a plain-language plan: what you found, what is missing, and the phases below with their sizes. Wait for their answer.

## Phase 1: get the repository and tools (approval)

```bash
cd "$HOME" && git clone https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast.git
cd "$HOME/qwen3.8-Flash-DGX-UltraFast" && python3 -m venv recipe/.venv && recipe/.venv/bin/pip install --upgrade huggingface_hub
cd "$HOME/qwen3.8-Flash-DGX-UltraFast" && recipe/.venv/bin/hf --help >/dev/null && echo "hf ok"
```

If a system tool from Phase 0 is missing, show the install command for the GB10's package manager and ask the person to run it, since it needs `sudo`.

## Phase 2: download the public model files (approval, detached)

Tell the person these are about 130 GB from Hugging Face and can take hours, and that the model licenses are theirs to check before use. Run as one detached job:

```bash
mkdir -p "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid" "$HOME/models/ple-table-fp8" && \
recipe/.venv/bin/hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid \
  --revision 8b82f0b7abe3d1150a7827d298c75e86267636ae \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid" && \
recipe/.venv/bin/hf download Saren/Qwen3.8-Flash-Next-ple-table-fp8 \
  --revision 50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --local-dir "$HOME/models/ple-table-fp8"
```

If it fails partway, show the log and ask before running it again. Re-running resumes the download.

## Phase 3: build the image (approval, detached)

Run this first and show the output. It lists the source and hash checks and the Docker commands:

```bash
bash recipe/build/image/build.sh --print
```

Tell the person `--run` pulls a large public vLLM image and runs two local Docker builds and CPU test gates. After approval, run `bash recipe/build/image/build.sh --run` as a detached job. If any gate or hash check fails, stop.

## Phase 4: build the drafter directory (approval, detached)

Run `bash recipe/build/model/build.sh --print` and show the output. After approval, run `bash recipe/build/model/build.sh --run` as a detached job. It writes about 4.8 GiB of new data next to the downloaded checkpoint and links the unchanged shards. The log must contain `VERIFY OK`. Do not run the builder with other options.

## Phase 5: launch and test (approval)

Run `bash recipe/config/v16b/launch.sh --print` and show the command. Then tell the person plainly what `--run` does:

- It starts a container named `qwen38-flash`, first removing any container with that name.
- It uses about 71 GiB of memory for the model plus a 16 GB KV cache.
- It publishes an OpenAI-compatible API on port 8000 on all network interfaces, with no authentication, so other machines on the network can reach it.
- It sets the container to restart automatically after a reboot.

After approval, start it. The command returns as soon as the container starts:

```bash
bash recipe/config/v16b/launch.sh --run
```

Loading takes a while. Check readiness every minute with separate calls, for up to 30 minutes:

```bash
curl -fsS -m 5 http://localhost:8000/v1/models >/dev/null && echo READY || docker logs --tail 20 qwen38-flash
```

Do not use `docker logs -f`, which never returns. Once it reports `READY`, send a test request:

```bash
curl -fsS -m 300 http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":512}'
```

A working setup answers with a normal completion. If the container exits, or it is not ready after 30 minutes, stop, show `docker logs --tail 50 qwen38-flash`, and ask.

## Finish

Report in plain language:

- What was installed and where, and the disk it uses.
- The API address. It is `http://localhost:8000/v1` on the GB10, and `http://<GB10 address>:8000/v1` from other machines on the network. The model name is `qwen`.
- How to stop or remove it:

```bash
docker stop qwen38-flash                  # stop the server
docker update --restart=no qwen38-flash   # do not restart it on reboot
docker rm -f qwen38-flash                 # remove the container
```

Mention that the benchmark scripts in `recipe/benchmarks/` reproduce the published numbers and that `docs/CONFIGURATION.md` explains every launch setting. Offer to run a benchmark, but do not start one without asking.
