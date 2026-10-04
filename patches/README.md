# Optional vLLM patches

Off by default. Turn one on by listing its name in `recipe.yaml`:

```yaml
server:
  patches: hermes-chat          # space-separated, applied in this order
```

At launch `run.sh` copies each file a patch touches out of the image, applies the patch and mounts the result
read-only over the image's file. The image is never changed; an empty
`patches` = the stock image. A patch that does not fit the image stops `run.sh` before anything starts.

| patch | what it does |
|---|---|
| `hermes-chat` | Hermes agent: reads its `{"reasoning": {…}}` object (thinking on/off, effort) and makes an omitted temperature greedy. Contributed by [@yume-arasaki](https://github.com/yume-arasaki) (#2) |
| `qsa-logits-workspace` | Long prefills: the QSA indexer reuses one logits workspace instead of growing a new buffer per chunk (host freezes on unified memory). Backport of [vllm-project/vllm#57105](https://github.com/vllm-project/vllm/pull/57105); reported by [@anzax](https://github.com/anzax) (#5) |

## Adding one

A unified diff with paths relative to the `vllm` package (`--- a/entrypoints/…`, `+++ b/entrypoints/…`), made against
the image in `recipe.yaml`. Lines before the first `---` are a free-text description.
