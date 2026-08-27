# GLM-5.3-Flash FP8 on 4x NVIDIA DGX Spark (GB10, sm121)

Serving [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total / 18B active MoE, native FP8, released 2026-08-26) across four DGX Spark nodes at tensor parallel 4 via SGLang, one day after the model dropped.

This is the official FP8 checkpoint lane: full FP8 weights, FP8 KV cache, and the model's native MTP draft head, on the day-0 SGLang image plus one community patch stack and one kernel patch of ours.

## Numbers

| Metric | Value |
|---|---|
| Single stream decode (streaming ruler, thinking off, temp 0) | 23.7 tok/s |
| Single stream, client wall including prefill | 23.4 tok/s |
| Aggregate c2 / c4 / c8 | 28.2 / 42.4 / 60.5 tok/s |
| KV pool | 1,104,000 FP8 tokens at 262K config (4.2 full depth concurrent requests) |
| Context (this config) | 262,144, gated: cold 169K needle retrieval passed, 3x concurrent 19K prefills survived |
| Decode at 200K depth | 20.0 tok/s (16 percent below short context; the linear attention layers earn their keep) |
| MTP accept length, warmed | ~3.8 |
| Boot | ~20 min (14 weights, 6 graph capture) |

For calibration against the field on the same hardware class: the accepted 4-Spark TP4 evidence in the community bundle this builds on reports C1 22.69 tok/s. Unpublished configs report 30 to 33. An 8-Spark TP8 deployment of the same lane reports c1 74. Losing attempts are in the ledger, including a MoE tile tune that made everything 10 percent slower and was reverted.

## The formula

```
FP8 weights (official checkpoint) + FP8 e4m3 KV cache
+ DSA attention via flashinfer_sparse_mla (sm121 patch stack)
+ NEXTN MTP: 5 steps, topk 1, 6 draft tokens, adaptive
+ decode CUDA graphs to bs 8 (prefill graphs auto off for KDA hybrid)
+ chunked prefill 2048 (measured better than 8192 on GB10)
+ mem-fraction-static 0.80, max-running-requests 8
+ context-length 262144 (2^18 exactly: a known upstream graph-replay bug fires only above 2^18 cold prefill, so this cap takes the full window and keeps decode CUDA graphs)
+ reasoning parser glm45, tool call parser glm47
```

Every flag's provenance, every failed boot, and every measurement ruler is in [LEDGER.md](LEDGER.md).

## Reproduce

1. **Weights**: download `zai-org/GLM-5.3-Flash` (62 shards, ~320 GB) to one node, fan out to the rest over your fast fabric. Every rank needs the full checkpoint locally.
2. **Image**: build [docker/Dockerfile.sm121](docker/Dockerfile.sm121) on top of the pinned `lmsysorg/sglang:glm-5.3-flash` digest. It bakes the six sm12x compatibility patches from 0xSero's bundle with `TORCH_CUDA_ARCH_LIST=12.1a` and `FLASHINFER_CUDA_ARCH_LIST=12.1f` for GB10. Fan the image to all nodes.
3. **Kernel patch**: apply [patches/tilelang-gb10-smem.patch](patches/tilelang-gb10-smem.patch) to the SGLang tilelang DSA kernel and bind mount it (the launcher does this). GB10 caps dynamic shared memory near 99 KB per block; the stock kernel double buffers a 148 KB KV tile and dies in warmup with `Failed to set the allowed dynamic shared memory size to 151552`. Single buffering fits. Only needed if you use the tilelang DSA backend; the flashinfer_sparse_mla formula above does not hit it, we ship it because the fallback path should work too.
4. **Launch**: edit the IP and host maps at the top of [scripts/launch_glm53_production.sh](scripts/launch_glm53_production.sh), then run it. Workers boot first, head last. Preflight requires 95 GB MemAvailable per node and kills orphaned GPU processes.
5. **Client defaults**: pass `chat_template_kwargs: {"clear_thinking": true}` for chat workloads and use `reasoning_effort` (`low` / `high` / default `max`) to control thinking cost. Do not set small `max_tokens` with thinking on; the budget burns inside the think block.

Operational notes that each cost us a boot or worse: drop the page cache before every launch and on a 5 minute cron (GB10 unified memory starves the CUDA allocator behind a full page cache), tear down all ranks before relaunching any, and never benchmark a cold server's single stream number against a warm one, MTP acceptance takes minutes to climb.

## Credits, chronological

- [Z-AI](https://huggingface.co/zai-org/GLM-5.3-Flash) for the model and the native FP8 checkpoint
- [sgl-project](https://github.com/sgl-project/sglang) for day-0 GLM-5.3 support and the pinned image
- [0xSero](https://github.com/0xSero/glm-5.3-flash-sglang-sm120) for the six sm12x compatibility patches this image bakes in, and the sm121 bundle lineage
- [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark) for the GB10 KV sizing doctrine, the cache flusher finding, and the parallel vLLM lane diagnosis
- [Light Foundry](https://x.com/light_foundry) for the chunked prefill 2048 finding on GB10 and the 8-Spark TP8 reference numbers
- [Lucas Fulks](https://x.com/lucasfulks) and [CosmicRaisins](https://github.com/CosmicRaisins) for the first native FP8 fleet numbers on 4x Spark, which set the bar this recipe chased
- The vLLM PR [#53906](https://github.com/vllm-project/vllm/pull/53906) authors for the day-0 vLLM lane; our three sm121 failure reports from that lane are in the ledger and filed upstream

## License

MIT. Model weights are not redistributed here; SGLang and the baked patches carry their own licenses.
