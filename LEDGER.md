# GLM-5.3-Flash on 4x DGX Spark (GB10, sm121) - Campaign Ledger

Dates: 2026-08-26 → 2026-08-27. Fleet: gx10-1..4 (192.168.100.11-14), TP=4 over dual-rail RoCE.
Weights: zai-org/GLM-5.3-Flash FP8 (62 shards, ~320GB) at /var/tmp/glm-legacy/hf/hub/glm53-flash-fp8 on every node.

## Phase 1: vLLM PR #53906 (DEAD END on sm121 - three boots, exact causes)

Own source build `glm53-flash-vllm:pr53906-933876c` (PR head 933876c, 12.1a, ~3h build).
Weights loaded, tilelang mHC kernels compiled clean on GB10 - that evidence later justified the SGLang TileLang path.

- Boot 1: `concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla` (cache_kernels.cu:866).
  SM120 sparse backend forces packed fp8_ds_mla layout; GLM-5.3 is NoPE (qk_rope_head_dim=0), pe_dim can never be 64.
- Boot 2: `--attention-backend FLASHMLA_SPARSE` → "compute capability not supported" (Hopper-only upstream).
- Boot 3: overlay canonicalization patch + fp8_e4m3 KV → backend itself raises
  `FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla KV cache layout` (flashinfer_mla_sparse_sm120.py:68).

Verdict: the PR tree has NO working sm121 sparse path for a NoPE model. Not "cannot run on Sparks" -
that overclaim cost credibility. Lesson filed: sweep all engines before declaring dead ends.

## Phase 2: SGLang (WORKS)

Engine: `lmsysorg/sglang:glm-5.3-flash` (day-0 image, arm64, sglang 0.0.0.dev1+gd6ab04bdf1, tilelang 0.1.12).
Merged cookbook: docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx. Known bug sglang #36550
(graph-replay abort, only cold prefill >262K with graphs on - irrelevant at 32K).

### The one GB10 patch required
Stock DSA sparse kernel (`sglang/kernels/ops/attention/dsa/tilelang_kernel.py`,
`sparse_attention_fwd_kernel_v1/v2`) requests 151552 B dynamic shared memory
(64x512 bf16 KV tile, double-buffered). GB10 sm121 caps ~99KB/block →
`tvm.error.InternalError: Failed to set the allowed dynamic shared memory size to 151552`.
Fix: `num_stages=2 → 1` (single-buffer, ~84KB). Overlay at /var/tmp/glm53-overlays/tilelang_kernel.py
on all 4 nodes (md5 eee1a43d74d23e97b30e3c84f058af75), bind-mounted read-only by the launcher.
UPSTREAM-WORTHY, not yet reported (pending).

### Launcher
scripts/launch_glm53_sglang_firstboot.sh - TP=4/nnodes=4, port 8213, dist 29653,
BF16 KV + tilelang DSA prefill+decode, 32K ctx, mem-fraction 0.75, preflight
(orphan kill + MemAvailable>=95G), no restart policy.

### Measured ledger (identical single-stream probe: 512 completion tokens, temp 1.0/top_p 0.95)
| boot | config delta | tok/s |
|---|---|---|
| SGLang boot 2 (first success, 2026-08-26 22:23) | eager, no spec | 6.7 |
| opt boot 1 | + decode CUDA graphs (capture 306s, 2.4GB; prefill graphs auto-off for KDA) + reasoning-parser glm45 | 14.0 |
| opt boot 2 | + native MTP head (NEXTN 5/1/6; draft = Glm5NextForConditionalGenerationNextN, 2.15GB, own graphs) | 20.6 |
| opt boot 3 | + speculative-adaptive + moe-runner flashinfer_cutlass + tool-call-parser glm47 | CRASH: `'Fp8MoEMethod' object has no attribute 'runner'` - cutlass MoE runner does not compose with the FP8 checkpoint in this build; NVFP4-lane lever only |
| opt boot 4 | adaptive + glm47 parser, MoE back to auto | (measuring) |

## Phase 4: tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark (community pointer, SAME HARDWARE)
Tony runs GLM-5.3 NVFP4 on 4x DGX Spark via the vLLM lane: day-0 vendor image
`vllm/vllm-openai:glm53-flash-arm64-cu130` + sm121-v8 patch stack (8 Dockerfiles).
35.7 tok/s generic median, 53-64 tok/s structured/agentic, TTFT 0.2s, 1.26M-token fp8 KV pool.
His patch 1 solves OUR vLLM dead end (extend SM90 NoPE backend gate to major 12, FA2 wrapper);
his phase-2 fp8-KV fix is the same GB10 ~100KB smem bug class as our tilelang patch (cap CTA_TILE_KV).
Also: FlashInfer >=0.6.18 mandatory on sm121 (FA2 MLA NaN below), nightly silently downgrades
NCCL (fabric-fatal) + cutlass-dsl (warmup ICE), --block-size 2304 (DeepGEMM arch-12 64-entry
pool pages), cache_flusher sidecar during weight load (GB10 driver vs page cache), reboot nodes
after many boot cycles (driver alloc-pool degradation), MTP per-position acceptance
[0.74,0.47,0.27,0.15]. Repo cloned to scratchpad; probes/ dir = reusable kernel test kit.
OPEN DECISION (the operator's): NVFP4 lane = 0xSero SGLang stack vs Tony vLLM v8 stack.

Accept length observed: 2.40 → 3.85 over warmup at NEXTN 5/1/6.
Per-node: weights 73GB, KV pool 433K tok BF16 (5.1GB), ~25GB free at mem-fraction 0.75.

## Phase 3: levers mined from 0xSero/glm-5.3-flash-sglang-sm120 (community pointer)

Locked deployment, 4x RTX PRO 6000 sm120, 143-208 tok/s single-stream. Same base image digest.
Arch gates in his six patches check `device_sm_major == 12` → sm121 QUALIFIES.
- Adopted in opt boot 3: speculative-adaptive, flashinfer_cutlass MoE, glm47 tool parser.
- Staged, NOT adopted (operator decision): NVFP4 checkpoint
  LibertAIDAI/GLM-5.3-Flash-NVFP4 @ 9e0d74e3 (182GiB, 120 shards) downloading to
  gx10-1:/var/tmp/glm-legacy/hf/hub/glm53-flash-nvfp4; derived image
  `glm53-flash-nvfp4-sglang:sm121-20260827` (his patches + TORCH_CUDA_ARCH_LIST=12.1a,
  FLASHINFER_CUDA_ARCH_LIST=12.1f; his Dockerfile had 2 missing CMD line continuations, fixed)
  built on gx10-1, fanned to all nodes.
- Future levers: flashinfer_sparse_mla DSA backend + FP8 KV (checkpoint-agnostic, works with
  our FP8 weights via patched image), ep-size 4 (RoCE all-to-all - measure, not assume),
  reasoning_effort low/high via chat_template_kwargs, never small max_tokens (burns inside think).
- His CuteDSL note: flashinfer cutedsl MoE hard-codes major [10] - never try it on GB10.

## Node hygiene (2026-08-27)
- GLM-5.2 QuantTrio weights (380G/node) DELETED from all 4 nodes; Storage archive verified
  complete beforehand (129/129 + snapshot tgz on /Volumes/Storage/weights-staging/).
- Ornith: previously archived + deleted (freed ~257G/node).
- ~400G free per node after sweep.

## Optimization ladder close-out (2026-08-27 early AM)
Probe: 512-tok completion, temp 1.0/top_p 0.95 (client-wall); community ruler = streaming
decode-only, thinking off, temp 0, 200 tok.

| boot | change | client-wall | decode-only |
|---|---|---|---|
| 5 | FP8 KV + flashinfer_sparse_mla (0xSero sm121 patch stack, our rebuild) | 19.6 | 22.2 |
| 6 | mem-fraction 0.80 (KV pool 1,233,408 fp8 tok) + persistent JIT cache | (polluted by tuner co-run) | - |
| 7 | chunked-prefill 2048 (Light Foundry GB10 finding) | 22.5 | 23.7 |
| 8 | max-running-requests 8 | c1 23.4 | agg: c2 28.2 / c4 42.4 / c8 60.5 |

Landscape (same-class): 0xSero accepted TP4 evidence C1 22.69 (we are at/above parity);
Lucas Fulks 30 / CosmicRaisins ~33 decode FP8 4-Spark (unpublished configs);
Light Foundry 8-Spark TP8: c1 74, c8 223 agg (twice the ranks).

MoE tuner: PARKED. Triton fused-MoE has no GB10 config (E=288,N=512 fp8_w8a8 falls to
defaults, startup warns sub-optimal). Exhaustive tune is O(hours) on GB10 because bad
tile configs cost ~70s each to benchmark; two false starts documented (arch missing from
tuner table -> patched common_utils, ray OOM when co-located with serving). NEXT: pruned
--search-space-file of GB10-sane tiles (small block_m, stages<=3), solo node, ~45 min.
Config filename serving expects: E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128, 128].json
DFlash 2 for GLM-5.3 announced by Zhijian Liu (vLLM circle) - watch upstream.

## PRODUCTION CONFIG FROZEN (2026-08-27)
**Boot 8 formula is the permanent launch config**: scripts/launch_glm53_production.sh
(byte-identical logic to the final firstboot launcher; image retagged glm53-sglang-sm121:20260827
- NO NVFP4 anywhere in the serving path, the old image name was inherited and misleading).
Revert-boot verification: aggregates reproduce (c4 40.4 / c8 59.0 ~ within variance of
42.4 / 60.5); c1 warms with MTP acceptance (19.8 cold -> 22.2 at accept 2.7, converges
to 23.4 at the ~3.8 plateau). c1 IS acceptance-coupled: never compare cold c1 to warm c1.

Boot 9 lesson (MoE tune regression, reverted): partial Triton MoE tuning is WORSE than
none. Serving reads TWO config files (base + _down.json); we supplied one, with only 4
batch keys, so non-decode shapes snapped to decode configs -> -10% everywhere. A proper
retune needs: both files, batch keys spanning decode AND prefill (incl 2048), then A/B.

Shelf (documented, not running): NVFP4 weights + lane on all nodes; pruned-space MoE
retune recipe; DFlash 2 watch; upstream report of the tilelang smem patch (needs the operator's go).

## Tony 2x-Spark companion repo intel (2026-08-27, community pointer)
- KV sizing doctrine: grow pool until ~8-10GB residual/node; GATE every bump behind
  CONCURRENT 20K prefills (his 32GiB passed single-prefill, died at 3x). Our gmu 0.80
  keeps ~20GB residual = safe zone. "Phantom backing": GB10 reservations can succeed
  then die on first touch in warmup.
- InstantTensor loader: 15x load speedup (10min -> 40-100s) but a rank silently dies
  post-load in multi-node; he ships it disabled. WATCH, retest when upstream moves.
- Phase-3 candidate (unbuilt): ~40-line zero-pad-rope shim routes NoPE onto the
  Blackwell-native trtllm packed-fp8 decode kernel = potential decode uplift beyond
  the current ladder, applicable to either engine.

## Power-off cross-check vs tonyd2wild/DGX-Spark-Hard-Poweroff-Fix (2026-08-27)
All 4 nodes on production firmware GX10DGX.0104.2026.0326 = his STABLE cohort (his
crashers ran pre-prod Sep-2025 SBIOS). Our Aug incidents = his over-commit lookalike
mode, already mitigated (SBSA watchdog, earlyoom, MemAvailable gates); zero recurrences
since 08-19. Clock cap NOT applied (~5% cost, protects a cohort we are not in; keep
`nvidia-smi -lgc 300,2200` as the reversible first response if a log-less off recurs).
ADOPTED: /etc/cron.d/drop-caches on all 4 nodes (pagecache-only echo 1, every 5 min).
WATCH: gx10-4 idled at 46C (his marginal band 43-48; siblings 39-41) right after the
tuner run - recheck when truly idle; if persistent, airflow/TIM flag for that unit.

## Client defaults (per Z-AI model-card guidance 2026-08-27)
Pass via chat_template_kwargs on every request unless a specific call needs otherwise:
- `clear_thinking: true` - template default is FALSE, which replays prior-turn thinking
  into context on multi-turn chats (context + token burn). Z-AI: "for chat scenarios,
  explicitly pass clear_thinking=true."
- `reasoning_effort`: "low" for simple Q&A/tool glue, "high" for balanced agent work,
  omit (= "max") only for hard reasoning and benchmark reproduction.
- Never set small max_tokens with thinking on - budget burns inside <think> with empty
  content (0xSero + our own probes). Omit max_tokens or disable thinking instead.
- Sampling per checkpoint generation_config: temperature 1.0, top_p 0.95.
Upstream sync check 2026-08-27: all serving files byte-identical to zai-org/GLM-5.3-Flash
(tokenizer sha256 19e77364 verified); only README changed. Fleet-kit watch now covers
3 HF + 10 GH repos + 4 issue threads (fleet glm53-flash upstream).
