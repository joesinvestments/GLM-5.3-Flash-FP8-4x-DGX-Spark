# Third party notices

| Component | Role here | Origin | License |
|---|---|---|---|
| GLM-5.3-Flash weights | The model (not redistributed) | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | MIT |
| SGLang | Serving engine, day-0 image base | [sgl-project/sglang](https://github.com/sgl-project/sglang) | Apache-2.0 |
| sm12x compatibility patch stack | Baked into docker/Dockerfile.sm121 | [0xSero/glm-5.3-flash-sglang-sm120](https://github.com/0xSero/glm-5.3-flash-sglang-sm120) | MIT |
| TileLang | DSA kernel runtime patched by patches/tilelang-gb10-smem.patch | [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | MIT |
| FlashInfer | Sparse MLA kernels used by the formula | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Apache-2.0 |
| GB10 operational doctrine | Cache flusher, KV sizing, boot rules cited in README/LEDGER | [tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark) | MIT |
| Chunked prefill 2048 finding | Serve flag adopted from published measurement | [Light Foundry](https://x.com/light_foundry) | n/a (published finding, credited) |
| First 4x Spark native FP8 numbers | Calibration targets in LEDGER | [Lucas Fulks](https://x.com/lucasfulks), [CosmicRaisins](https://github.com/CosmicRaisins) | n/a (published findings, credited) |
| DFlash 2 drafter | Evaluation-only speculative draft model used by historical and EXL3 A/B lanes (not redistributed) | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | CC BY-NC-ND 4.0 |
| EXL3 target checkpoint | Evaluation-only two-Spark target model (not redistributed) | [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) | ShapleyMcg License 1.0 |
| Abliteration donor checkpoint | Evaluation-only source for the layer transplant (not redistributed) | [dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4) | MIT |
| DFlash 2 SGLang integration | PRs applied in docker build | [sgl-project/sglang PR 36708](https://github.com/sgl-project/sglang/pull/36708), [PR 36755](https://github.com/sgl-project/sglang/pull/36755) | Apache-2.0 |
| vLLM day-0 GLM-5.3 lane | Failure analysis in LEDGER, issues filed upstream | [vllm-project/vllm PR #53906](https://github.com/vllm-project/vllm/pull/53906) | Apache-2.0 |
