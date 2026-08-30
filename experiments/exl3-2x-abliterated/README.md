# Two-Spark EXL3 abliterated evaluation

This lane evaluates the public MiaAI recipe against the qualified four-Spark
FP8/SGLang service. It is not the production recipe and must not replace the
four-Spark systemd service without a separate promotion decision.

## Frozen candidate

| Component | Pin |
|---|---|
| Recipe | `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks@b5ab8091dec88e324c943deb96c2dfd957db9f36` |
| Container | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58` |
| Target | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw@25a44fdbf16862a46b7cc9921142c6c81350af2f` |
| Draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, k=7, draft TP2 |
| Ablation donor | `dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4@e90ef415a2bf1274f848965a61144eca1e99eabf` |
| Ablation | transplant, layers 15-45, `ABLIT=1` |
| Runtime | vLLM TP2, FP8 target KV, DFlash2, 1M max context |

The cluster-specific, credential-free environment is in
[`cluster.spark2-spark1.env`](cluster.spark2-spark1.env).

Install the exact upstream commit and this tested environment on the intended
head Spark:

```bash
./install-pinned.sh /opt/glm53-exl3-2x-abliterated
cd /opt/glm53-exl3-2x-abliterated
./start.sh download
set -a; source .env; set +a
python3 ablit/fetch_transplant.py
./start.sh start
```

The installer refuses to replace a non-git path and checks the detached
upstream commit before writing `.env`. It also applies the included one-line
download-path repairs that make the declared DFlash2 and ablation-donor
revisions effective; upstream pins the target revision but otherwise resolves
both of those inputs from moving `main` branches. Review the two node addresses
and cache paths in the environment file before starting it on another cluster.

### Fabric routing on this fleet

All NICs on this fleet share the `192.168.4.0/22` subnet. Linux initially
routed the worker address through Spark 2's 2.5 Gb management NIC even though
the selected `enp1s0f0np0` ports are 200 Gb ConnectX-7. Before syncing or
launching, install reciprocal `/32` host routes on the chosen pair and remove
them after the evaluation if the qualified service does not use them:

```bash
# Spark 2 (head)
sudo ip route replace 192.168.6.109/32 dev enp1s0f0np0 src 192.168.6.48
# Spark 1 (worker)
sudo ip route replace 192.168.6.48/32 dev enp1s0f0np0 src 192.168.6.109
```

Verify both directions with `ip route get <peer-ip>`. The corrected route
raised the encrypted weight-sync throughput from about 112 MB/s to roughly
350-400 MB/s on this run.

## License boundary

The serve code is MIT, but the downloaded components are not all MIT.
The EXL3 checkpoint uses ShapleyMcg License 1.0, and DFlash2 is
CC BY-NC-ND 4.0. No weights or ablation tensors belong in this repository.
Treat this as evaluation-only until DFlash2 commercial-use rights are cleared.

## Fair comparison

Both backends use the same `glm53ae20` Hermes profile, native high reasoning,
temperature 0, the same 16 AE read tools, and the same frozen repaired 20-job
suite. The candidate serves the drop-in model id `glm-5.3-flash` on the same
head address and port. Telegram and Buzz stay disabled.

Fresh four-Spark baseline on 2026-08-30:

- 19/20 normal exits; job 20 timed out before its first tool call.
- Zero malformed calls, tool errors, CRM writes, or tenant-gate failures in
  jobs 1-19.
- Median job time: 34.341 seconds.
- Structured decode median: 73.255 tok/s; prose: 39.329 tok/s.

Matched decode measurements are recorded in [`RESULTS.md`](RESULTS.md). The
two-Spark lane was 9.1% slower on structured output and 31.4% slower on prose
than the four-Spark control, while delivering substantially higher throughput
per Spark. The guarded AE run completed 20/20 but doubled median job latency
and failed the exact-once reducer contract in job 14, so the lane is not a
promotion candidate.

## Known Hermes compatibility issue

The tested vLLM build creates tool-call IDs as `chatcmpl-tool-*`. Hermes uses
that ID in the persisted spillover filename, while the fixed privacy reducer
accepts the qualified SGLang lane's `call_*` form. Do not loosen the reducer or
patch Hermes for this experiment. If the lane is revisited, normalize the ID
at the vLLM parser/API layer and prove the change with focused job 14 before
rerunning the suite.
