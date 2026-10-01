# Qwen3.8 Flash-Next experiments

| Experiment | Decision / evidence |
| --- | --- |
| Greedy prompt-lookup proposals ahead of MTP | Rejected in local research: exact target token/frontier-logit/RNG replay passed on 11 corpora including accepted and rejected lookup proposals, and all 25 HTTP outputs matched upstream. The 20-request mixed workload nevertheless slowed about 2.2% versus the request-local controller; lookup displaced productive MTP widths and disrupted cost-observation continuity. Correct speculation alone is not a throughput win. |
| Request-local measured C1 draft costs | Local research: model-global C1 costs improved code but regressed the complete 20-request mix by 5.1% in total decode time; rejected that iteration. Request-local costs recovered chat/prose and improved aggregate decode from 41.83 to 43.55 tok/s on identical 8,941 output tokens. Sampled control remains deterministic. |
| Snapshot allocation without huge-page promotion | Local research retained: with resident weights, 1 GiB copying under MADV_HUGEPAGE took 1.2-2.6 s versus 0.16-0.18 s with MADV_NOHUGEPAGE. Whole-model mean snapshot capture fell from 1.42 to 0.13 s at 32K and 5.51 to 0.37 s at 128K. Allocation policy is local to snapshots; exact sampled snapshot replay passes. Fragmentation-dependent, not a universal huge-page result. |
| Snapshot retention independent of GPU session count | Local research retained: a bounded 24-entry host checkpoint cache preserved three interleaved 32K conversations using one GPU session; first-token latency fell from about 24.3 s to 0.17-0.19 s. Byte reservations and active-restore lifetime remain enforced. |
| Full-width MTP RMSNorm and split projection | Retained after independent CPU stage audit; one 10240-wide normalization, embedding projection shared across HC branches. |
| Full Q8 vocabulary head | Retained; private Q4 shortlist removed. Sampled top-64 proposals use exact target verification. |
| Batched MTP transformer and heads | Retained; independent body/head comparisons, private KV/recurrent/rollback/RNG state. |
| Batched decode mixers, residual epilogues and MTP norms | Retained; each request keeps its scalar reduction and private state; exact C2/C4/C6/C8 logits, acceptance and RNG. |
| HC Q8 weight prefetch | Retained for 1–8 rows of the 320×10240 projection; exact original products/FMA order, no extra allocation. |
| Q4 shared-expert weight reuse | Retained with a separate compact kernel for single-request experts; faster repetitive C1/C4/C8, no mixed-work regression, exact projection/model replay. |
| Q5 high-bit expansion | Retained; exact integer multiply/mask replaces repeated shifts, with unchanged dot products and faster serving. |
| Short convolution/history fusion | Retained for 1–8 tokens; exact output, rolling state and rollback snapshots, fewer launches and no additional allocation. |
| Batched GDN recurrence | Retained; private ragged state/rollback rows, cancellation isolation and exact session replay. Helps shallow batches most; end-to-end gains are modest. |
| Batched small projections and MoE preparation | Retained; group independent rows, quantize activations once in existing scratch, and batch router/shared-expert work. Exact scalar/batch outputs and sampled state; C1 unchanged. |
| Q4 expert grouping across the full batch | Retained; bounded groups share weights across request boundaries, with original scalar arithmetic and exact session replay. Q5 down grouping/reordering was slower on mixed routing and was rejected. |
| Expert-ordered prefill intermediates | Rejected: exact gate/up → down results with captured routing at 1024/2048 tokens, but no useful projection-chain gain for Q4_K/Q5_1 or Q5_K/Q8_0. |
| Wave64 batched Q4 gate/up | Retained above eight rows; independent 32-lane reductions preserve exact outputs and sampled state. Repetitive serving improves, C1 remains stable. Four output rows, 16-input groups and replacing the slot scan did not improve performance. |
| Register-cached vision softmax | Retained for bounded row sizes; unchanged reduction order, byte-identical Flash-Next/Qwen27B embeddings, no additional allocation. |
| 4096-patch vision attention tiles | Retained; byte-identical GEMMs/embeddings, lower latency and 24 MiB less attention scratch. Other shapes keep their original tiles. |
| Partitioned BF16 WMMA vision value projection | Rejected: failed the full-encoder reference gate despite lower isolated FP64 error. |
| Integer WMMA Q8 verification | Retained through 48 input rows for wide target projections/heads; preserves K8 partials/FMA order and reduces each result once, with exact session replay. Small batches retain vector kernels. |
| Q8 activation reuse across output rows | Rejected: no repeatable gain on the real projection shapes. |
| Wider Q8 decode and Q5 expert tiles | Rejected: 56–64 dense rows, phased/packed token tiles, Q5 wave64 and 16/32 Q5 expert rows were slower despite exact output. |
| Compact expert launch groups | Rejected: improved shared routing but negligible mixed-routing gain. |
| Sparse attention register/cache retuning | Rejected: exact d128K output, but register scheduling/occupancy gave no material gain and reloading queries was slower. |
| Shared attention selection lists | Rejected: exact outputs, but roughly 1% isolated gain did not justify another buffer and setup kernel. |
| Skip unused attention Q8 staging | Rejected: exact chunk/full-logit checks passed, but no clear end-to-end prefill gain justified the extra dispatch logic. |
| FP32 selector load scheduling | Retained; bounded scheduling removes scalar-register spills, preserves every score bit and lowers d32K selection time to 22.4 ms per pp2048. Matched d128K AR prefill improves about 1.5%. |
| Integer WMMA value transpose and paired FP32 selector lanes | Rejected: bit-preserving transpose and exact selector scores, but both were slower on deep-context inputs. |
| Packed Q8 prefill staging | Rejected: exact output, but extra decode/register/transpose costs outweighed reduced LDS use. |
| Persistent packed Q8 SSM weights | Rejected: small prefill/1–4-row gains would cost roughly 14% on eight-row decoding; two-step prefetch did not recover it. No second weight copy or private format retained. |
| Transient F16 SSM weights and parallel HC branches | Rejected: F16 staging was exact but slower overall; parallel HC branches changed quantization ties. |
| Smaller-LDS SSM projection, unrolled HC expert sum and wave64 HC combine | Rejected: exact output but no useful prefill speed gain. |
| HC quantized-output store layouts | Rejected: row-major staging plus transpose is slower; cooperative stores within one kernel save only about 2% in isolation. Residuals, normalized activations and Q8 bytes remain exact, including ragged rows. |
| Transient key transpose and mixed expert tiles | Rejected: exact outputs; key transpose slows deep selection, mixed tile sizes provide no useful prefill gain. |
| Query sharing, paired-key FP32 scoring, query LDS caching, MoE prefetch barriers/unrolling | Rejected: exact outputs, but no useful speed gain. Four-wave Q8 matrix reduction also lost to eight waves. |
| Approximate selector screening followed by exact rescoring | Rejected: GPU thresholding and compaction erased the isolated gain, before accounting for runtime error bounds. Remains a synthetic experiment; no approximate production selector added. |
| Ratio-four predictor QSA, FP32 ranking queries | Retained with sparse state, rewind and deep selector checks. |
| Greedy batch cost controller | Retained only for all-greedy C>1; separate occupancy/context bins, stable plain controls, no transition timings. Sampled replay uses fixed curves calibrated from median warmed cycles on 2026-09-20. |
| One-row MTP prefill lag | Retained; 320 KiB kept hidden state/session, avoids replaying a final prefill chunk. |
| Final-tile MTP catch-up | Retained; preserve every KV/indexer row, rank only the final attention tile, then restrict output projection, mixers and shared experts to the final aligned tile(s), with one routed Q8 expert row. Full predictor/catch-up stages, candidates and recursive carry match exactly at 224/257/2047/2048 rows. |
| Batched MTP catch-up tail | Retained; after writing every attention cache row, compact final request rows into existing scratch and skip unused output projections/FFNs. Preserve the scalar arithmetic of one-row tails; full-head candidates and recursive carry match independent execution. |
| Routed Q8 accumulation order | Retained; explicit rounded product/FMA prevents identical rows changing with column placement. Independent FP64 dot and predictor carry checks pass. |
| Isolated MMQ quantizer rounding change | Deferred: it changes half-integer tie behavior shared with W8A8 and fails their existing agreement gate. No quantizer change retained. |
| SSM tile, wave and compiler scheduling variants | Rejected: four/sixteen-wave groups, wave64, wider token tiles and iterative ILP scheduling retained exact output but did not beat the existing eight-wave projection. |
| SSM fragment lifetime and prefetch pipeline | Rejected: shorter-lived K16 fragments reduce register use without a useful gain; delayed prefetch and double-buffered LDS are slower. Projection/convolution outputs remain exact at 2048/2049 tokens. |
| Captured shared stages in batched target decoding | Rejected: exact C2/C4/C6/C8 logits, sampled state and cancellation checks, but no serving or warmed-cycle gain justified graph metadata/capture overhead. |
| Compact recurrent rollback | Retained; one full state plus exact FP32 update operands, with the original product/FMA order. All rollback prefixes match fresh execution. Depth grows on demand; seven-draft cap about 147 MiB/session, released on reset. |
| Live final frontier and async prompt snapshots | Retained; immutable branch snapshot, worker capture, bounded persistence outside the lookup lock. |
| Thread-local decode graph capture | Retained; independent snapshot workers no longer invalidate a peer's capture. Snapshot bytes and captured/replayed logits are exact; no request serialization added. |
| Chunk-equivalent projections/attention | Retained with exact continued-image/cache/full-logit gates; one-token tails keep prefill arithmetic. |
| More Q8 vocabulary rows/block | Rejected: no C1 improvement. |
| Private shared-expert F16 rows | Retained; the shared expert's SwiGLU rows for its F16 down projection get their own buffer instead of overwriting the routed experts' narrowed token rows, which removes one full-batch narrowing pass per layer. Every GEMM reads identical bytes; greedy output, chunk-boundary logits and the C2–C8 batch checks are exact. Interleaved Nix A/B at d0: 1559.6 → 1568.1 tok/s (+0.6%); the profiled kernel time fell 1362.8 → 1333.1 ms. |
| 256-row routed down-projection blocks | Rejected: exact output, Q5_1 unchanged and Q8_0 slower. The routed kernels already stream expert weights at roughly two thirds of DRAM bandwidth, so per-block prologue/epilogue and activation restaging were not the bound. |
| 64-token HC down-projection tiles | Rejected: exact output, 23% slower than the 128-token tile despite twice the resident blocks. |
| One-tile-ahead LDS fragment prefetch (dense and routed WMMA) | Rejected: exact output, but the explicit double buffer raised register use (dense 221 → 253 VGPRs; the 128-token pair reached 256 with spills) and every GEMM slowed 3–14%. |
| hipBLASLt F16 dense projections | Rejected: the pinned library reaches 19–26 TFLOPS on the 2048-token dense shapes against 30–34 TFLOPS for the Q8→F16 WMMA kernels; `tools/qwen-flash/dense_blaslt_sweep.hip` reproduces the sweep. |
| Side-stream inject/shared-expert overlap | Rejected: exact output and real kernel overlap in the trace, but the co-running kernels slowed each other and interleaved wall-clock runs were 0.5–0.9% slower. |

Separate d32K pp2048 profiling attributes 29.1% of kernel time to MoE, 34.9%
to dense projections and 12.6% to attention/indexing. Final-tile catch-up
takes 20.9 ms of MTP kernel time; the target takes 1450.2 ms.

A d0 pp2048 profile (2026-09-21) is GPU-bound: 1362.8 ms of kernel time in a
1380 ms span. Routed expert GEMMs take 35%, dense F16 projections 29%,
hyper-connection combines 10.5%, the GDN recurrence 9.5%, HC down/inject/shared
projections 7% and attention 5.5%. The dense projections run at 30–34 TFLOPS
against the measured 59 TFLOPS matrix ceiling and the combines at about
200 GB/s; the routed gate/up and down kernels stream 0.9–1.1 GB of expert
weights per layer at 160–190 GB/s.

A C4 mixed MTP trace with 32 output tokens per request is 84.9% GPU-busy
during inference, excluding model loading. Batched GDN updates take 40.9 ms,
rollback replay 14.5 ms and lazy allocations 31.8 ms. The scheduling thread
spends 5.4% of the window outside HIP API calls, including model-side CPU
work; this is not pure scheduler overhead. These are profile observations,
not unprofiled throughput measurements.

Next: improve prefill at depth and target/draft batch projection reuse while
preserving [quality](QUALITY.md). The 1700 tok/s PP and flat d0–d128K
objectives remain unmet; see [current benchmarks](BENCHMARKS.md).

## 2026-09-26 CPU-path research (Strix Halo platform audit, session continuation)

| Experiment | Decision / evidence |
| --- | --- |
| Qwen BPE merge loop rewrite (hybrid quadratic / lazy priority queue + linked list, merge targets precomputed at load, ASCII class table for the qwen35 pre-tokenizer) | Retained in local research: token-stream FNV-1a identical to the previous implementation on all 7 benchmark samples and 576/576 fuzz cases across 5 seeds (unicode, NFC, contractions, specials, whitespace backtracking); tokenizer unit tests and chat-template HF goldens pass. A 261 KB chat prompt encodes 8.2 → 4.5 ms; pathological single-piece inputs (base64/`====` runs that reach the server inside tool outputs) fall from 6.75 s to 8.8 ms per 64 KB piece, removing a multi-second admission stall per affected request. |
| Replacing hot `hipStreamSynchronize`/`hipEventSynchronize` with bounded spin (1 ms) then 50 µs sleep polls | Implemented in local research (`src/core/hip/wait_policy.hpp`; six executor call sites and `SnapshotTransfer` copy waits), shim-compiled against real HIP signatures; GPU-window validation pending. Motivation: HIP's default busy-wait pins a core at 100% for the whole GPU wait — the full prefill-chunk span on the scheduler thread, the whole snapshot capture copy, and ~30 ms per decode step — which also wastes STAPM power budget the iGPU could use. |
| Full-vocab sampling rewrite | Rejected as unnecessary: `PrepareSelected` already keeps a bounded top-k heap over the 248K vocabulary (O(V log k)); only the sparse-penalty per-candidate binary search remains as a measurable cost at high aggregate decode. |
| n-cpu-moe style CPU expert offload | Rejected for this APU without measurement: Strix Halo precedent shows GPU-resident experts win because the iGPU reads the same LPDDR5X at ~215 GB/s (gufo already streams routed experts at 160–190 GB/s on-GPU). |
| NPU (XDNA2) port of prefill or decode | Rejected: no HIP-like programming model for the NPU (FLM/ONNX-VitisAI only, LLM decode 2–5× slower than the 8060S), and this machine's `amd_iommu=off` disables the NPU entirely; re-enabling costs the 5–12% iGPU bandwidth the parameter buys. |

Platform notes recorded: measured GPU bandwidth on this class is 212–234 GB/s
(256 GB/s theoretical); gfx1151 is WMMA-only (no MFMA) with a 59.4 TFLOPS
FP16 matrix ceiling and 36.9 TFLOPS best-known measured; `ttm.pages_limit` on
this host is already ~124 GB; the `tuned` `accelerator-performance` profile
and a `performance` governor are unapplied free wins on this host; the ROCm
stuck-at-885 MHz clock bug warrants a production watchdog. Full report with
sources: `STRIX-HALO-OPTIMIZATION-20260926.md` in the research artifacts
directory.

## 2026-09-26 live captured-workload A/B (halogen 0.13.4 vs modified gufo)

Two user-authorized maintenance windows replayed the frozen 12-request
corpus (3 chat groups, ~155K-token mean prompts) against production-config
halogen and the modified gufo candidate. Halogen won end-to-end: 12/12
completions inside the 1200 s client deadline at 392 s duration p50 with an
80.8% prompt-cache share, versus gufo 2 sessions (10/12, 812 s p50, 55.2%
cache) and gufo 4 sessions (4/12, cache 0%). Gufo's engine rates were
competitive (prefill 1240 tok/s solo, MTP decode to 30 tok/s single-stream,
193K-token checkpoint restore in 74 ms), but (a) 2 sessions queue-starved 12
concurrent requests (queue_ms up to 1128 s) and (b) at 4 sessions the
host-snapshot budget (min(MemAvailable/2) after weights + KV, ~6.1 GiB)
could not hold even two ~4.3 GiB conversation snapshots, so every capture
was skipped (`byte_capacity`) and every follow-up turn re-prefilled.
Halogen's `HALOGEN_CACHE_INPLACE=1` keeps cache entries as O(1) state inside
its 26.6 GiB KV pool with zero host copies, which is structurally immune and
measured at 80.8% reuse. The transferable lesson: whole-snapshot host
budgets do not survive multi-session pressure on 124 GiB; retention must be
per-history (shared append-only regions, ~2 GiB per active conversation) or
KV-in-place for resident sessions. The HIP wait-policy and tokenizer rewrite
from this session ran under load in both windows without error. Full report:
`AB-RESULT-20260926.md` in the research artifacts directory.

## 2026-09-27 per-history snapshot retention rewrite (P0) and validation

| Experiment | Decision / evidence |
| --- | --- |
| Delta-charged snapshot reservations | Retained in local research: `TryReserveSnapshot` now receives the incremental physical estimate (full allocation bound minus the region data the state's pinned parent serves, via new `Storage::InheritedBytes`, `Executor::SnapshotInheritedBytes` and `Session::SnapshotIncrementalBytes`; the Flash-Next runner overrides the new `TextModelRunner::SnapshotIncrementalBytes` while the disk path keeps reserving full payloads). Commit re-deduplicates and fails closed on underestimates. 19/19 CPU tests pass including two new cases (growing-child inheritance estimate; delta reservation admitting a second shared-history capture that a full-size reservation would reject). |
| `--host-snapshot-gib` budget override; derived budget 1/2 → 3/4 of post-load headroom (cap 24 GiB) | Retained in local research. Validation at 4 sessions: 12 GiB overcommitted host memory (MemAvailable 386 MiB, kernel OOM killed the server, the cascade also took the user systemd manager — restore path hardened accordingly); 9 GiB held 3.1–4.5 GiB headroom for the whole window. |
| Captured-workload replay, gufo 4 sessions, before/after rewrite | After: 12/12 client completions (before: 4/12), cache share 58.7% (before: 0%), prefilled tokens 769K vs 1.33M (−42%), zero generation failures. Follow-up turns restore ~150K tokens in ~57 ms and prefill only 281–5.5K tail tokens. Halogen (in-place cache, 80.8% share) remains ahead on duration p50 (337 s vs 712 s); the residual gap is the host-RAM retention ceiling and admission behavior — see the P1 shared-device-pool design in `REWRITE-PLAN-CACHE-RETENTION.md`. |

## 2026-09-27 P1 stage 1: lineage-sticky admission, paired fresh A/B

| Experiment | Decision / evidence |
| --- | --- |
| Lineage-sticky admission (`TextSchedulerPolicy::sticky_admission`, default on; `--no-sticky-admission`) | Retained in local research: `PopQueued` prefers a queued request whose input identity matches a free slot's resident lineage (new `ContinuationCache::FreeSlotLineages` / `TextRunnerPool::FreeSlotLineages`), deepest live state first, per-client FIFO untouched, unrelated heads bounded by `capacity()` consecutive sticky admissions. CPU test `TestStickyAdmissionPrefersResidentLineage` proves the mechanism; 19/19 suite passes. On the captured workload it changed nothing measurable: with ~5 conversation lineages exceeding the 4-session slot count, the freed slot's lineage never matched the queue head, and every cache hit still restored from a host snapshot (52–70 ms). |
| 5 sessions × 262144 (device-side retention via parked sessions) | Rejected on this host by measurement: +7.4 GiB device KV drove post-load MemAvailable to 0.85 GiB mid-run (the 2026-09-26 12 GiB-budget attempt had already OOM-killed the server and the user systemd manager). Window aborted safely; production restored. Four sessions with a 9 GiB snapshot budget is the memory-safe ceiling here. |
| Paired fresh A/B (both arms cold, same window, same platform tuning) | Halogen 0.13.4: 12/12, duration p50 388 s / mean 400 s, cache share 80.8% — reproducing the earlier cold-start arm. Gufo (P0 + sticky, 4 sessions, 9 GiB): 12/12, duration p50 829 s / mean 766 s, cache share 58.7%. The residual gap is the host-RAM retention ceiling (9 GiB ≈ 2 lineages vs halogen's in-place pool holding all ~5 lineages' rows at zero copy), not admission order. |
| Stage-2 region-carving scoping | `Executor::CreateSession` allocates history-sized buffers per session (`k/v_cache`, `index_k`, `block_k`) flowing into kernels as launch arguments; carving them into a refcounted device pool with session rebind touches the Session struct, snapshot walk/restore and MTP rollback without kernel-body changes. Designed in `REWRITE-PLAN-CACHE-RETENTION.md` as the path to halogen-class in-place retention; requires dedicated GPU windows. |

## 2026-09-27 prompt-lookup proposal substitution (co-opted from the jadidbourbaki/llama.cpp fork research)

The fork's blog ("42x Faster Prompt Lookup Drafting in llama.cpp") optimizes the
cost of n-gram drafting: never copy follower maps, one flat open-addressed
context map, sorted follower arrays scanned branchlessly (64% of contexts have
a single follower), immutable constmaps for static corpora. Gufo's MTP path
drafts on-GPU, so drafting cost was never the bottleneck — but halogen
production demonstrably runs prompt-lookup decoding alongside MTP on greedy
requests (`pld ... acc/round` in its live logs), so the open question was
proposal quality, not drafting speed.

| Experiment | Decision / evidence |
| --- | --- |
| `lookup::ContextNgramCache` (per-session context cache, blog data structures adapted) | Retained behind `--prompt-lookup` (default off): flat open-addressed map keyed by 64-bit mixed context tokens (context lengths 0–3), per-key sorted follower arrays with count-tracked maxima, thresholds min-count/min-probability (2,2,1,1)/(0.66,0.5,0.5,0.5), longest-match-first early exit, lazy watermark catch-up from `tokens_` (rebuilt after snapshot restore), host-only state outside snapshot payloads. New `qwen38_flash_next.lookup_cache` CPU test (reference equivalence, catch-up idempotence, thresholds, rehash growth, collision containment); CPU suites pass (5/5 model, 18/18 CLI). |
| Substitute-into-MTP policy (avoids the documented ahead-of-MTP rejection mode) | Replaces at most the first three greedy draft-chain slots with confident lookup followers; the MTP length controller keeps choosing widths, so cost observation and verification batch shape are unchanged — the substitution can only move acceptance rate. Wired into both single and batched greedy verify paths. |
| Captured-workload A/B, round 7 vs round 6 (identical config + flag; halogen fresh control reproduced 388 s p50 / 80.8%) | Negative, second independent rejection: duration p50 829 → 915 s (+10.4%), mean 766 → 827 s, decode p50 1.9 → 1.2 tok/s (−37%), 11/12 (largest request missed the harness deadline). Every request slowed uniformly (+15…+124 s) with byte-identical completions; cache share unchanged (58.7 → 59.1%), so the loss is pure acceptance. |
| Mechanism and verdict | The trunk verifies greedily and stops at the first mismatch, so one wrong lookup proposal discards the remaining MTP proposals in the same cycle; on this workload the model's continuations diverge from copied prompt spans more often than the MTP head diverges from the target. The fork's 42× drafting-speedup does not transfer because gufo's bottleneck is proposal quality, not drafting cost; halogen's PLD advantage appears specific to its request mix/drafter. `--prompt-lookup` stays available (flag-gated) for future threshold/ngram-3-only retries without a rebuild; production defaults are unchanged. |

## 2026-09-27 P1 stage 2: elastic session history (implemented; GPU validation pending)

| Experiment | Decision / evidence |
| --- | --- |
| Region-carved pool → elastic capacity (design simplification) | Grounding findings: the indexer raw ring is bounded by top-k and the batch (never needed carving), snapshots are position-bounded logicalized views (cross-capacity rebind needs full relayout machinery), and the runner already parks sessions warm (stage-1 sticky admission routes same-lineage requests onto them). The only waste is fixed-shape capacity: every session pre-allocates 262144 positions regardless of content. Built instead: `--history-pool-gib N` (default 0 = legacy) makes every session's position-scaled history (per-attention-layer K/V + block_k, MTP K/V + block_k) start at 32768 positions and double on demand inside an executor-wide byte budget. |
| Elastic growth machinery (`Executor::EnsureHistoryCapacity`) | Doubling schedule (`kernels/rocm/history_capacity.hpp`, CPU-testable `history::BytesFor/NextCapacity`), device-to-device prefix copy, zero-fill, old-buffer release, decode-graph invalidation (kernel nodes bake the old pointers), atomic executor-wide budget reserve/release (`~Session` releases), device sync before frees. Growth wired into `Forward`, `MtpForward`, `ForwardBatch`, `MtpForwardBatch` and `RestoreSnapshotView` (a 150K restore grows the session to the snapshot position first). Fails closed on budget exhaustion: request fails, session stays at its old capacity and recovers via Reset. |
| Claim plumbing | `TextRunnerResourceClaim::shared_state_bytes` reserves the pool once for all states; the Flash-Next claim reports `SessionBytes − ElasticHistory(full) + ElasticHistory(initial)` per state so `--sessions N` under elastic history no longer claims N × full-context KV. `MeasuredResources` excludes live history bytes under elastic mode. |
| CPU validation | New `qwen38_flash_next.history_capacity` test (byte accounting incl. the compress_ratio block-row steps, growth schedule saturation/overflow/exact-need, copy amortization ≤ 2× final size). CPU suite 94/98 — the 4 failures are pre-existing artifact-gated quality tests, identical before the change. |
| Economics and status | At the captured workload's real lengths (~150K lineages) a warm lineage costs ~4.1 GiB of device history instead of 7.4 GiB, so the 29.6 GiB that today holds 4 fixed sessions holds 6–8 warm lineages; with stage-1 sticky admission every lineage keeps its rows in place (halogen's actual mechanism for its 80.8% share), and the 9 GiB host-snapshot budget shrinks to restart-safety. **The device path has never executed** — production served live traffic throughout, so no window was taken. Validation checklist and the follow-on research (`OPTIMIZATION-RESEARCH-20260927.md`: KV q8_0 as the next big lever, wave32 MoE audit, live concurrency measurement) are recorded in the research artifacts. |

## 2026-09-27 KV q8_0 cache rewrite (implemented; GPU validation pending)

The first deep-research target, implemented per `OPTIMIZATION-RESEARCH-20260927.md`
item 1: session K/V caches (trunk and MTP) can now store packed q8_0 blocks
(32 int8 quanta + one f16 scale per block, 34 bytes per 32 elements — 53% of
the f16 row). Per-block scales, not per-tensor: hipEngine's quality battery
measured direct per-tensor INT8 KV failing 9/11 prompts while scale-correct
schemes hit 100% top-1 agreement.

| Piece | Detail |
| --- | --- |
| Format (`kernels/rocm/kv_quant.hpp`) | `kv::RowBytes/QuantizeRow/DequantRow` reference arithmetic, CPU-tested: round-trip error ≤ ½ quantum + f16 scale rounding over 2000 random rows, zero-row exactness, dominant-magnitude clamping, scale invariance. New `qwen38_flash_next.kv_quant` test. |
| Store | `QuantizeKvKernel` (warp-per-block amax shuffle, scale, quantize, packed write) replaces `StoreKv` under `--kv-quant`. The fused prefill projection routes (`AttentionF16Gemm`, `PrepareAttention`) are bypassed in quantized mode so every store flows through the f32 scratch → quantize path; a later stage can re-fuse. |
| Read | `AttentionKernel` gains a q8 load path (byte quanta + block scale, dequantized in registers — 34-byte blocks give no vector alignment, so quanta load as bytes and still halve traffic). `WmmaCausalAttentionKernel` gains a `kQ8` template lane: `load_k`/`load_v` dequantize into the same f16 register groups, so the LDS staging, WMMA fragments, softmax and PV pipelines are untouched. |
| Plumbing | `Executor::Options::kv_quant`; `Session::AttentionState/MtpState::k_store/v_store`; `Executor::KvRowBytes()` feeds `SessionBytes`, `ElasticHistoryBytes` and the history-budget math (elastic budget buys ~2× positions under q8). `WalkSnapshot` emits the packed stores as the K/V regions; `SnapshotHeader::kv_row_bytes` distinguishes modes and `SameGeometry` rejects cross-mode restores; `kSnapshotPayloadVersion` 14 → 15. `--kv-quant` threaded through both serve parsers into the Flash-Next model options. |
| CPU validation | 95/99 CPU tests pass (the 4 failures are the same pre-existing artifact-gated quality tests). GPU tests compile (attention ops test updated for the new signatures). |
| Status | **Device code never executed** — production kept serving live traffic. Validation checklist for the next window: (1) quality gates first — reference-probe and exact-replay corpora with `--kv-quant` on (the oracle A/B the research report requires), plus the captured-workload quality comparison; (2) decode tok/s at depth vs f16 in the same window (the +14–41% class per the Strix Halo forks); (3) elastic + q8 combined (`--history-pool-gib 26 --kv-quant --sessions 6..8`) for the retention endgame; (4) snapshot round-trip capture→restore under q8. |

## 2026-09-27 round-8 window: first GPU validation of elastic history and KV q8_0

| Gate | Result |
| --- | --- |
| q8 smoke → bug found before any workload | FAIL → fixed: "logit distribution contains no finite values" on every request. New `qwen38_flash_next_kv_gpu_test` (quantize round trip; dense/sparse/prefill-shape attention q8 vs f16) isolated a WMMA LDS packing bug — the q8 `load_group` returned dequantized floats where the caller stages raw half pairs. Fixed by packing back to halves. Post-fix GPU test: |q8−f16| ≤ 0.135 across all shapes, zero non-finite. |
| q8 quality (needle-in-haystack ×5, ~12K depth, greedy) | PASS: q8 5/5 = f16 5/5, identical latency (~23–25 s/question). |
| q8 deep single-stream | PASS: 184K-token prompt in 151 s, both planted needles retrieved; 20K-token follow-up turn restored+answered in 2–3 s; q8 snapshots ≈ 53% of f16 size. |
| Arm A: 4 sessions + q8, captured workload | FAIL: GPU memory-access fault ~7 min in (MTP × q8 × multi-session concurrency; bisected clean with MTP off; not a race; single-stream clean). All 12 requests failed; production restored cleanly. |
| Arm B: 8 sessions + `--history-pool-gib 26`, f16 | FAIL: 12/12 `generation_failed`, consistent with elastic budget exhaustion (5 × ~150K f16 lineages ≈ 24.5 GB + floors ≈ 4.8 GB > 26 GiB budget / VRAM ceiling). Elastic mechanics themselves worked (8-session boot at 94.7 GB device, growth, snapshots). |
| Conclusions | (1) Elastic and q8 are co-dependent on this host: five deep lineages only fit with q8's halved footprint. (2) The MTP×q8 concurrency fault is the one blocking kernel bug — batched two-session q8 MTP unit test is the required repro before the next window. (3) Next configuration to validate: `--sessions 6..8 --history-pool-gib 16..18 --kv-quant`. Full detail: `ROUND8-RESULT-20260927.md` in the research artifacts. |

## 2026-09-27 round-9 readiness: fault fixed, observability wired, endgame staged

| Follow-up | Outcome |
| --- | --- |
| MTP×q8 concurrency fault (round 8, Arm A) | Root-caused and fixed: the batched MTP catch-up path (`batch.cpp`) hand-built its per-session `AttentionState` from the raw-cache fields only — under `--kv-quant` those are null (the packed stores are the caches), so every batch of ≥2 sessions dereferenced null in the draft attention; single-session short-circuits to `MtpForward`, which is exactly why the fault needed concurrency. Fix: `Session::MtpAttentionView()` is now the single constructor for draft-side views (used by `MtpForward` and the batched catch-up), so a future cache family cannot miss a site; plus a defensive guard in `Executor::Attention` fails the request cleanly ("quantized attention view is missing its packed stores") instead of faulting the GPU if a view is ever built wrong again. |
| Elastic observability | `Executor::Options::history_logger` (server-injected callback, model layer stays logging-free): history growth logs `[engine] history growth: session capacity N positions, live L of B budget bytes`; budget refusals log the full error at ERROR. The HTTP layer now logs the swallowed exception under `generation_failed` (`[chat] generation failed: <what>`) instead of dropping it — round 8's silent `generation_failed` class is diagnosable from logs alone. |
| Build fallout | `openai_chat_test`/`tool_batch_validation_test` link `gufo_logging` (they compile the chat streaming TU). CPU suite back to 95/99 (the same 4 pre-existing artifact-gated failures). |
| Round-9 assets staged | `gufo-candidate-round9.bin` (sha `61a40f81…`, fixes + observability); `gufo-candidate-wave32.bin` variant with a proper wave32 GEMM TU (`w8a8_wave32.hip`, `-mwavefrontsize32` default, WM×WN=8 per the template's static assert) and a `GUFO_WAVE32_MOE=1` env dispatch so one binary A/Bs both schedules; `run-round9.sh` (gates → 4-conversation × 2-turn concurrency smoke, which is the exact round-8 fault scenario, then the captured-workload arm, restore, aggregate; aborts and restores production on any smoke failure); `round9_concurrency_smoke.py`. |
| Round-9 configuration and arithmetic | `R9_SESSIONS=8 R9_POOL_GIB=16 ./run-round9.sh`. q8 floors 8 × ~0.29 GB ≈ 2.3 GB + five ~150K lineages × ~2.45 GB ≈ 12.3 GB ≈ 14.6 GB of the 16 GiB budget (refusals now visible in logs); VRAM ≈ boot 92 GB + ~12 GB growth ≈ 104 GB of 124. Halogen target stands at 388 s p50 / 80.8% cache. |

## 2026-09-28 round-10: decode/retention search; q8-4s stands as champion

Two parallel code analyses (batched-decode round anatomy; elastic-hang
hypotheses with pointer-staleness provably ruled out) drove five more arms.
Implemented and retained: stream-ordered growth (`hipMallocAsync`/
`hipFreeAsync`), all-or-nothing growth with a fail-closed corrupt-session
flag, refusal logging outside the budget mutex, and a batched greedy
verification epilogue (one argmax + one sync per decode round). A device
slab allocator for growth was attempted and reverted: it produced a
deterministic GPU memory fault at the first growths.

| Arm | Result |
| --- | --- |
| graphs-off + stream-ordered elastic (H2 isolation) | Hang persists identically — graph re-capture racing snapshot D2H eliminated; the wedge is at the ROCm driver/VM level under concurrent deep growth. Elastic remains default-off; next step is device-level capture at the freeze (`AMD_LOG_LEVEL=4`, `HSA_ENABLE_SDMA=0`, /proc kernel stacks — no gdb on this host). |
| fixed-6 q8 (no growth, 262144 sessions) | 8/12, cache 23.3%, decode 0.4 tok/s — regression: decode per request falls as 1/sessions and the host snapshot budget fragments across more lineages. More sessions is not automatically better. |
| **Standing champion: q8, 4 fixed sessions** (`--kv-quant --sessions 4 --host-snapshot-gib 9`) | **12/12, duration p50 680 s (−18% vs f16), cache 64.9%, decode parity, zero faults** (round-9). |

The decode-side lever (aggregate batched decode 18–25 vs 35 tok/s serial —
eager launches, 13–15 syncs/round, per-session epilogue, mid-burst snapshot
ejections; blueprint and ranked fixes recorded in
`ROUND10-RESULT-20260928.md`) is the next major optimization; the elastic
hang requires device-level diagnosis first.

## 2026-09-27 round-9 window: KV q8_0 validated (−18% p50); elastic hang isolated

Five arms in one passthrough window (users transparently on an upstream
provider via the new `gpas` tailscale passthrough; local halogen restored
healthy after every arm).

| Arm | Result |
| --- | --- |
| Concurrency smoke (round-8 fault scenario, 4 convos × 2 turns, q8+elastic) | PASS — the `MtpAttentionView` fix holds; turn-2 restores 3–4 s. |
| q8+elastic arms (8×16 GiB, 8×24 GiB, 6×24 GiB graveyard) | 0/12, 0/12, aborted: the first two hit budget exhaustion (correctly logged and fail-closed), then a **silent GPU pipeline hang** ~675–900 s in — reproduced across f16/q8 and three free strategies (immediate `hipFree`, graveyard deferral, stream-ordered `hipMallocAsync`/`hipFreeAsync`), always after large growths under ≥5-session concurrent deep decode; fixed capacity never hangs. Below the free mechanics; needs device-level queue diagnosis. |
| **q8-only isolation arm (4 sessions, fixed, `--kv-quant`)** | **12/12, duration p50 680 s (−18% vs round-6 f16's 829 s), cache share 64.9% (vs 58.7% — halved payloads fit more lineages in the 9 GiB host budget), decode 1.7 tok/s (parity), zero faults.** |
| Decisions | (1) `--kv-quant` at fixed capacity is the new promoted-candidate configuration (`gufo-candidate-round9c.bin`, sha 9aed027b…; payload version bump discards old f16 snapshots — one cold re-prefill). (2) Elastic stays default-off and blocked on hang diagnosis: reproduce with the concurrency smoke at 24 GiB elastic and capture device queue state at the freeze instead of further replay arms. (3) Growth is now stream-ordered and fully logged; the round-8 fault class remains fixed. |
