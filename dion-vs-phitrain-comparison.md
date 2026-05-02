# Dion vs Phitrain Training Configuration Comparison

Comparison of `dion` (this repo) with `phitrain` (`~/aifsdk.worktrees/dion-speed/phitrain/`), focusing on the `dion_160m.yaml` and `mixformer-1b.yaml` configs.

## Configuration System

| | **dion** | **phitrain** |
|---|---|---|
| Format | Flat YAML + argparse CLI overrides | Deeply nested YAML with `omegaconf` (interpolation, env vars) |
| Parsing | `yaml.safe_load` → copy to `argparse.Namespace` | Structured dataclass schemas (`StrategyArguments`, etc.) |
| Training loop | Custom hand-written loop | PyTorch Lightning `Trainer` |
| Stopping criterion | Fixed `num_iterations` | Token budget (`max_tokens`) via `plan_manager` |
| LR schedule | `warmup_ratio` / `warmdown_ratio` (fractions) | Named scheduler type with explicit step counts and cooldown |

## Distributed Training

### FSDP
Both use PyTorch native FSDP2 (`fully_shard`) with per-layer wrapping + root wrapping. Same `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`.

| | **dion** | **phitrain** |
|---|---|---|
| Framework | Raw PyTorch + torchrun | Lightning `ModelParallelStrategy` |
| Mesh dims | `(dp, fs, tp)` — 3D | `(data_parallel_replicate, data_parallel_shard, context_parallel, tensor_parallel)` — 4D |
| Context parallel | Not supported | Supported |
| `fast_fsdp` flag | `reshard_after_forward=False` on all layers (speed vs memory) | No equivalent |
| CPU offload | Not supported | Supported |
| TP+FSDP shard placement | Custom `shard_placement_fn` to shard opposite dim | Not visible (handled separately) |

### DDP
| | **dion** | **phitrain** |
|---|---|---|
| Implementation | Manual `torch.nn.parallel.DDP` wrapper; optimizer gets `process_group` directly | Lightning's built-in `DDPStrategy` (just a config string) |
| Fallback | DDP is the default when no mesh dims specified | Explicit `strategy.type: "ddp"` |

## Precision

Both configs produce the same effective precision:
- Forward/backward: bf16 autocast
- FSDP params: bf16, reductions: fp32
- Optimizer states: bf16

| | **dion** | **phitrain** |
|---|---|---|
| Autocast | Hardcoded `torch.amp.autocast(dtype=bf16)` | Configurable via `trainer.precision: "bf16-mixed"` |
| FSDP dtypes | Hardcoded | Configurable (`fsdp_param_dtype`, `fsdp_reduce_dtype`) |
| FP8 support | No | Yes (fp8 all-gather, quantized matmuls) |
| User knobs | 1 flag (`mixed_precision` for optimizer states) | ~6 independent settings |

## Performance-Relevant Differences

### Attention
| | **dion** | **phitrain (mixformer-1b)** |
|---|---|---|
| Kernel | `F.scaled_dot_product_attention` → dispatches to FA2 | Auto-selects FA4 → FA2 → fallback |
| Type | Standard causal MHA | Gated attention + sliding window (2048) + global every 4 layers |
| KV heads | Full MHA (n_head == n_kv_head) | GQA (`num_key_value_heads: 8`) |
| Seq length | 1024 | 8192 |

### torch.compile
| | **dion** | **phitrain** |
|---|---|---|
| Granularity | Two graphs: `_forward` + `_forward_emb` | Per-layer compile before FSDP, then whole model |

### Data Loading
| | **dion** | **phitrain** |
|---|---|---|
| Loader | Custom mmap binary, single-threaded | `num_workers: 1`, `prefetch_factor: 256` |

### Other
| | **dion** | **phitrain** |
|---|---|---|
| Activation checkpointing | Not supported | Supported (not enabled in mixformer-1b) |
| Gradient clipping | Handled in optimizer | `gradient_clip_val: 1.0` via Lightning |
| MLP | ReLU² 4x expansion | GLU (SiLU-gated) |

## Key Speed Factors (phitrain advantages on B200)

1. **Flash Attention 4** — used automatically; dion gets FA2 via SDPA
2. **Sliding window attention** — most layers attend only 2048 tokens, not full sequence
3. **GQA** — fewer KV heads → less memory bandwidth in attention
4. **Per-layer compile** — tighter fusion with FSDP resharding boundaries
5. **Data prefetching** — large prefetch buffer reduces stalls
