#!/usr/bin/env python3
"""
Benchmark the optimizer step in isolation (no forward/backward pass).

Usage:
  # Single GPU (DDP with world_size=1):
  CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 benchmark_optimizer.py --config configs/dion2_160m.yaml

  # Multi-GPU with FSDP:
  torchrun --standalone --nproc_per_node=4 benchmark_optimizer.py --config configs/dion2_160m.yaml --fs_size 4

  # Sweep across configurations:
  CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 benchmark_optimizer.py sweep
  torchrun --standalone --nproc_per_node=4 benchmark_optimizer.py sweep --fs_size 4 --config configs/muon_160m.yaml

  # Programmatic sweep (from another script):
  from benchmark_optimizer import build_and_benchmark, init_distributed_benchmark
  device_mesh = init_distributed_benchmark(dp_size=None, fs_size=None, tp_size=None)
  result = build_and_benchmark(config="configs/dion2_160m.yaml", overrides={"ortho_fraction": 0.5}, device_mesh=device_mesh)
"""

import argparse
import os
import torch
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional

from models.gpt_model import GPT, GPTConfig, parallelize_gpt_model
import train as train_module
from train import (
    Hyperparameters,
    parse_cli_args,
    override_args_from_cli,
    init_optimizer,
    print0,
)


def init_distributed_benchmark(dp_size, fs_size, tp_size) -> Optional[DeviceMesh]:
    """Initialize distributed (same logic as train.py but sets train module's MASTER_PROCESS)."""
    assert torch.cuda.is_available(), "CUDA must be available"
    assert torch.distributed.is_available(), "Distributed must be available"
    assert all(
        var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    ), "This script must be launched using 'torchrun'."

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    train_module.MASTER_PROCESS = rank == 0

    mesh_dims = (dp_size, fs_size, tp_size)
    if all(d is None for d in mesh_dims):
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(f"cuda:{local_rank}")
        return None
    else:
        dp_size = dp_size if dp_size is not None else 1
        fs_size = fs_size if fs_size is not None else 1
        tp_size = tp_size if tp_size is not None else 1
        total_gpus = dp_size * fs_size * tp_size
        assert world_size == total_gpus, (
            f"World size {world_size} != expected {total_gpus} "
            f"(DP {dp_size}, FS {fs_size}, TP {tp_size})"
        )
        torch.cuda.set_device(f"cuda:{local_rank}")
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(dp_size, fs_size, tp_size),
            mesh_dim_names=("dp", "fs", "tp"),
        )
        return device_mesh


def build_model_and_optimizer(hp: Hyperparameters, cli_args: argparse.Namespace, device_mesh):
    """Build the GPT model and optimizer using the same logic as train.py."""
    gpt_config = GPTConfig(
        sequence_len=hp.sequence_length,
        vocab_size=hp.vocab_size,
        n_layer=hp.n_layer,
        n_head=hp.n_head,
        n_embd=hp.model_dim,
        use_bias=hp.use_bias,
    )

    with torch.device("meta"):
        model = GPT(gpt_config)

    if device_mesh is not None:
        parallelize_gpt_model(
            model,
            device_mesh=device_mesh,
            dp_name=(None if hp.replicate_mesh_grad_sync else "dp"),
            fs_name="fs",
            tp_name="tp",
            fsdp_reshard_after_forward=(not cli_args.fast_fsdp),
        )
        raw_model = model
        ddp_model = None
    else:
        raw_model = model
        ddp_model = None

    model.to_empty(device="cuda")
    model.init_weights()

    if device_mesh is None:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        raw_model = model.module
        ddp_model = model

    optimizer = init_optimizer(
        model=raw_model,
        device_mesh=device_mesh,
        ddp_model=ddp_model,
        hp=hp,
        cli_args=cli_args,
    )

    return model, optimizer


def assign_fake_gradients(model):
    """Assign random gradients to all parameters."""
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)


def benchmark_optimizer_step(
    model,
    optimizer,
    num_warmup: int = 25,
    num_iterations: int = 100,
):
    """
    Time optimizer.step() using CUDA events for accurate GPU timing.
    Returns median time in milliseconds.
    """
    # Warmup
    for _ in range(num_warmup):
        assign_fake_gradients(model)
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Timed iterations
    times_ms = []
    for _ in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        assign_fake_gradients(model)

        start.record()
        optimizer.step()
        end.record()

        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

        optimizer.zero_grad()

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    return median_ms, times_ms


def build_and_benchmark(
    config: Optional[str] = None,
    overrides: Optional[dict] = None,
    device_mesh: Optional[DeviceMesh] = None,
    num_warmup: int = 25,
    num_iterations: int = 100,
) -> dict:
    """
    High-level API for sweeps: build model+optimizer from config and benchmark.

    Args:
        config: Path to YAML config file (same as train.py --config).
        overrides: Dict of hyperparameter overrides (e.g. {"ortho_fraction": 0.5}).
        device_mesh: Pre-initialized DeviceMesh, or None for DDP mode.
        num_warmup: Number of warmup iterations before timing.
        num_iterations: Number of timed iterations.

    Returns:
        Dict with keys: "median_ms", "times_ms", "num_params", "hp".

    Note: Distributed must already be initialized before calling this.
          Call init_distributed_benchmark() first.
    """
    # Build a namespace that mimics CLI args
    cli_args = argparse.Namespace(
        config=config,
        dp_size=None, fs_size=None, tp_size=None,
        data_dir=None, checkpoint_dir=None, checkpoint_freq=None,
        optimizer=None, scalar_opt=None, lr=None, adjust_lr=None,
        qr_method=None, mixed_precision=False, ortho_fraction=None,
        mu=None, weight_decay=None, time_optimizer=False,
        model_dim=None, n_layer=None, n_head=None, vocab_size=None,
        num_iterations=None, batch_size=None, device_batch_size=None,
        sequence_length=None, warmup_ratio=None, warmdown_ratio=None,
        no_wandb=True, wandb_project_name=None, wandb_job_name=None,
        replicate_mesh_grad_sync=False, fast_fsdp=False,
        debug=False, no_compile=True, no_triton=False,
        use_polar_express=False, use_gns_package=False, use_gns_alg=False, 
        split_heads=False,
    )

    # Load YAML config if provided
    if config:
        from pathlib import Path
        import yaml
        with Path(config).open("r") as f:
            yaml_cfg = yaml.safe_load(f)
        for k, v in yaml_cfg.items():
            if hasattr(cli_args, k) and getattr(cli_args, k) is None:
                setattr(cli_args, k, v)
            elif hasattr(cli_args, k) and isinstance(getattr(cli_args, k), bool) and not getattr(cli_args, k):
                # Handle store_true flags from YAML
                if yaml_cfg.get(k, False):
                    setattr(cli_args, k, True)

    # Apply overrides
    if overrides:
        for k, v in overrides.items():
            setattr(cli_args, k, v)

    hp = Hyperparameters()
    hp = override_args_from_cli(hp, cli_args)

    model, optimizer = build_model_and_optimizer(hp, cli_args, device_mesh)
    num_params = sum(p.numel() for p in model.parameters())

    median_ms, times_ms = benchmark_optimizer_step(
        model, optimizer, num_warmup=num_warmup, num_iterations=num_iterations
    )

    return {
        "median_ms": median_ms,
        "times_ms": times_ms,
        "num_params": num_params,
        "hp": hp,
    }


def main():
    torch._dynamo.config.cache_size_limit = 100

    cli_args = parse_cli_args()

    # Initialize distributed
    device_mesh = init_distributed_benchmark(
        dp_size=cli_args.dp_size,
        fs_size=cli_args.fs_size,
        tp_size=cli_args.tp_size,
    )

    print0("=" * 80)
    print0("Optimizer Step Benchmark")
    print0(f"GPU: {torch.cuda.get_device_name(0)}")
    print0("=" * 80)

    # Convert cli_args to overrides dict (non-None values only)
    overrides = {k: v for k, v in vars(cli_args).items() if v is not None and k != "config"}
    result = build_and_benchmark(
        config=cli_args.config,
        overrides=overrides,
        device_mesh=device_mesh,
        num_warmup=25,
        num_iterations=100,
    )

    hp = result["hp"]
    print0(f"Optimizer: {hp.optimizer}")
    print0(f"Model dim: {hp.model_dim}, Layers: {hp.n_layer}, Heads: {hp.n_head}")
    print0(f"Total parameters: {result['num_params']:,}")
    print0(f"\nResults:")
    print0(f"  Median: {result['median_ms']:.3f} ms")
    print0(f"  Min:    {min(result['times_ms']):.3f} ms")
    print0(f"  Max:    {max(result['times_ms']):.3f} ms")
    print0(f"  Mean:   {sum(result['times_ms'])/len(result['times_ms']):.3f} ms")
    print0("=" * 80)

    if dist.is_initialized():
        dist.destroy_process_group()


def sweep():
    """
    Run optimizer step benchmark across a predefined list of configurations.
    Edit CONFIG and SWEEP_OVERRIDES below to customize.

    Usage:
      torchrun --standalone --nproc_per_node=1 benchmark_optimizer.py sweep
      torchrun --standalone --nproc_per_node=4 benchmark_optimizer.py sweep --fs_size 4
    """
    torch._dynamo.config.cache_size_limit = 100

    DEFAULT_CONFIG = "configs/benchmark_optimizer.yaml"

    SWEEP_OVERRIDES = [
        {"optimizer": "muon", "no_triton": True, "use_gns_package": False, "use_gns_alg": False, "split_heads": False},
        {"optimizer": "muon", "use_gns_package": False, "use_gns_alg": False, "split_heads": False},
        {"optimizer": "muon", "use_gns_package": True, "use_gns_alg": False, "split_heads": False},
        {"optimizer": "muon", "use_gns_package": True, "use_gns_alg": True, "split_heads": False},
        {"optimizer": "muon", "use_gns_package": True, "use_gns_alg": True, "split_heads": True},
    ] + [
        {"optimizer": "dion2", "use_gns_package": True, "use_gns_alg": True, "split_heads": True, "ortho_fraction": frac}
        for frac in (1.0, 0.5, 0.25, 0.125)
    ] + [
        {"optimizer": "dion2", "use_gns_package": True, "use_gns_alg": True, "split_heads": True, "ortho_fraction": 0.25},
        {"optimizer": "adamw"},
    ]

    cli_args = parse_cli_args()
    config = cli_args.config if cli_args.config else DEFAULT_CONFIG

    # Initialize distributed
    device_mesh = init_distributed_benchmark(
        dp_size=cli_args.dp_size,
        fs_size=cli_args.fs_size,
        tp_size=cli_args.tp_size,
    )

    print0("=" * 80)
    print0("Optimizer Step Sweep")
    print0(f"Config: {config}")
    print0(f"GPU: {torch.cuda.get_device_name(0)}")
    print0(f"Number of configurations: {len(SWEEP_OVERRIDES)}")
    print0("=" * 80)

    # Merge CLI overrides (non-None values) with each sweep configuration
    cli_overrides = {k: v for k, v in vars(cli_args).items() if v is not None and k != "config"}

    results = []
    for i, sweep_overrides in enumerate(SWEEP_OVERRIDES):
        # CLI args serve as base, sweep-specific overrides take precedence
        merged_overrides = {**cli_overrides, **sweep_overrides}
        print0(f"\n--- Configuration {i+1}/{len(SWEEP_OVERRIDES)}: {sweep_overrides} ---")

        result = build_and_benchmark(
            config=config,
            overrides=merged_overrides,
            device_mesh=device_mesh,
            num_warmup=25,
            num_iterations=100,
        )
        result["overrides"] = sweep_overrides
        results.append(result)

        print0(f"  Median: {result['median_ms']:.3f} ms")

    # Summary table
    override_strs = [str(r["overrides"]) for r in results]
    col_width = min(200, max(len(s) for s in override_strs))
    table_width = col_width + 42  # account for other columns and separators
    print0(f"\n{'=' * table_width}")
    print0("SWEEP RESULTS")
    print0(f"{'=' * table_width}")
    print0(f"{'Overrides':<{col_width}s} | {'Median (ms)':>12s} | {'Min (ms)':>10s} | {'Max (ms)':>10s}")
    print0("-" * table_width)
    for r, os_str in zip(results, override_strs):
        print0(
            f"{os_str:<{col_width}s} | {r['median_ms']:>12.3f} | "
            f"{min(r['times_ms']):>10.3f} | {max(r['times_ms']):>10.3f}"
        )
    print0("=" * table_width)

    if dist.is_initialized():
        dist.destroy_process_group()

    return results


def profile():
    """
    Profile optimizer.step() and export a Chrome trace.

    Usage:
      torchrun --standalone --nproc_per_node=1 benchmark_optimizer.py profile --config configs/benchmark_optimizer.yaml
    """
    torch._dynamo.config.cache_size_limit = 100

    cli_args = parse_cli_args()

    device_mesh = init_distributed_benchmark(
        dp_size=cli_args.dp_size,
        fs_size=cli_args.fs_size,
        tp_size=cli_args.tp_size,
    )

    hp = Hyperparameters()
    hp = override_args_from_cli(hp, cli_args)

    model, optimizer = build_model_and_optimizer(hp, cli_args, device_mesh)

    print0("=" * 80)
    print0("Profiling optimizer step")
    print0(f"GPU: {torch.cuda.get_device_name(0)}")
    print0(f"Optimizer: {hp.optimizer}")
    print0(f"Model dim: {hp.model_dim}, Layers: {hp.n_layer}, Heads: {hp.n_head}")
    print0("=" * 80)

    # Warmup
    num_warmup = 5
    for _ in range(num_warmup):
        assign_fake_gradients(model)
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Profile
    num_profiled = 3
    print0(f"Capturing {num_profiled} profiled iterations...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) as prof:
        for _ in range(num_profiled):
            assign_fake_gradients(model)
            optimizer.step()
            optimizer.zero_grad()

    trace_path = "optimizer_trace.json"
    prof.export_chrome_trace(trace_path)
    print0(f"Trace exported to: {trace_path}")
    print0("Open in chrome://tracing or https://ui.perfetto.dev")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sys.argv.pop(1)
        sweep()
    elif len(sys.argv) > 1 and sys.argv[1] == "profile":
        sys.argv.pop(1)
        profile()
    else:
        main()
