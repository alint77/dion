#!/usr/bin/env python3
"""
Profile a FULL training step (forward + backward + optimizer.step), not the
optimizer step in isolation.

Unlike benchmark_optimizer.py -- which assigns fake gradients and times only
optimizer.step() -- this drives a real forward/backward through the GPT model so
the captured trace covers the whole step: embeddings, attention, MLP, the loss,
the backward pass, and the optimizer update. This mirrors what nanoplm's
pure-pipeline profiler captures.

It reuses train.py's model + optimizer construction (via
benchmark_optimizer.build_model_and_optimizer) and train.py's bf16 autocast, but
feeds SYNTHETIC random token batches so no fineweb .bin data files are required
(profiling fwd/bwd/opt is agnostic to token content).

Timing/trace uses the same scheduled PyTorch profiler as
benchmark_optimizer.profile (schedule(wait, warmup, active), Chrome-trace export;
no nsys/NCU mode).

Usage:
  # Single GPU (DDP world_size=1):
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
      profile_training_step.py --config configs/benchmark_optimizer.yaml

  # Multi-GPU FSDP:
  torchrun --standalone --nproc_per_node=4 \
      profile_training_step.py --config configs/benchmark_optimizer.yaml --fs_size 4

  # Options:
  #   --profile_wait 5 --profile_warmup 1 --profile_active 3 --profile_out .
  #   --grad_accum 1   (micro-steps of fwd/bwd per optimizer step)
"""

import argparse
import os
import time
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FSDPModule

from train import (
    Hyperparameters,
    parse_cli_args,
    override_args_from_cli,
    print0,
)
from benchmark_optimizer import (
    init_distributed_benchmark,
    build_model_and_optimizer,
    _make_bench_profiler,
)


def synthetic_batch(hp: Hyperparameters):
    """Random (x, y) token batch on the GPU, shaped like train.py's loader:
    x, y are (device_batch_size, sequence_length) int64 token ids, y = next-token.
    """
    B = hp.device_batch_size
    T = hp.sequence_length
    # +1 so y is a shifted view of the same stream, matching next_batch().
    buf = torch.randint(
        0, hp.vocab_size, (B * T + 1,), device="cuda", dtype=torch.long
    )
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    return x, y


def training_step(
    model, optimizer, x, y, autocast_ctx, grad_accum_steps: int, sync_log: bool = True
):
    """One full optimizer step: grad_accum_steps micro fwd/bwd, then step().

    Returns a dict with keys: ``grad_norm``, ``loss`` (host float, NaN if not
    ``sync_log``), and -- when ``sync_log`` -- GPU/CPU timings:
    ``gpu_step_ms`` (whole step), ``gpu_opt_ms`` (optimizer.step only),
    ``cpu_opt_ms`` (host time in optimizer.step), and ``opt_gpu_pct``
    (optimizer's share of total GPU step time).

    The loss/backward path matches train.py's inner loop, minus the data loader
    and the DDP/FSDP grad-sync toggling.

    When ``sync_log`` is True, a ``torch.cuda.synchronize()`` is issued after the
    step and the accumulated loss is read back to the host (``.item()``). This is
    the nanochat/nanoplm logging pattern: the CPU waits for the step's GPU work
    to finish before reading the loss scalar, so the per-step log line reflects
    completed work and each step's CPU window closes on real GPU completion
    (collapsing the CPU/GPU launch skew at the logging boundary). Set False to
    keep the pure async launch path. GPU timings use cuda events (device time,
    async) and are only read when sync_log forces the synchronize.
    """
    if sync_log:
        # cuda events bracket the whole step and the optimizer.step alone, so we
        # can report the optimizer's share of GPU time. Events are async markers;
        # elapsed_time is read after the synchronize below.
        ev_step_start = torch.cuda.Event(enable_timing=True)
        ev_opt_start = torch.cuda.Event(enable_timing=True)
        ev_opt_end = torch.cuda.Event(enable_timing=True)
        ev_step_end = torch.cuda.Event(enable_timing=True)
        ev_step_start.record()

    loss_accum = 0.0
    for i in range(1, grad_accum_steps + 1):
        with autocast_ctx:
            loss = model(x, y)
        loss = loss / grad_accum_steps
        # Overlap: next micro-batch is generated while backward runs, same as
        # train.py which calls next_batch() right after the forward.
        if isinstance(model, FSDPModule):
            model.set_is_last_backward(i == grad_accum_steps)
            model.set_requires_gradient_sync(True)
        loss.backward()
        loss_accum = loss_accum + loss.detach()

    grad_norm = torch.nn.utils.get_total_norm(
        [p.grad for p in model.parameters() if p.grad is not None]
    )
    if sync_log:
        ev_opt_start.record()
        cpu_opt_t0 = time.perf_counter()
    optimizer.step()
    if sync_log:
        cpu_opt_ms = (time.perf_counter() - cpu_opt_t0) * 1000.0
        ev_opt_end.record()
    model.zero_grad(set_to_none=True)

    out = {"grad_norm": grad_norm, "loss": float("nan")}
    if sync_log:
        # nanochat-style logging sync point: block on the step's GPU work, then
        # read the loss scalar and the cuda-event timings to the host.
        ev_step_end.record()
        torch.cuda.synchronize()
        out["loss"] = loss_accum.item()
        gpu_step_ms = ev_step_start.elapsed_time(ev_step_end)
        gpu_opt_ms = ev_opt_start.elapsed_time(ev_opt_end)
        out.update(
            gpu_step_ms=gpu_step_ms,
            gpu_opt_ms=gpu_opt_ms,
            cpu_opt_ms=cpu_opt_ms,
            opt_gpu_pct=100.0 * gpu_opt_ms / gpu_step_ms if gpu_step_ms > 0 else 0.0,
        )
    return out


# Profiler-window knobs, set by the __main__ dispatch after popping the flags
# from argv (train.parse_cli_args rejects unknown flags, so they can't go
# through the shared parser). Defaults match benchmark_optimizer.profile.
PROFILE_WAIT = 5
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 3
PROFILE_OUT = "."
GRAD_ACCUM = 1
COMPILE = True  # torch.compile the model (dynamic=False); --no_compile disables
SYNC_LOG = True  # nanochat-style cuda-sync + loss readback per step; --no_sync_log disables


def main():
    torch._dynamo.config.cache_size_limit = 100

    cli_args = parse_cli_args()

    wait, warmup, active = PROFILE_WAIT, PROFILE_WARMUP, PROFILE_ACTIVE
    output_dir = PROFILE_OUT
    grad_accum_steps = GRAD_ACCUM

    device_mesh = init_distributed_benchmark(
        dp_size=cli_args.dp_size,
        fs_size=cli_args.fs_size,
        tp_size=cli_args.tp_size,
    )

    hp = Hyperparameters()
    hp = override_args_from_cli(hp, cli_args)

    model, optimizer = build_model_and_optimizer(hp, cli_args, device_mesh)
    num_params = sum(p.numel() for p in model.parameters())

    # torch.compile the model, matching nanoplm's pure pipeline
    # (torch.compile(model, dynamic=False)). build_model_and_optimizer does not
    # compile, so without this the profiled step runs eager -- unlike nanoplm.
    # dynamic=False: synthetic batches are fixed-shape, so specialize once.
    # GPT.compile() compiles the inner _forward methods in place (torchtitan
    # embedding workaround) and preserves the module type, so the FSDPModule
    # isinstance check in training_step still holds. It takes the `dynamic` kwarg.
    if COMPILE:
        model.compile(dynamic=False)

    is_main = int(os.environ.get("RANK", 0)) == 0
    print0("=" * 80)
    print0("Full training-step profile (forward + backward + optimizer.step)")
    print0(f"GPU: {torch.cuda.get_device_name(0)}")
    print0(f"Optimizer: {hp.optimizer}   Scalar opt: {hp.scalar_opt}")
    print0(
        f"Model dim: {hp.model_dim}, Layers: {hp.n_layer}, Heads: {hp.n_head}, "
        f"seq_len: {hp.sequence_length}, device_batch: {hp.device_batch_size}"
    )
    print0(f"Total parameters: {num_params:,}")
    print0(f"Grad-accum steps per optimizer step: {grad_accum_steps}")
    print0(f"torch.compile: {COMPILE} (dynamic=False)")
    print0(f"Per-step logging cuda-sync: {SYNC_LOG}")
    print0("=" * 80)

    # bf16 autocast, same as train.py.
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    x, y = synthetic_batch(hp)

    # Warmup (compile + caches) before the scheduled window opens. The schedule
    # also has its own `wait`/`warmup` phases, but a couple of eager warmups here
    # keep the first captured step from paying one-time setup costs.
    for _ in range(2):
        training_step(model, optimizer, x, y, autocast_ctx, grad_accum_steps, sync_log=SYNC_LOG)
        x, y = synthetic_batch(hp)
    torch.cuda.synchronize()

    prof_ctx, profiler_step_cb = _make_bench_profiler(
        output_dir=output_dir, is_main=is_main, wait=wait, warmup=warmup, active=active
    )

    total_iters = wait + warmup + active
    print0(
        f"Running {total_iters} steps ({wait} wait + {warmup} warmup + "
        f"{active} active), each = {grad_accum_steps} fwd/bwd + 1 opt.step..."
    )
    with prof_ctx:
        t_prev = time.perf_counter()
        for it in range(total_iters):
            m = training_step(
                model, optimizer, x, y, autocast_ctx, grad_accum_steps, sync_log=SYNC_LOG
            )
            x, y = synthetic_batch(hp)
            profiler_step_cb()
            # Per-step log line. The synchronize inside training_step (sync_log)
            # means dt spans real GPU completion, nanochat-style, and the cuda
            # events give per-step GPU time + the optimizer's GPU/CPU cost and
            # its share of total GPU step time.
            t_now = time.perf_counter()
            gn = m["grad_norm"]
            gn = gn.item() if hasattr(gn, "item") else float(gn)
            if SYNC_LOG:
                print0(
                    f"[step {it}/{total_iters}] loss={m['loss']:.4f} "
                    f"grad_norm={gn:.4f} dt={(t_now - t_prev) * 1000:.1f}ms "
                    f"gpu_step={m['gpu_step_ms']:.2f}ms "
                    f"opt_gpu={m['gpu_opt_ms']:.2f}ms opt_cpu={m['cpu_opt_ms']:.2f}ms "
                    f"opt/gpu={m['opt_gpu_pct']:.1f}%"
                )
            else:
                print0(
                    f"[step {it}/{total_iters}] grad_norm={gn:.4f} "
                    f"dt={(t_now - t_prev) * 1000:.1f}ms"
                )
            t_prev = t_now

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys

    # Extract our extra flags before parse_cli_args (train.py's parser rejects
    # unknown args). Same pattern as benchmark_optimizer.py's profile dispatch.
    def _pop_flag(name, default, cast):
        if name in sys.argv:
            idx = sys.argv.index(name)
            val = sys.argv[idx + 1]
            sys.argv.pop(idx)
            sys.argv.pop(idx)
            return cast(val)
        return default

    def _pop_bool(name):
        # store_true style: present -> True, absent -> False.
        if name in sys.argv:
            sys.argv.pop(sys.argv.index(name))
            return True
        return False

    PROFILE_WAIT = _pop_flag("--profile_wait", 5, int)
    PROFILE_WARMUP = _pop_flag("--profile_warmup", 1, int)
    PROFILE_ACTIVE = _pop_flag("--profile_active", 3, int)
    PROFILE_OUT = _pop_flag("--profile_out", ".", str)
    GRAD_ACCUM = _pop_flag("--grad_accum", 1, int)
    COMPILE = not _pop_bool("--no_compile")
    SYNC_LOG = not _pop_bool("--no_sync_log")
    main()
