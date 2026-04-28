"""
Audit tests for communication behavior of sharded DTensors:

1. Batch-sharded 3D DTensors (sharded on dim 0) must incur NO communication.
2. Matrix-sharded 2D DTensors (sharded on dim 0 or 1) MUST use communication.

Both tests are run in two modes:
  a) Manual DTensor construction
  b) FSDP2 fully_shard() on a real module

Run with:
    torchrun --standalone --nproc_per_node=4 tests/test_batch_sharded_no_comms.py
"""

import sys
import unittest.mock as mock

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DeviceMesh, DTensor, Shard

from dion import Muon, Dion2, NorMuon


class ToyModel(nn.Module):
    """Simple model with both 3D (conv-like) and 2D (linear) parameters."""
    def __init__(self, n_batch=20, rows=32, cols=64, hidden=64, out=32):
        super().__init__()
        # 3D parameter: (n_batch, rows, cols) — batch dim will be sharded by FSDP
        self.conv_weight = nn.Parameter(torch.randn(n_batch, rows, cols))
        # 2D parameter: (hidden, out) — a matrix dim will be sharded by FSDP
        self.linear = nn.Linear(hidden, out, bias=False)

    def forward(self, x):
        # Dummy forward — just use both params so gradients flow
        return (x @ self.linear.weight.T).sum() + self.conv_weight.sum()


def describe_dtensor(name, dt):
    """Print shape, local shape, and sharding placement of a DTensor."""
    if isinstance(dt, DTensor):
        placements_str = ", ".join(str(p) for p in dt.placements)
        print(f"    {name}: full_shape={tuple(dt.shape)}, "
              f"local_shape={tuple(dt.to_local().shape)}, "
              f"placements=[{placements_str}]")
    else:
        print(f"    {name}: shape={tuple(dt.shape)} (not a DTensor)")


def run_manual_tests(rank, mesh, device, optimizers_to_test, original_all_to_all, original_all_gather):
    """Test 1: Manual DTensor construction."""
    if rank == 0:
        print("--- Test 1: Manual DTensors ---")
        print("--- 1a: Batch-sharded 3D tensors must NOT communicate ---")

    world_size = mesh.size()
    torch.manual_seed(42)
    local_data1 = torch.randn(20, 32, 64, device=device)
    local_data2 = torch.randn(20, 32, 64, device=device)

    for name, opt_cls, opt_kwargs in optimizers_to_test:
        p1 = torch.nn.Parameter(
            DTensor.from_local(local_data1.clone(), device_mesh=mesh, placements=[Shard(0)])
        )
        p2 = torch.nn.Parameter(
            DTensor.from_local(local_data2.clone(), device_mesh=mesh, placements=[Shard(0)])
        )
        if rank == 0:
            describe_dtensor("p1", p1)
            describe_dtensor("p2", p2)

        optimizer = opt_cls([p1, p2], distributed_mesh=mesh, **opt_kwargs)

        torch.manual_seed(100 + rank)
        p1.grad = DTensor.from_local(
            torch.randn_like(local_data1), device_mesh=mesh, placements=[Shard(0)]
        )
        p2.grad = DTensor.from_local(
            torch.randn_like(local_data2), device_mesh=mesh, placements=[Shard(0)]
        )

        def fail_all_to_all(*args, **kwargs):
            raise AssertionError(f"{name}: unexpected dist.all_to_all call!")
        def fail_all_gather(*args, **kwargs):
            raise AssertionError(f"{name}: unexpected dist.all_gather call!")

        with mock.patch.object(dist, "all_to_all", side_effect=fail_all_to_all), \
             mock.patch.object(dist, "all_gather", side_effect=fail_all_gather):
            try:
                optimizer.step()
                if rank == 0:
                    print(f"  PASS: {name} — no communication")
            except AssertionError as e:
                if rank == 0:
                    print(f"  FAIL: {name} — {e}")
                return False

    if rank == 0:
        print("\n--- 1b: Matrix-sharded 2D tensors MUST communicate ---")

    for name, opt_cls, opt_kwargs in optimizers_to_test:
        torch.manual_seed(42)
        local_shape = (128 // world_size, 64)
        p = torch.nn.Parameter(
            DTensor.from_local(
                torch.randn(*local_shape, device=device),
                device_mesh=mesh, placements=[Shard(0)],
            )
        )
        if rank == 0:
            describe_dtensor("p", p)

        optimizer = opt_cls([p], distributed_mesh=mesh, **opt_kwargs)
        torch.manual_seed(100 + rank)
        p.grad = DTensor.from_local(
            torch.randn(*local_shape, device=device),
            device_mesh=mesh, placements=[Shard(0)],
        )

        comm_called = False
        def tracking_all_to_all(*args, _orig=original_all_to_all, **kwargs):
            nonlocal comm_called
            comm_called = True
            return _orig(*args, **kwargs)
        def tracking_all_gather(*args, _orig=original_all_gather, **kwargs):
            nonlocal comm_called
            comm_called = True
            return _orig(*args, **kwargs)

        with mock.patch.object(dist, "all_to_all", side_effect=tracking_all_to_all), \
             mock.patch.object(dist, "all_gather", side_effect=tracking_all_gather):
            optimizer.step()

        if comm_called:
            if rank == 0:
                print(f"  PASS: {name} Shard(0) — communication occurred as expected")
        else:
            if rank == 0:
                print(f"  FAIL: {name} Shard(0) — no communication detected!")
            return False

    return True


def run_fsdp_tests(rank, mesh, device, optimizers_to_test, original_all_to_all, original_all_gather):
    """Test 2: FSDP2 fully_shard() on a real module."""
    if rank == 0:
        print("\n--- Test 2: FSDP2 fully_shard() module ---")

    for name, opt_cls, opt_kwargs in optimizers_to_test:
        # Create model on meta device, apply FSDP, then materialize
        model = ToyModel()
        model.to(device)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        fully_shard(model, mesh=mesh, mp_policy=mp_policy)

        if rank == 0:
            print(f"\n  {name}:")
            for pname, p in model.named_parameters():
                describe_dtensor(pname, p)

        # Forward + backward to generate gradients
        torch.manual_seed(100 + rank)
        x = torch.randn(4, 64, device=device)
        loss = model(x)
        loss.backward()

        # Single optimizer with all params
        optimizer = opt_cls(list(model.parameters()), distributed_mesh=mesh, **opt_kwargs)

        # Track the ndim of tensors involved in communication.
        # In the all_to_all path, input_tensor_list items have an extra stacked dim,
        # so a 2D param becomes 3D after stack, and a 3D param becomes 4D.
        # We record the ndim of the communicated chunks.
        comm_ndims = set()

        def tracking_all_to_all(*args, _orig=original_all_to_all, **kwargs):
            # args: (output_tensor_list, input_tensor_list, ...)
            if len(args) >= 2:
                for t in args[1]:
                    comm_ndims.add(t.ndim)
            return _orig(*args, **kwargs)

        def tracking_all_gather(*args, _orig=original_all_gather, **kwargs):
            # args: (output_tensor_list, input_tensor, ...)
            if len(args) >= 2:
                comm_ndims.add(args[1].ndim)
            return _orig(*args, **kwargs)

        with mock.patch.object(dist, "all_to_all", side_effect=tracking_all_to_all), \
             mock.patch.object(dist, "all_gather", side_effect=tracking_all_gather):
            optimizer.step()

        # 2D params (stacked → 3D in comm) should have communicated
        if 3 not in comm_ndims:
            if rank == 0:
                print(f"    FAIL: expected communication for 2D params (ndims seen: {comm_ndims})")
            return False
        # 3D params (stacked → 4D in comm) should NOT have communicated
        if 4 in comm_ndims:
            if rank == 0:
                print(f"    FAIL: unexpected communication for 3D params (ndims seen: {comm_ndims})")
            return False

        if rank == 0:
            print(f"    PASS: 2D params communicated, 3D params did not (comm ndims: {comm_ndims})")

    return True


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = DeviceMesh("cuda", list(range(world_size)))

    if rank == 0:
        print(f"  world_size={world_size}\n")

    # Capture originals once before any patching
    original_all_to_all = dist.all_to_all
    original_all_gather = dist.all_gather

    optimizers_to_test = [
        ("Muon", Muon, dict(lr=0.01, flatten=False)),
        ("Dion2", Dion2, dict(lr=0.01, flatten=False)),
        ("NorMuon", NorMuon, dict(lr=0.01, flatten=False)),
    ]

    ok = run_manual_tests(rank, mesh, device, optimizers_to_test, original_all_to_all, original_all_gather)
    if not ok:
        dist.destroy_process_group()
        sys.exit(1)

    ok = run_fsdp_tests(rank, mesh, device, optimizers_to_test, original_all_to_all, original_all_gather)
    if not ok:
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        print("\nAll tests passed.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
