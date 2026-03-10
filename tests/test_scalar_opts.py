import pytest
import torch

from dion.muon import Muon
from dion.normuon import NorMuon
from dion.scalar_opts import (
    adamw_update_auto,
    adamw_update_foreach,
    validate_scalar_backend,
)

# Allow enough compile cache entries for the foreach fallback tests.
torch._dynamo.config.cache_size_limit = 100  # noqa: SLF001

CUDA_AVAILABLE = torch.cuda.is_available()


def _clone_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.clone() for tensor in tensors]


def _assert_lists_close(lhs: list[torch.Tensor], rhs: list[torch.Tensor], *, tol: float):
    assert len(lhs) == len(rhs)
    for left, right in zip(lhs, rhs, strict=True):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.allclose(left, right, atol=tol, rtol=tol), (
            f"max-abs-diff {(left - right).abs().max().item():.3e} > {tol}"
        )


def _run_adamw_pair(
    *,
    device: str,
    dtype: torch.dtype,
    cautious_wd: bool,
    scalar_backend: str = "auto",
):
    X_ref = [torch.randn(128, device=device, dtype=dtype) for _ in range(4)]
    G_ref = [torch.randn_like(x) for x in X_ref]
    M_ref = [torch.randn_like(x) for x in X_ref]
    V_ref = [torch.rand_like(x).abs_().add_(0.1) for x in X_ref]

    X_test = _clone_tensors(X_ref)
    G_test = _clone_tensors(G_ref)
    M_test = _clone_tensors(M_ref)
    V_test = _clone_tensors(V_ref)

    common_kwargs = dict(
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.01,
        step=3,
        epsilon=1e-8,
        cautious_wd=cautious_wd,
    )

    adamw_update_foreach(
        X_ref,
        G_ref,
        M_ref,
        V_ref,
        lr=torch.tensor(common_kwargs["lr"], device=device),
        beta1=torch.tensor(common_kwargs["beta1"], device=device),
        beta2=torch.tensor(common_kwargs["beta2"], device=device),
        weight_decay=torch.tensor(common_kwargs["weight_decay"], device=device),
        step=common_kwargs["step"],
        epsilon=common_kwargs["epsilon"],
        cautious_wd=common_kwargs["cautious_wd"],
    )
    adamw_update_auto(
        X_test,
        G_test,
        M_test,
        V_test,
        scalar_backend=scalar_backend,
        **common_kwargs,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    tol = 5e-3 if dtype == torch.bfloat16 else 1e-6
    _assert_lists_close(X_ref, X_test, tol=tol)
    _assert_lists_close(M_ref, M_test, tol=tol)
    _assert_lists_close(V_ref, V_test, tol=tol)


def test_validate_scalar_backend_rejects_bad_value():
    with pytest.raises(ValueError, match="Invalid scalar_backend"):
        validate_scalar_backend("bad-backend")


def test_adamw_update_auto_cpu_matches_foreach():
    _run_adamw_pair(device="cpu", dtype=torch.float32, cautious_wd=False)


def test_adamw_update_auto_cpu_cautious_wd_matches_foreach():
    _run_adamw_pair(device="cpu", dtype=torch.float32, cautious_wd=True)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adamw_update_auto_cuda_matches_foreach(dtype: torch.dtype):
    _run_adamw_pair(device="cuda", dtype=dtype, cautious_wd=False)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
def test_adamw_update_auto_cuda_cautious_wd_matches_foreach():
    _run_adamw_pair(device="cuda", dtype=torch.float32, cautious_wd=True)


def test_adamw_update_fused_raises_for_unsupported_configuration():
    X = [torch.randn(16)]
    G = [torch.randn_like(X[0])]
    M = [torch.randn_like(X[0])]
    V = [torch.rand_like(X[0]).abs_().add_(0.1)]

    with pytest.raises(RuntimeError, match="fused AdamW is unavailable"):
        adamw_update_auto(
            X=X,
            G=G,
            M=M,
            V=V,
            lr=1e-3,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.01,
            step=3,
            epsilon=1e-8,
            scalar_backend="fused",
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
@pytest.mark.parametrize(
    ("optimizer_cls", "algorithm"),
    [(Muon, "muon"), (NorMuon, "normuon")],
)
def test_scalar_backend_auto_optimizer_step_smoke(optimizer_cls, algorithm: str):
    linear = torch.nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    vector = torch.nn.Parameter(torch.randn(64, device="cuda", dtype=torch.float32))
    optimizer = optimizer_cls(
        [
            {"params": [linear.weight], "algorithm": algorithm, "lr": 1e-2},
            {"params": [vector], "algorithm": "adamw", "lr": 1e-3},
        ],
        scalar_backend="auto",
        use_triton=True,
    )

    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    loss = linear(x).float().sum() + vector.square().sum() * 1e-4
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(True)
    torch.cuda.synchronize()

    assert torch.isfinite(vector).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
@pytest.mark.parametrize(
    ("optimizer_cls", "algorithm"),
    [(Muon, "muon"), (NorMuon, "normuon")],
)
def test_scalar_backend_fused_ignores_cautious_wd_for_scalar_groups(
    optimizer_cls, algorithm: str
):
    linear = torch.nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    vector = torch.nn.Parameter(torch.randn(64, device="cuda", dtype=torch.float32))
    optimizer = optimizer_cls(
        [
            {"params": [linear.weight], "algorithm": algorithm, "lr": 1e-2},
            {"params": [vector], "algorithm": "adamw", "lr": 1e-3},
        ],
        cautious_wd=True,
        scalar_backend="fused",
        use_triton=True,
    )

    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    loss = linear(x).float().sum() + vector.square().sum() * 1e-4
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(True)
    torch.cuda.synchronize()

    assert torch.isfinite(vector).all()
