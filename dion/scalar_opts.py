from collections import defaultdict
from typing import Generator, List

import torch
from torch import Tensor
from torch.optim.adamw import adamw as torch_adamw

SCALAR_BACKENDS = ("auto", "fused", "foreach")


def validate_scalar_backend(scalar_backend: str) -> str:
    if scalar_backend not in SCALAR_BACKENDS:
        raise ValueError(
            f"Invalid scalar_backend: {scalar_backend}. "
            f"Expected one of {SCALAR_BACKENDS}."
        )
    return scalar_backend


def _as_float(value: float | Tensor) -> float:
    return float(value.item()) if isinstance(value, Tensor) else float(value)


def _as_int(value: int | Tensor) -> int:
    return int(value.item()) if isinstance(value, Tensor) else int(value)


def _to_scalar_tensor(value: float | Tensor, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.tensor(value, device=device)


def _fused_adamw_support_reason(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    cautious_wd: bool,
) -> str | None:
    if cautious_wd:
        return "cautious_wd=True is only implemented by Dion's foreach AdamW path"
    if not X:
        return "empty parameter list"
    tensors = [*X, *G, *M, *V]
    if not all(isinstance(t, Tensor) for t in tensors):
        return "all inputs must be tensors"
    if not all(t.device.type == "cuda" for t in tensors):
        return "fused AdamW requires CUDA tensors"
    if not all(torch.is_floating_point(t) and not torch.is_complex(t) for t in tensors):
        return "fused AdamW requires real floating-point tensors"
    if any(g.is_sparse for g in G):
        return "fused AdamW does not support sparse gradients"
    return None


def _adamw_update_fused(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: float | Tensor,
    beta1: float | Tensor,
    beta2: float | Tensor,
    weight_decay: float | Tensor,
    step: int | Tensor,
    epsilon: float | Tensor,
) -> None:
    grouped_indices: dict[tuple[torch.device, torch.dtype], list[int]] = defaultdict(list)
    for idx, param in enumerate(X):
        grouped_indices[(param.device, param.dtype)].append(idx)

    lr = _as_float(lr)
    beta1 = _as_float(beta1)
    beta2 = _as_float(beta2)
    weight_decay = _as_float(weight_decay)
    step = _as_int(step)
    epsilon = _as_float(epsilon)

    for (device, _dtype), indices in grouped_indices.items():
        params = [X[i] for i in indices]
        grads = [G[i] for i in indices]
        exp_avgs = [M[i] for i in indices]
        exp_avg_sqs = [V[i] for i in indices]

        # Fused AdamW tracks a per-parameter step tensor internally. Dion keeps
        # one scalar step per param-group, so create an ephemeral vector at the
        # prior step value and let the functional kernel advance it once.
        state_steps = list(
            torch.full(
                (len(indices),),
                float(step - 1),
                device=device,
                dtype=torch.float32,
            ).unbind(0)
        )

        torch_adamw(
            params=params,
            grads=grads,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=[],
            state_steps=state_steps,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=True,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=epsilon,
            maximize=False,
        )


@torch.compile(fullgraph=True)
def adamw_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    V: Tensor,  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
):
    """
    AdamW optimizer algorithm.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    M.lerp_(G.to(M.dtype), 1 - beta1)
    # V = beta2 * V + (1 - beta2) * G * G
    V.mul_(beta2).addcmul_(G, G, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = V.sqrt().div_(bias_correction2_sqrt).add_(epsilon)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    if cautious_wd:
        # Compute update direction (pre-LR) for CWD mask
        update_dir = M / denom

        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (update_dir * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    X.addcdiv_(M, denom, value=-adj_lr)


@torch.compile(fullgraph=True)
def lion_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
):
    """
    Lion optimizer algorithm. Sign update should guarantee RMS norm equal to 1.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    G = G.to(M.dtype)

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = M.lerp(G, 1 - beta1).sign_()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * G
    M.lerp_(G, 1 - beta2)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (U * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    X.add_(U, alpha=-lr)


@torch.compile(fullgraph=True)
def adamw_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
):
    """
    AdamW optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)

    # V = beta2 * V + (1 - beta2) * G * G
    G_square = torch._foreach_mul(G, G)
    G_square = [g.to(dtype=V_dtype) for g in G_square]
    torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # Compute the denominator for the weight update
    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    M_div = torch._foreach_div(M, denom)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay

        decay_masks = torch._foreach_mul(X, M_div)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        torch._foreach_mul_(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_sub_(X, M_div)


def adamw_update_auto(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: float | Tensor,
    beta1: float | Tensor,
    beta2: float | Tensor,
    weight_decay: float | Tensor,
    step: int | Tensor,
    epsilon: float | Tensor,
    cautious_wd: bool = False,
    scalar_backend: str = "auto",
):
    """
    AdamW update that prefers PyTorch's native fused CUDA kernel when possible.

    The current Dion foreach implementation is kept as a correctness-preserving
    fallback for CPU tensors and cautious weight decay.
    """
    scalar_backend = validate_scalar_backend(scalar_backend)
    support_reason = _fused_adamw_support_reason(X, G, M, V, cautious_wd)
    use_fused = support_reason is None

    if scalar_backend == "fused" and not use_fused:
        raise RuntimeError(
            "scalar_backend='fused' requested, but fused AdamW is unavailable: "
            f"{support_reason}"
        )
    if scalar_backend == "foreach" or (scalar_backend == "auto" and not use_fused):
        scalar_device = X[0].device if X else torch.device("cpu")
        adamw_update_foreach(
            X=X,
            G=G,
            M=M,
            V=V,
            lr=_to_scalar_tensor(lr, device=scalar_device),
            beta1=_to_scalar_tensor(beta1, device=scalar_device),
            beta2=_to_scalar_tensor(beta2, device=scalar_device),
            weight_decay=_to_scalar_tensor(weight_decay, device=scalar_device),
            step=step,
            epsilon=epsilon,
            cautious_wd=cautious_wd,
        )
        return

    _adamw_update_fused(
        X=X,
        G=G,
        M=M,
        V=V,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        step=step,
        epsilon=epsilon,
    )


@torch.compile(fullgraph=True)
def lion_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
):
    """
    Lion optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)

    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = torch._foreach_lerp(M, G, [1 - beta1] * batch_size)
    torch._foreach_sign_(U)

    # Update momentum in place with new gradient
    # M = beta2 * M + (1 - beta2) * G
    torch._foreach_lerp_(M, G, [1 - beta2] * batch_size)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay

        decay_masks = torch._foreach_mul(X, U)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        torch._foreach_mul_(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    torch._foreach_mul_(U, lr)
    torch._foreach_sub_(X, U)


def adamw_update_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
    scalar_backend: str = "auto",
) -> Generator[None, None, None]:
    adamw_update_auto(
        X=X,
        G=G,
        M=M,
        V=V,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        step=step,
        epsilon=epsilon,
        cautious_wd=cautious_wd,
        scalar_backend=scalar_backend,
    )
    yield


def adamw_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    adamw_update_foreach(
        X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, cautious_wd
    )
    yield


def lion_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay, cautious_wd)
    yield
