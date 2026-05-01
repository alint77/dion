import torch
import torch.distributed as dist
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .megabatch_base import (
    DistributedOrthoBase,
    megabatch_orthogonalize_async,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)
from .opt_utils import AsyncTask, to_local
from .muon import muon_update_pre_orthogonalize


def _canonical_normalization(normalization: Optional[str]) -> Optional[str]:
    if normalization is None:
        return None
    if isinstance(normalization, str):
        value = normalization.lower()
        if value == "none":
            return None
        if value in ("neuron", "short_axis"):
            return value
    raise ValueError(
        f"Invalid normalization: {normalization}. "
        "Must be None, 'None', 'neuron', or 'short_axis'."
    )


class MuonH(DistributedOrthoBase):
    """
    Distributed MuonH optimizer for PyTorch FSDP2. Also compatible with DDP.

    MuonH uses Muon's momentum + orthogonalization direction, then applies a
    hyperball update: the step is scaled relative to each parameter's Frobenius
    norm, and the parameter is projected back to its initial Frobenius norm.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base hyperball learning rate. This controls the relative move size
            before projection back to the Frobenius sphere.
        mu: Momentum factor for Muon.
        muon_beta2: EMA factor for optional normalization buffers.
        normalization: Optional update normalization. None/'None' disables it,
            'neuron' normalizes along the last dimension, and 'short_axis'
            normalizes along the shorter matrix axis.
        betas: Tuple of (beta1, beta2) for AdamW and Lion param groups.
        weight_decay: Weight decay factor for AdamW/Lion param groups.
            MuonH matrix params do not use decoupled weight decay.
        epsilon: Small value for orthogonalization.
        hyperball_eps: Small value for hyperball norm divisions.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate ("spectral_norm" or "rms_norm" or None).
        flatten: Whether to flatten 3D+ tensors to 2D for orthogonalization.
        use_gram_newton_schulz: Whether to use Gram Newton-Schulz for orthogonalization.
        use_triton: Whether to use Triton kernel for Newton-Schulz.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    Hyperball optimization: https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        normalization: Optional[str] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        hyperball_eps: float = 1e-10,
        nesterov: bool = False,
        adjust_lr: Optional[str] = None,
        flatten: bool = False,
        use_gram_newton_schulz: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        newton_schulz_func: Optional[Callable] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if muon_beta2 < 0.0:
            raise ValueError(f"Invalid muon_beta2: {muon_beta2}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if hyperball_eps < 0.0:
            raise ValueError(f"Invalid hyperball_eps: {hyperball_eps}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        normalization = _canonical_normalization(normalization)
        defaults = dict(
            lr=lr,
            mu=mu,
            muon_beta2=muon_beta2,
            normalization=normalization,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="muonh",
            step=0,
            epsilon=epsilon,
            hyperball_eps=hyperball_eps,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(
            params, distributed_mesh, "muonh", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = super()._get_or_initialize_state(param, algo)
        if algo == self._algo_name and "hyperball_radius" not in state:
            local = to_local(param)
            state["hyperball_radius"] = torch.zeros(
                (), device=local.device, dtype=torch.float32
            )
            state["hyperball_radius_initialized"] = False
        return state

    def _get_or_initialize_variance(
        self, param: Tensor, state: dict, normalization: Optional[str]
    ) -> Optional[Tensor]:
        if normalization is None:
            return None

        local = to_local(param)
        shape = list(local.shape)
        if normalization == "neuron":
            shape[-1] = 1
        elif normalization == "short_axis":
            red_dim = -1 if param.shape[-2] >= param.shape[-1] else -2
            shape[red_dim] = 1
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        variance = state.get("variance_normalization")
        if variance is None or tuple(variance.shape) != tuple(shape):
            variance = torch.zeros(shape, device=local.device, dtype=local.dtype)
            state["variance_normalization"] = variance
        return variance

    def _initialize_hyperball_radii(
        self,
        params: List[Tensor],
        states: List[dict],
        process_group: Optional[ProcessGroup],
        shard_dim: Optional[int],
        hyperball_eps: Tensor,
    ):
        if all(s["hyperball_radius_initialized"] for s in states):
            return

        local_params = to_local(params)
        sq_norms = _local_square_sums(local_params)
        if process_group is not None and shard_dim is not None:
            dist.all_reduce(sq_norms, group=process_group)
        radii = sq_norms.sqrt()

        eps = float(hyperball_eps)
        if bool((radii <= eps).any().item()):
            raise ValueError(
                "MuonH requires non-zero matrix parameters because the "
                "hyperball radius is initialized from the parameter norm."
            )

        for state, radius in zip(states, radii):
            if not state["hyperball_radius_initialized"]:
                state["hyperball_radius"].copy_(radius)
                state["hyperball_radius_initialized"] = True

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched MuonH task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "MuonH optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            normalization = _canonical_normalization(group["normalization"])
            hyperball_eps = torch.tensor(group["hyperball_eps"])
            update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                muon_beta2=torch.tensor(group["muon_beta2"]),
                normalization=normalization,
                epsilon=torch.tensor(group["epsilon"]),
                hyperball_eps=hyperball_eps,
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            num_heads = self._resolve_num_heads(group)

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, self._algo_name) for p in params]
                momentums = [s["momentum"] for s in states]

                if num_heads is not None:
                    params, gradients, momentums = self._prepare_head_split(
                        num_heads, params, gradients, momentums
                    )
                    megabatch_args = {**update_args, "process_group": None}
                    shard_dim = None
                else:
                    is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                        self._get_shard_info(params[0], group)
                    )
                    megabatch_args = update_args
                    if is_batch_sharded and not is_matrix_sharded:
                        megabatch_args = {**update_args, "process_group": None}
                    shard_dim = sharded_tensor_dim

                variances = [
                    self._get_or_initialize_variance(p, s, normalization)
                    for p, s in zip(params, states)
                ]
                radii = [s["hyperball_radius"] for s in states]
                self._initialize_hyperball_radii(
                    params=params,
                    states=states,
                    process_group=megabatch_args["process_group"],
                    shard_dim=shard_dim,
                    hyperball_eps=hyperball_eps,
                )

                yield AsyncTask(
                    muonh_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V=variances,
                        R=radii,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
                )


def muonh_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Optional[Tensor]],
    R: List[Tensor],
    lr: Tensor,
    momentum: Tensor,
    muon_beta2: Tensor,
    normalization: Optional[str],
    epsilon: Tensor,
    hyperball_eps: Tensor,
    nesterov: bool,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """
    Mega-batched MuonH update: Muon orthogonalization followed by optional
    normalization and a Frobenius-sphere hyperball step.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(V) == len(R)

    U = muon_update_pre_orthogonalize(
        G=to_local(G), M=to_local(M), momentum=momentum, nesterov=nesterov,
    )

    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None
    U = yield from megabatch_orthogonalize_async(
        U,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=newton_schulz_func,
        flatten=flatten,
        epsilon=epsilon,
    )

    if normalization is not None:
        U = yield from muonh_normalization_async(
            U=U,
            V=V,
            muon_beta2=muon_beta2,
            normalization=normalization,
            param_shape=X[0].shape,
            shard_dim=shard_dim,
            process_group=process_group,
            epsilon=hyperball_eps,
        )

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    norm_group = process_group if shard_dim is not None else None
    yield from muonh_update_post_orthogonalize_async(
        X=to_local(X),
        U=U,
        R=R,
        adjusted_lr=adjusted_lr,
        epsilon=hyperball_eps,
        process_group=norm_group,
    )


def muonh_normalization_async(
    U: List[Tensor],
    V: List[Tensor],
    muon_beta2: Tensor,
    normalization: str,
    param_shape: torch.Size,
    shard_dim: Optional[int],
    process_group: Optional[ProcessGroup],
    epsilon: Tensor,
) -> Generator[None, None, List[Tensor]]:
    """Optional MuonH update normalization, with reductions when the normalized axis is sharded."""
    U_stacked = torch.stack(U)
    V_stacked = torch.stack(V)

    red_dim = -1
    if normalization == "short_axis" and param_shape[-2] < param_shape[-1]:
        red_dim = -2

    sum_sq = U_stacked.float().square().sum(dim=red_dim, keepdim=True)
    shard_dim_neg = None
    if shard_dim is not None:
        shard_dim_neg = shard_dim if shard_dim < 0 else shard_dim - len(param_shape)
    if process_group is not None and shard_dim_neg == red_dim:
        work = dist.all_reduce(sum_sq, group=process_group, async_op=True)
        yield
        work.wait()

    red_dim_size = param_shape[red_dim]
    variance_new = sum_sq / red_dim_size
    norm_U = U_stacked.float().norm(p=2, dim=(-2, -1), keepdim=True)

    V_dtype = V_stacked.dtype
    V_stacked = torch.lerp(V_stacked, variance_new.to(dtype=V_dtype), 1 - muon_beta2)

    normalized_U = U_stacked.float() / (V_stacked.float().sqrt() + float(epsilon))
    norm_U_new = normalized_U.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=float(epsilon))
    normalized_U = normalized_U * (norm_U / norm_U_new)
    normalized_U = normalized_U.to(dtype=V_dtype)

    for i in range(len(V)):
        V[i].copy_(V_stacked[i])
    return [normalized_U[i] for i in range(len(U))]


def muonh_update_post_orthogonalize_async(
    X: List[Tensor],
    U: List[Tensor],
    R: List[Tensor],
    adjusted_lr: Tensor,
    epsilon: Tensor,
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, None]:
    """Apply a scale-invariant hyperball step and project back to stored radii."""
    device = X[0].device
    eps = float(epsilon)
    radii = torch.stack([r.to(device=device, dtype=torch.float32) for r in R])
    lr = adjusted_lr.to(device=device, dtype=torch.float32)

    update_norms = yield from _fro_norms_async(U, process_group)
    step_scales = lr * radii / update_norms.clamp_min(eps)
    candidates = [
        x - u.to(dtype=x.dtype) * step_scales[i].to(device=x.device, dtype=x.dtype)
        for i, (x, u) in enumerate(zip(X, U))
    ]

    candidate_norms = yield from _fro_norms_async(candidates, process_group)
    project_scales = radii / candidate_norms.clamp_min(eps)
    for i, (x, candidate) in enumerate(zip(X, candidates)):
        x.copy_(candidate * project_scales[i].to(device=x.device, dtype=candidate.dtype))


def _local_square_sums(tensors: List[Tensor]) -> Tensor:
    return torch.stack([t.float().square().sum() for t in tensors])


def _fro_norms_async(
    tensors: List[Tensor],
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, Tensor]:
    sq_norms = _local_square_sums(tensors)
    if process_group is not None:
        work = dist.all_reduce(sq_norms, group=process_group, async_op=True)
        yield
        work.wait()
    return sq_norms.sqrt()
