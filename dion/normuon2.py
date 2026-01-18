import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Generator, List, Optional, Tuple, Union

from .modded_normuon import apply_normuon_variance_reduction, polar_express
from .muon import muon_update_post_orthogonalize
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async


class NorMuon2(Optimizer):
    """
    NorMuon2 optimizer using Polar Express orthogonalization.
    Supports DDP and FSDP (DTensor) for matrix parameters.
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
    ):
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    "Only 1D DeviceMesh is supported. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        defaults = dict(
            lr=lr,
            momentum=mu,
            muon_beta2=muon_beta2,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            algorithm="normuon2",
            step=0,
        )
        super().__init__(params, defaults)

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param, dtype=torch.bfloat16)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param, dtype=torch.bfloat16)
            if algo == "normuon2":
                if param.ndim < 2:
                    raise ValueError(
                        "NorMuon2 only supports matrix parameters. Use a scalar optimizer for others."
                    )
                state["variance_neuron"] = torch.zeros_like(
                    param[..., :, :1], dtype=torch.bfloat16
                )
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        normuon_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1
            algo = group["algorithm"]
            if algo == "normuon2":
                normuon_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        normuon_tasks = self._create_normuon2_tasks(normuon_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(normuon_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _create_normuon2_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "normuon2",
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "NorMuon2 only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            normuon_update_args = dict(
                lr=group["lr"],
                momentum=group["momentum"],
                muon_beta2=group["muon_beta2"],
                weight_decay=group["weight_decay"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
            )

            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]
                variances_neuron = [s["variance_neuron"] for s in states]

                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                    is_batch_sharded = any(
                        p.dim not in matrix_dims for _, p in shard_placements
                    )
                    shard_placements = [
                        (i, p) for i, p in shard_placements if p.dim in matrix_dims
                    ]

                    if any(p.dim == params[0].ndim - 1 for _, p in shard_placements):
                        raise NotImplementedError(
                            "NorMuon2 does not support parameters sharded along the last dimension."
                        )

                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "NorMuon2 does not support parameters with multiple sharded dimensions."
                        )

                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            "DTensor mesh does not match optimizer mesh for NorMuon2."
                        )

                if is_batch_sharded and not is_matrix_sharded:
                    for x, g, m, v in zip(
                        params, gradients, momentums, variances_neuron
                    ):
                        yield AsyncTask(
                            normuon2_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                V=[v],
                                shard_dim=None,
                                **normuon_update_args,
                            )
                        )
                else:
                    yield AsyncTask(
                        normuon2_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            V=pad_batch(variances_neuron, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            **normuon_update_args,
                        )
                    )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == algo_name

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    cautious_wd=True,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == algo_name

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            step = torch.tensor(group["step"])
            epsilon = torch.tensor(group["epsilon"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                    cautious_wd=True,
                )
            )


def normuon2_update_batch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: float,
    momentum: float,
    muon_beta2: float,
    weight_decay: float,
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
) -> Generator[None, None, None]:
    assert len(X) == len(G)
    assert len(X) == len(M)

    G_local = to_local(G)
    M_local = to_local(M)
    U = normuon2_pre_orthogonalize(G_local, M_local, momentum)

    if shard_dim is not None:
        assert len(X) == world_size, "Batch size must equal world size"
        assert process_group is not None, "process_group required for sharded DTensor"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), "Shard dimension size must be divisible by world size"

        single_matrix_shards = [torch.empty_like(u) for u in U]
        work = dist.all_to_all(
            single_matrix_shards, U, group=process_group, async_op=True
        )
        yield
        work.wait()

        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = polar_express(single_matrix)

        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]
        U_out = [torch.empty_like(x) for x in single_matrix_shards]
        work = dist.all_to_all(
            U_out, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()
        U = U_out

    elif len(U) > 1:
        assert len(U) == world_size, "Batch size must equal world size"
        assert process_group is not None, "process_group required for all_gather"

        single_matrix = U[device_rank]
        single_matrix = polar_express(single_matrix)

        U_out = [torch.empty_like(single_matrix) for _ in range(len(U))]
        work = dist.all_gather(
            U_out, single_matrix.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()
        U = U_out

    else:
        U[0] = polar_express(U[0])

    X_local = to_local(X)
    V_local = to_local(V)
    if not X_local:
        return

    device = X_local[0].device
    lr_tensor = torch.tensor(lr, device=device, dtype=torch.float32)
    wd_tensor = torch.tensor(weight_decay, device=device, dtype=torch.float32)

    U_scaled = []
    for u, v in zip(U, V_local):
        red_dim = -1
        U_scaled.append(apply_normuon_variance_reduction(u, v, muon_beta2, red_dim))

    muon_update_post_orthogonalize(
        X_local,
        U_scaled,
        base_lr=lr_tensor,
        adjusted_lr=lr_tensor,
        weight_decay=wd_tensor,
        cautious_wd=True,
    )


def normuon2_pre_orthogonalize(
    G: List[Tensor], M: List[Tensor], momentum: float
) -> List[Tensor]:
    """
    Update momentum with lerp and return bf16 inputs for orthogonalization.
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]
    torch._foreach_lerp_(M, G, 1 - momentum)
    U = torch._foreach_lerp(G, M, momentum)
    U = [u.to(dtype=torch.bfloat16) for u in U]
    return U
