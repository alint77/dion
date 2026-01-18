from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from .modded_normuon import polar_express, cautious_wd_and_update_inplace


def _adjust_lr_factor(param_shape, adjust_lr):
    if adjust_lr is None:
        return 1.0
    fan_out, fan_in = param_shape[-2:]
    if adjust_lr == "spectral_norm":
        return math.sqrt(fan_out / fan_in)
    if adjust_lr == "rms_norm":
        return 0.2 * math.sqrt(max(fan_out, fan_in))
    raise ValueError(
        f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
    )


def _select_dion2_submatrix(buffer: Tensor, fraction: float, ef_decay: float, select_dim: int):
    num_select = buffer.size(select_dim)
    k = max(1, int(math.ceil(fraction * num_select)))
    norm_dim = -1 if select_dim == -2 else -2

    slice_norms = buffer.abs().sum(dim=norm_dim)
    _, indices = torch.topk(slice_norms, k, dim=-1, sorted=False)

    if select_dim == -2:
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, buffer.size(-1))
        selected = torch.gather(buffer, dim=-2, index=indices_expanded)
    else:
        indices_expanded = indices.unsqueeze(-2).expand(-1, buffer.size(-2), -1)
        selected = torch.gather(buffer, dim=-1, index=indices_expanded)

    for i in range(buffer.size(0)):
        idx = indices[i]
        if select_dim == -2:
            selected_slice = buffer[i].index_select(dim=0, index=idx)
            buffer[i].index_copy_(dim=0, index=idx, source=selected_slice.mul(ef_decay))
        else:
            selected_slice = buffer[i].index_select(dim=1, index=idx)
            buffer[i].index_copy_(dim=1, index=idx, source=selected_slice.mul(ef_decay))

    return selected, indices


def _scatter_selected(out: Tensor, selected: Tensor, indices: Tensor, select_dim: int):
    for i in range(selected.size(0)):
        idx = indices[i]
        if select_dim == -2:
            out[i].index_copy_(dim=0, index=idx, source=selected[i])
        else:
            out[i].index_copy_(dim=1, index=idx, source=selected[i])


class ModdedDion2(torch.optim.Optimizer):
    """
    Modded Dion2 optimizer adapted from modded-nanogpt. DDP-only implementation.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """

    def __init__(
        self,
        params,
        distributed_mesh: Optional[ProcessGroup] = None,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        fraction: float = 0.25,
        ef_decay: float = 0.95,
        adjust_lr: Optional[str] = None,
        custom_sizing: bool = True,
    ):
        if distributed_mesh is not None and not isinstance(distributed_mesh, ProcessGroup):
            raise TypeError("distributed_mesh must be a ProcessGroup or None for DDP")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if ef_decay < 0.0:
            raise ValueError(f"Invalid ef_decay: {ef_decay}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            fraction=fraction,
            ef_decay=ef_decay,
            adjust_lr=adjust_lr,
        )
        self._process_group = distributed_mesh
        if dist.is_initialized():
            if self._process_group is not None:
                self.world_size = dist.get_world_size(self._process_group)
                self.rank = dist.get_rank(self._process_group)
            else:
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self._use_dist = dist.is_initialized() and self.world_size > 1

        param_groups = self._normalize_param_groups(params, defaults, custom_sizing)
        super().__init__(param_groups, defaults)

    def _normalize_param_groups(self, params, defaults, custom_sizing):
        params_list = list(params)
        if params_list and isinstance(params_list[0], dict):
            param_groups = []
            for group in params_list:
                group_params = group.get("params")
                if group_params is None:
                    raise ValueError("param group missing 'params' key")
                if isinstance(group_params, torch.Tensor):
                    group_params = [group_params]
                group_params = list(group_params)

                groups_by_label = defaultdict(list)
                has_label = False
                for param in group_params:
                    label = getattr(param, "label", None)
                    if label is not None:
                        has_label = True
                    groups_by_label[label].append(param)

                for params in (groups_by_label.values() if has_label else [group_params]):
                    group = dict(group)
                    group["params"] = list(params)
                    group.pop("chunk_size", None)
                    if "mu" in group and "momentum" not in group:
                        group["momentum"] = group["mu"]
                    group.setdefault("lr", defaults["lr"])
                    group.setdefault("weight_decay", defaults["weight_decay"])
                    group.setdefault("momentum", defaults["momentum"])
                    group.setdefault("fraction", defaults["fraction"])
                    group.setdefault("ef_decay", defaults["ef_decay"])
                    group.setdefault("adjust_lr", defaults["adjust_lr"])
                    if "chunk_size" not in group:
                        group["chunk_size"] = (
                            len(group["params"]) + self.world_size - 1
                        ) // self.world_size
                    param_groups.append(group)
            return param_groups

        if custom_sizing and self.world_size == 8:
            return self.generate_custom_param_groups(params_list)
        return self.generate_standard_param_groups(params_list)

    def reset(self):
        for group in self.param_groups:
            if "momentum_buffer" in group:
                group["momentum_buffer"].zero_()
                group["mantissa"].zero_()
                group["ef_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        """
        Use this method if running on less than 8 GPU or experimenting with additional attn or mlp modules.
        Creates one param group per module.
        """
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for group_params in groups.values():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))

        return param_groups

    def generate_custom_param_groups(self, params):
        """
        Implementation requires that a single GPU does not receive both attn
        and mlp params when a param group is split across GPUs.
        """
        params_list = list(params)
        module_group_order = ["attn", "mlp"]
        group_sizes = [16, 16]  # 10 attn + 6 mlp, then 16 mlp
        params_list.sort(key=lambda x: module_group_order.index(x.label))

        idx = 0
        assert len(params_list) == sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            chunk_size = (size + self.world_size - 1) // self.world_size
            group_params = params_list[idx : idx + size]
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
            idx += size

        return param_groups

    def _reduce_scatter(self, out, inp):
        if self._process_group is None:
            return dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.AVG, async_op=True)
        return dist.reduce_scatter_tensor(
            out, inp, op=dist.ReduceOp.AVG, async_op=True, group=self._process_group
        )

    def _all_gather(self, out, inp):
        if self._process_group is None:
            return dist.all_gather_into_tensor(out, inp, async_op=True)
        return dist.all_gather_into_tensor(out, inp, async_op=True, group=self._process_group)

    def step(self):
        self.step_p1()
        self.step_p2()
        self.step_p3()

    @torch.no_grad()
    def step_p1(self):
        self.group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue

            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            stacked_grads = torch.empty(
                (padded_num_params, *params[0].shape),
                dtype=params[0].dtype,
                device=params[0].device,
            )
            for i, p in enumerate(params):
                stacked_grads[i].copy_(p.grad, non_blocking=True)
            if len(params) < padded_num_params:
                stacked_grads[len(params) :].zero_()

            grad_chunk = torch.empty_like(stacked_grads[:chunk_size])

            if self._use_dist:
                reduce_future = self._reduce_scatter(grad_chunk, stacked_grads).get_future()
            else:
                reduce_future = None
                grad_chunk.copy_(stacked_grads[:chunk_size])

            self.group_infos.append(
                dict(grad_chunk=grad_chunk, reduce_future=reduce_future)
            )

    @torch.no_grad()
    def step_p2(self):
        group_infos = self.group_infos

        self.all_gather_infos = []
        for group, info in zip(self.param_groups, group_infos):
            if info["reduce_future"] is not None:
                info["reduce_future"].wait()

            params = group["params"]
            grad_chunk = info["grad_chunk"].float()
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            start_idx = self.rank * chunk_size
            module_idx = start_idx if start_idx < len(params) else 0

            num_params = min(chunk_size, max(0, len(params) - start_idx))

            if "momentum_buffer" not in group:
                group["momentum_buffer"] = torch.zeros_like(
                    grad_chunk[:num_params], dtype=torch.float32
                )

            momentum_buffer = group["momentum_buffer"]
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(
                momentum_buffer, group["momentum"]
            )

            grad_shape = updated_grads.shape
            if params[module_idx].label == "attn":
                for p in params[module_idx : module_idx + num_params]:
                    assert p.label == "attn"
                updated_grads = updated_grads.view(
                    4 * grad_shape[0], grad_shape[1] // 4, grad_shape[2]
                )

            ref_param = params[module_idx]
            param_shape = ref_param.shape

            if "param_lr_cpu" not in group:
                lr_mults = []
                wd_mults = []
                for p in params:
                    shape = p.shape
                    if len(shape) >= 2:
                        shape_mult = max(1.0, shape[-2] / shape[-1]) ** 0.5
                    else:
                        shape_mult = 1.0
                    adjust_factor = _adjust_lr_factor(shape, group["adjust_lr"])
                    lr_mults.append(shape_mult * adjust_factor * getattr(p, "lr_mul", 1.0))
                    wd_mults.append(getattr(p, "wd_mul", 1.0))
                group["param_lr_cpu"] = torch.tensor(
                    lr_mults, dtype=torch.float32, device="cpu"
                )
                group["param_wd_cpu"] = torch.tensor(
                    wd_mults, dtype=torch.float32, device="cpu"
                )

            eff_lr_all = group["param_lr_cpu"] * group["lr"]
            eff_wd_all = group["param_wd_cpu"] * group["weight_decay"] * group["lr"]

            eff_lr_cpu = eff_lr_all[module_idx : module_idx + num_params]
            eff_wd_cpu = eff_wd_all[module_idx : module_idx + num_params]

            if num_params == 0:
                v_chunk = updated_grads
            else:
                if "ef_buffer" not in group or group["ef_buffer"].shape != updated_grads.shape:
                    group["ef_buffer"] = torch.zeros_like(
                        updated_grads, dtype=torch.float32
                    )
                ef_buffer = group["ef_buffer"]
                ef_buffer.add_(updated_grads)

                select_dim = -2 if updated_grads.size(-2) <= updated_grads.size(-1) else -1
                selected, indices = _select_dion2_submatrix(
                    ef_buffer, group["fraction"], group["ef_decay"], select_dim
                )

                v_selected = polar_express(
                    selected, split_baddbmm=(ref_param.label == "mlp")
                )

                v_view = torch.zeros_like(ef_buffer, dtype=v_selected.dtype)
                _scatter_selected(v_view, v_selected, indices, select_dim)
                v_chunk = v_view.view(grad_shape)

            updated_params = torch.empty_like(grad_chunk, dtype=torch.bfloat16)
            if num_params > 0:
                param_chunk = torch.stack(params[module_idx : module_idx + num_params])

                if "mantissa" not in group or group["mantissa"].shape != param_chunk.shape:
                    group["mantissa"] = torch.zeros_like(param_chunk, dtype=torch.uint16)
                mantissa = group["mantissa"]

                for local_idx in range(num_params):
                    cautious_wd_and_update_inplace(
                        param_chunk[local_idx].view(torch.uint16),
                        mantissa[local_idx],
                        v_chunk[local_idx],
                        eff_wd_cpu[local_idx],
                        eff_lr_cpu[local_idx],
                    )
            else:
                param_chunk = torch.zeros_like(v_chunk)

            updated_params[:num_params].copy_(param_chunk)
            if num_params < chunk_size:
                updated_params[num_params:].zero_()

            stacked_params = torch.empty(
                (padded_num_params, *param_shape),
                dtype=updated_params.dtype,
                device=updated_params.device,
            )

            if self._use_dist:
                gather_future = self._all_gather(stacked_params, updated_params).get_future()
            else:
                gather_future = None
                stacked_params[:chunk_size].copy_(updated_params)

            self.all_gather_infos.append(
                {
                    "gather_future": gather_future,
                    "stacked_params": stacked_params,
                    "orig_params": params,
                }
            )

        for info in self.all_gather_infos[:-1]:
            if info["gather_future"] is not None:
                info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]

            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)

    @torch.no_grad()
    def step_p3(self):
        if not self.all_gather_infos:
            return
        info = self.all_gather_infos[-1]
        if info["gather_future"] is not None:
            info["gather_future"].wait()
        stacked_params = info["stacked_params"]
        orig_params = info["orig_params"]

        unstacked_params = torch.unbind(stacked_params)
        for i, p in enumerate(orig_params):
            p.copy_(unstacked_params[i], non_blocking=True)
