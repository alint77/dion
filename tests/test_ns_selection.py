"""Tests for Newton-Schulz function selection via use_gns_package / use_gns_alg flags.

Verifies that NorMuon selects the correct orthogonalization function for all
combinations of (use_gns_package, use_gns_alg, use_polar_express, use_triton).
"""

import pytest
import torch

from dion import NorMuon
from dion.megabatch_base import DistributedOrthoBase
from dion.newton_schulz_triton import (
    TRITON_AVAILABLE,
    newton_schulz_triton,
    zeropower_via_newtonschulz5,
)
from dion.polar_express import polar_express, polar_express_triton

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()

torch._dynamo.config.cache_size_limit = 64


def _make_params(shapes, device=DEVICE):
    torch.manual_seed(42)
    return [torch.nn.Parameter(torch.randn(s, device=device)) for s in shapes]


def _make_normuon(**kwargs):
    params = _make_params([(64, 128)])
    return NorMuon(params, lr=0.01, **kwargs)


def _run_steps(opt, params, n_steps=3):
    for step in range(n_steps):
        torch.manual_seed(100 + step)
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    return [p.data.clone() for p in params]


# ---------------------------------------------------------------------------
# Native dion function selection (use_gns_package=False)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestNativeNSSelection:
    """Verify _newton_schulz_func is set to the correct native function."""

    def test_polar_express_no_triton(self):
        opt = _make_normuon(use_polar_express=True, use_triton=False)
        assert opt._newton_schulz_func is polar_express

    def test_no_polar_express_no_triton(self):
        opt = _make_normuon(use_polar_express=False, use_triton=False)
        assert opt._newton_schulz_func is zeropower_via_newtonschulz5

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton required")
    def test_polar_express_with_triton(self):
        opt = _make_normuon(use_polar_express=True, use_triton=True)
        assert opt._newton_schulz_func is polar_express_triton

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton required")
    def test_no_polar_express_with_triton(self):
        opt = _make_normuon(use_polar_express=False, use_triton=True)
        assert opt._newton_schulz_func is newton_schulz_triton

    def test_default_is_polar_express(self):
        """Default (no flags) should use polar_express."""
        opt = _make_normuon()
        assert opt._newton_schulz_func is polar_express


# ---------------------------------------------------------------------------
# GNS package function selection (use_gns_package=True)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestGNSSelection:
    """Verify GNS package is used when use_gns_package=True."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_gns(self):
        pytest.importorskip("gram_newton_schulz")

    def test_gns_package_without_gns_alg(self):
        opt = _make_normuon(use_gns_package=True, use_gns_alg=False)
        # Should be a lambda wrapping GramNewtonSchulz, not a native func
        assert opt._newton_schulz_func not in DistributedOrthoBase._NATIVE_NS_FUNCS.values()

    def test_gns_package_with_gns_alg(self):
        opt = _make_normuon(use_gns_package=True, use_gns_alg=True)
        assert opt._newton_schulz_func not in DistributedOrthoBase._NATIVE_NS_FUNCS.values()

    def test_gns_package_without_alg_runs(self):
        """GNS package without GNS alg should run without error."""
        params = _make_params([(64, 128)])
        opt = NorMuon(params, lr=0.01, use_gns_package=True, use_gns_alg=False)
        _run_steps(opt, params)

    def test_gns_package_with_alg_runs(self):
        """GNS package with GNS alg should run without error."""
        params = _make_params([(64, 128)])
        opt = NorMuon(params, lr=0.01, use_gns_package=True, use_gns_alg=True)
        _run_steps(opt, params)


# ---------------------------------------------------------------------------
# Custom newton_schulz_func override
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestCustomFunc:
    def test_custom_func_takes_priority(self):
        """newton_schulz_func should override all flags."""
        custom = lambda X, epsilon=None: X
        opt = _make_normuon(
            newton_schulz_func=custom,
            use_gns_package=True,
            use_polar_express=True,
        )
        assert opt._newton_schulz_func is custom

    def test_custom_func_must_be_callable(self):
        with pytest.raises(TypeError, match="callable"):
            _make_normuon(newton_schulz_func="not_a_function")


# ---------------------------------------------------------------------------
# End-to-end: native variants produce valid updates
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestNativeNSRuns:
    """Each native NS variant runs without error and updates params."""

    def test_polar_express_runs(self):
        params = _make_params([(64, 128)])
        before = params[0].data.clone()
        opt = NorMuon(params, lr=0.01, use_polar_express=True, use_triton=False)
        _run_steps(opt, params)
        assert not torch.equal(params[0].data, before)

    def test_zeropower_runs(self):
        params = _make_params([(64, 128)])
        before = params[0].data.clone()
        opt = NorMuon(params, lr=0.01, use_polar_express=False, use_triton=False)
        _run_steps(opt, params)
        assert not torch.equal(params[0].data, before)

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton required")
    def test_polar_express_triton_runs(self):
        params = _make_params([(64, 128)])
        before = params[0].data.clone()
        opt = NorMuon(params, lr=0.01, use_polar_express=True, use_triton=True)
        _run_steps(opt, params)
        assert not torch.equal(params[0].data, before)

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton required")
    def test_newton_schulz_triton_runs(self):
        params = _make_params([(64, 128)])
        before = params[0].data.clone()
        opt = NorMuon(params, lr=0.01, use_polar_express=False, use_triton=True)
        _run_steps(opt, params)
        assert not torch.equal(params[0].data, before)
