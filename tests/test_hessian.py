"""Test and benchmark the analytical Hessian against autodiff (jacfwd/jacrev).

Verifies correctness by comparing Hessian matrices, eigenvalues, spectral
statistics, and stochastic estimators, then times all approaches across
several network sizes.

Approaches compared:
    1. Full spectrum via analytical materialization + eigvalsh (DLN-specific)
    2. Full spectrum via jacfwd(jacrev(...)) + eigvalsh (general-purpose)
    3. Extremal eigenvalues via implicit analytical HVPs + ARPACK eigsh
    4. Spectral statistics: exact vs SLQ vs Hutchinson
"""

import time

import numpy as np
import torch as t
from torch import nn
from torch.func import jacfwd, jacrev, functional_call
from scipy.sparse.linalg import LinearOperator, eigsh as scipy_eigsh
from omegaconf import OmegaConf

from dln.model import DeepLinearNetwork
from dln.metrics import (
    _materialize_hessian_analytical,
    _hessian_precompute,
    _hessian_vector_product_analytical,
    _spectral_statistics,
    _slq_spectral_statistics,
    _hutchinson_spectral_statistics,
)


def create_model(
    in_dim=5, out_dim=5, num_hidden=2, hidden_dim=10, gamma=1.5, model_seed=0
):
    cfg = OmegaConf.create(
        dict(
            in_dim=in_dim,
            out_dim=out_dim,
            num_hidden=num_hidden,
            hidden_dim=hidden_dim,
            gamma=gamma,
            model_seed=model_seed,
        )
    )
    return DeepLinearNetwork(cfg)


def materialize_hessian_autodiff(model, inputs, targets):
    """Materialize the full Hessian using jacfwd(jacrev(...)).

    This is the standard general-purpose approach: reverse-mode gives the
    gradient (efficient for scalar output), then forward-mode differentiates
    that gradient w.r.t. all P parameters. Works for any differentiable model.
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def loss_fn(flat_params):
        param_dict = {}
        offset = 0
        for name, p in params.items():
            numel = p.numel()
            param_dict[name] = flat_params[offset : offset + numel].view(p.shape)
            offset += numel
        output = functional_call(model, (param_dict, buffers), (inputs,))
        return nn.functional.mse_loss(output, targets)

    flat = t.cat([p.detach().flatten() for p in params.values()])
    flat = flat.requires_grad_(True)

    H = jacfwd(jacrev(loss_fn))(flat)
    return H.detach()


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_correctness(sizes_list, n_samples=50, seed=42):
    """Compare Hessian matrices and eigenvalues from both methods."""
    t.manual_seed(seed)
    print("=" * 70)
    print("CORRECTNESS: analytical vs autodiff (full Hessian + eigenvalues)")
    print("=" * 70)

    all_passed = True
    for sizes in sizes_list:
        in_dim, out_dim = sizes[0], sizes[-1]
        hidden_dim = sizes[1]
        num_hidden = len(sizes) - 2

        model = create_model(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )
        inputs = t.randn(n_samples, in_dim)
        targets = t.randn(n_samples, out_dim)
        P = sum(p.numel() for p in model.parameters())

        with t.no_grad():
            H_analytical = _materialize_hessian_analytical(model, inputs, targets)
        eigs_analytical = t.linalg.eigvalsh(H_analytical)

        H_autodiff = materialize_hessian_autodiff(model, inputs, targets)
        eigs_autodiff = t.linalg.eigvalsh(H_autodiff)

        max_eig_err = (eigs_analytical - eigs_autodiff).abs().max().item()
        H_scale = H_autodiff.abs().max().item()
        rel_H_err = (H_analytical - H_autodiff).abs().max().item() / H_scale if H_scale > 0 else 0.0
        sym_err = (H_analytical - H_analytical.T).abs().max().item()

        passed = max_eig_err < 1e-3 and rel_H_err < 1e-4
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"\n  {sizes}  (P={P})")
        print(f"    Max eigenvalue error:  {max_eig_err:.2e}")
        print(f"    Max Hessian rel error: {rel_H_err:.2e}")
        print(f"    Symmetry error:        {sym_err:.2e}")
        print(f"    [{status}]")

    print(f"\n{'=' * 70}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}")
    return all_passed


def test_hvp_correctness(sizes_list, n_samples=50, seed=42):
    """Verify analytical HVP matches dense H @ v."""
    t.manual_seed(seed)
    print("\n" + "=" * 70)
    print("CORRECTNESS: analytical HVP vs dense H @ v")
    print("=" * 70)

    all_passed = True
    for sizes in sizes_list:
        in_dim, out_dim = sizes[0], sizes[-1]
        hidden_dim = sizes[1]
        num_hidden = len(sizes) - 2

        model = create_model(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )
        inputs = t.randn(n_samples, in_dim)
        targets = t.randn(n_samples, out_dim)
        P = sum(p.numel() for p in model.parameters())

        with t.no_grad():
            H = _materialize_hessian_analytical(model, inputs, targets)
            pre = _hessian_precompute(model, inputs, targets)

        # Test with several random vectors
        max_err = 0.0
        for _ in range(5):
            v = t.randn(P)
            hvp_dense = H @ v
            hvp_analytical = _hessian_vector_product_analytical(pre, v)
            err = (hvp_dense - hvp_analytical).abs().max().item()
            scale = hvp_dense.abs().max().item()
            max_err = max(max_err, err / scale if scale > 0 else 0.0)

        passed = max_err < 1e-4
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"\n  {sizes}  (P={P})")
        print(f"    Max relative HVP error: {max_err:.2e}")
        print(f"    [{status}]")

    print(f"\n{'=' * 70}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}")
    return all_passed


def test_arpack_correctness(sizes_list, n_samples=50, seed=42):
    """Verify ARPACK eigsh via analytical HVP matches full spectrum."""
    t.manual_seed(seed)
    print("\n" + "=" * 70)
    print("CORRECTNESS: ARPACK eigsh vs full spectrum")
    print("=" * 70)

    all_passed = True
    for sizes in sizes_list:
        in_dim, out_dim = sizes[0], sizes[-1]
        hidden_dim = sizes[1]
        num_hidden = len(sizes) - 2

        model = create_model(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )
        inputs = t.randn(n_samples, in_dim)
        targets = t.randn(n_samples, out_dim)
        P = sum(p.numel() for p in model.parameters())

        with t.no_grad():
            H = _materialize_hessian_analytical(model, inputs, targets)
            pre = _hessian_precompute(model, inputs, targets)

        full_eigs = t.linalg.eigvalsh(H).numpy()
        device, dtype = inputs.device, inputs.dtype

        eigsh_k = min(10, P // 4)

        def matvec(x):
            v = t.tensor(x, device=device, dtype=dtype)
            return _hessian_vector_product_analytical(pre, v).detach().cpu().numpy()

        op = LinearOperator((P, P), matvec=matvec, dtype="float64")
        arpack_top, _ = scipy_eigsh(op, k=eigsh_k, which="LA")
        arpack_bottom, _ = scipy_eigsh(op, k=eigsh_k, which="SA")

        arpack_top_sorted = np.sort(arpack_top)
        arpack_bottom_sorted = np.sort(arpack_bottom)
        ref_top = np.sort(full_eigs[-eigsh_k:])
        ref_bottom = np.sort(full_eigs[:eigsh_k])

        arpack_top_err = np.max(np.abs(arpack_top_sorted - ref_top))
        arpack_bottom_err = np.max(np.abs(arpack_bottom_sorted - ref_bottom))
        eig_scale = max(abs(full_eigs[0]), abs(full_eigs[-1]))

        passed = (arpack_top_err / eig_scale < 1e-3) and (arpack_bottom_err / eig_scale < 1e-3)
        all_passed = all_passed and passed

        print(f"\n  {sizes}  (P={P})")
        print(f"    ARPACK eigsh top-{eigsh_k} error:    {arpack_top_err:.2e}  (rel: {arpack_top_err/eig_scale:.2e})")
        print(f"    ARPACK eigsh bottom-{eigsh_k} error: {arpack_bottom_err:.2e}  (rel: {arpack_bottom_err/eig_scale:.2e})")
        print(f"    [{'PASS' if passed else 'FAIL'}]")

    print(f"\n{'=' * 70}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}")
    return all_passed


def test_spectral_statistics_correctness(sizes_list, n_samples=50, seed=42):
    """Compare spectral statistics: exact vs SLQ vs Hutchinson."""
    t.manual_seed(seed)
    print("\n" + "=" * 70)
    print("CORRECTNESS: spectral statistics — exact vs SLQ vs Hutchinson")
    print("=" * 70)

    all_passed = True
    for sizes in sizes_list:
        in_dim, out_dim = sizes[0], sizes[-1]
        hidden_dim = sizes[1]
        num_hidden = len(sizes) - 2

        model = create_model(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )
        inputs = t.randn(n_samples, in_dim)
        targets = t.randn(n_samples, out_dim)
        P = sum(p.numel() for p in model.parameters())

        with t.no_grad():
            H = _materialize_hessian_analytical(model, inputs, targets)
            pre = _hessian_precompute(model, inputs, targets)

        eigenvalues = t.linalg.eigvalsh(H)
        exact = _spectral_statistics(eigenvalues)

        slq = _slq_spectral_statistics(
            pre, P, inputs.device, inputs.dtype,
            n_probes=50, lanczos_steps=min(100, P), seed=0,
        )
        hutch = _hutchinson_spectral_statistics(
            pre, P, inputs.device, inputs.dtype,
            n_probes=100, seed=0,
        )

        # Two scaling factors for inherent limitations of stochastic methods:
        # 1. Small P → fewer effective samples, higher variance.
        # 2. trace ≈ 0 → cancellation between pos/neg eigenvalues makes
        #    trace, stable_rank, participation_ratio inherently noisy.
        #    Variance of trace estimator ∝ ‖H‖²_F, so relative error ∝ ‖H‖_F/|trace|.
        base_tol = max(1.0, 200 / P)  # looser for small P
        trace_to_frob = abs(exact["trace"]) / exact["frobenius_norm"] if exact["frobenius_norm"] > 0 else 1.0
        cancel_tol = max(1.0, 2.0 / max(trace_to_frob, 0.01))  # looser when trace cancels

        slq_tols = {
            "trace": 0.10 * base_tol * cancel_tol,
            "frobenius_norm": 0.05 * base_tol,
            "lambda_max": 0.02,      # Lanczos extremals are very accurate
            "lambda_min": 0.05,
            "stable_rank": 0.10 * base_tol * cancel_tol,
            "participation_ratio": 0.15 * base_tol * cancel_tol,
            "entropic_rank": 0.25 * base_tol,
        }
        hutch_tols = {
            "trace": 0.15 * base_tol * cancel_tol,
            "frobenius_norm": 0.10 * base_tol,
            "lambda_max": 0.15,      # power iteration — may need more steps for large P
            "stable_rank": 0.20 * base_tol * cancel_tol,
            "participation_ratio": 0.25 * base_tol * cancel_tol,
        }

        print(f"\n  {sizes}  (P={P})")
        print(f"    {'Statistic':<25} {'Exact':>12} {'SLQ':>12} {'SLQ err':>10} {'Hutch':>12} {'Hutch err':>10}")
        print(f"    {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

        size_passed = True
        for key in ["trace", "frobenius_norm", "lambda_max", "lambda_min",
                     "stable_rank", "participation_ratio", "entropic_rank"]:
            exact_val = exact[key]
            slq_val = slq.get(key)
            hutch_val = hutch.get(key)

            scale = abs(exact_val) if abs(exact_val) > 1e-10 else 1.0
            slq_err = abs(slq_val - exact_val) / scale if slq_val is not None else None
            hutch_err = abs(hutch_val - exact_val) / scale if hutch_val is not None else None

            slq_str = f"{slq_val:>12.4f}" if slq_val is not None else f"{'n/a':>12}"
            hutch_str = f"{hutch_val:>12.4f}" if hutch_val is not None else f"{'n/a':>12}"
            slq_err_str = f"{slq_err:>9.1%}" if slq_err is not None else f"{'n/a':>10}"
            hutch_err_str = f"{hutch_err:>9.1%}" if hutch_err is not None else f"{'n/a':>10}"

            print(f"    {key:<25} {exact_val:>12.4f} {slq_str} {slq_err_str} {hutch_str} {hutch_err_str}")

            # Check tolerances
            if key in slq_tols and slq_err is not None and slq_err > slq_tols[key]:
                size_passed = False
            if key in hutch_tols and hutch_err is not None and hutch_err > hutch_tols[key]:
                size_passed = False

        # Also show stats only available from exact
        for key in ["neg_fraction", "gini"]:
            exact_val = exact[key]
            slq_val = slq.get(key)
            print(f"    {key:<25} {exact_val:>12.4f} {slq_val:>12.4f} {'':>10} {'n/a':>12} {'n/a':>10}" if slq_val is not None else
                  f"    {key:<25} {exact_val:>12.4f} {'n/a':>12} {'':>10} {'n/a':>12} {'n/a':>10}")

        all_passed = all_passed and size_passed
        print(f"    [{'PASS' if size_passed else 'FAIL'}]")

    print(f"\n{'=' * 70}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}")
    return all_passed


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------


def _sync(device):
    if device.type == "cuda":
        t.cuda.synchronize()


def _time_fn(fn, device, n_warmup, n_runs):
    """Time a function, returning (best_time, all_times)."""
    for _ in range(n_warmup):
        fn()
        _sync(device)

    times = []
    for _ in range(n_runs):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return min(times), times


def benchmark(sizes_list, n_samples=50, n_warmup=2, n_runs=5, seed=42):
    """Time all approaches across network sizes.

    Approaches:
        1. Analytical materialization + full eigvalsh
        2. Autodiff materialization + full eigvalsh (general-purpose baseline)
        3. Implicit analytical HVPs + ARPACK eigsh (top/bottom k)
        4. SLQ spectral statistics (n_probes × lanczos_steps HVPs)
        5. Hutchinson spectral statistics (n_probes HVPs)
    """
    t.manual_seed(seed)
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for sizes in sizes_list:
        in_dim, out_dim = sizes[0], sizes[-1]
        hidden_dim = sizes[1]
        num_hidden = len(sizes) - 2

        model = create_model(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        ).to(device)
        inputs = t.randn(n_samples, in_dim, device=device)
        targets = t.randn(n_samples, out_dim, device=device)
        P = sum(p.numel() for p in model.parameters())
        k_extremal = min(20, P // 2 - 1)

        print(f"\n  {sizes}  (P={P}, N={n_samples})")
        print(f"  {'-' * 60}")

        # --- 1. Analytical materialization ---
        def analytical_materialize():
            with t.no_grad():
                return _materialize_hessian_analytical(model, inputs, targets)

        t_mat_analytical, _ = _time_fn(analytical_materialize, device, n_warmup, n_runs)

        # --- Eigendecomposition (full, on materialized H) ---
        H_ref = analytical_materialize()

        def full_eigvalsh():
            return t.linalg.eigvalsh(H_ref)

        t_eig_full, _ = _time_fn(full_eigvalsh, device, n_warmup, n_runs)

        # --- 2. ARPACK eigsh with analytical HVPs ---
        with t.no_grad():
            pre = _hessian_precompute(model, inputs, targets)

        def make_op():
            def matvec(x):
                v = t.tensor(x, device=device, dtype=inputs.dtype)
                return _hessian_vector_product_analytical(pre, v).detach().cpu().numpy()
            return LinearOperator((P, P), matvec=matvec, dtype="float64")

        def eigsh_implicit():
            op = make_op()
            top, _ = scipy_eigsh(op, k=k_extremal, which="LA")
            bottom, _ = scipy_eigsh(op, k=k_extremal, which="SA")
            return top, bottom

        t_eigsh, _ = _time_fn(eigsh_implicit, device, n_warmup, n_runs)

        # Also time k=100 (or largest feasible)
        k_100 = min(100, P // 2 - 1)

        def eigsh_implicit_100():
            op = make_op()
            top, _ = scipy_eigsh(op, k=k_100, which="LA")
            bottom, _ = scipy_eigsh(op, k=k_100, which="SA")
            return top, bottom

        t_eigsh_100, _ = _time_fn(eigsh_implicit_100, device, n_warmup, n_runs)

        # --- 3. SLQ spectral statistics ---
        def slq_30_64():
            with t.no_grad():
                return _slq_spectral_statistics(
                    pre, P, device, inputs.dtype, n_probes=30, lanczos_steps=64,
                )

        t_slq_30, _ = _time_fn(slq_30_64, device, n_warmup, n_runs)

        def slq_10_32():
            with t.no_grad():
                return _slq_spectral_statistics(
                    pre, P, device, inputs.dtype, n_probes=10, lanczos_steps=32,
                )

        t_slq_10, _ = _time_fn(slq_10_32, device, n_warmup, n_runs)

        # --- 4. Hutchinson ---
        def hutch_50():
            with t.no_grad():
                return _hutchinson_spectral_statistics(
                    pre, P, device, inputs.dtype, n_probes=50,
                )

        t_hutch, _ = _time_fn(hutch_50, device, n_warmup, n_runs)

        del H_ref

        # --- 5. Autodiff materialization (skip for large P) ---
        if P > 15000:
            t_mat_autodiff = None
        else:
            def autodiff_materialize():
                return materialize_hessian_autodiff(model, inputs, targets)
            t_mat_autodiff, _ = _time_fn(autodiff_materialize, device, n_warmup, n_runs)

        # --- Report ---
        print(f"    {'Method':<50} {'Time':>10} {'HVPs':>8}")
        print(f"    {'-'*50} {'-'*10} {'-'*8}")

        if t_mat_autodiff is not None:
            print(f"    Autodiff materialize                            {t_mat_autodiff:>9.4f}s {'':>8}")
            print(f"    Analytical materialize                          {t_mat_analytical:>9.4f}s {'':>8}  ({t_mat_autodiff/t_mat_analytical:.0f}x faster)")
        else:
            print(f"    Autodiff materialize                             skipped")
            print(f"    Analytical materialize                          {t_mat_analytical:>9.4f}s")

        print(f"    Full eigvalsh                                   {t_eig_full:>9.4f}s {'':>8}")

        t_full_analytical = t_mat_analytical + t_eig_full
        print(f"    ---")
        print(f"    Full spectrum (analytical + eigvalsh)            {t_full_analytical:>9.4f}s {'':>8}")
        if t_mat_autodiff is not None:
            t_full_autodiff = t_mat_autodiff + t_eig_full
            print(f"    Full spectrum (autodiff + eigvalsh)              {t_full_autodiff:>9.4f}s {'':>8}")

        print(f"    ARPACK eigsh top+bottom {k_extremal} (implicit HVP)       {t_eigsh:>9.4f}s {'':>8}")
        print(f"    ARPACK eigsh top+bottom {k_100} (implicit HVP)      {t_eigsh_100:>9.4f}s {'':>8}")
        print(f"    SLQ (30 probes × 64 steps)                      {t_slq_30:>9.4f}s {30*64:>8}")
        print(f"    SLQ (10 probes × 32 steps)                      {t_slq_10:>9.4f}s {10*32:>8}")
        print(f"    Hutchinson (50 probes)                          {t_hutch:>9.4f}s {50:>8}")


if __name__ == "__main__":
    # --- Correctness tests ---
    correctness_sizes = [
        (3, 4, 3),           # 2 layers, P=24
        (5, 8, 5),           # 2 layers, P=80
        (5, 10, 10, 5),      # 3 layers, P=200
        (5, 10, 10, 10, 5),  # 4 layers, P=350
        (5, 20, 20, 20, 5),  # 4 layers, P=1300
    ]
    test_correctness(correctness_sizes)
    test_hvp_correctness(correctness_sizes)
    test_arpack_correctness(correctness_sizes)
    test_spectral_statistics_correctness(correctness_sizes)

    # --- Performance benchmarks ---
    benchmark_sizes = [
        (5, 10, 10, 10, 5),    # P=350
        (5, 20, 20, 20, 5),    # P=1300
        (5, 50, 50, 50, 5),    # P=7750
        (5, 100, 100, 100, 5), # P=21000
    ]
    benchmark(benchmark_sizes, n_samples=500)
