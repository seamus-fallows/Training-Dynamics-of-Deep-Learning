"""Test and benchmark the analytical Hessian against autodiff (jacfwd/jacrev).

Verifies correctness by comparing Hessian matrices, eigenvalues, and partial
eigenvalue methods, then times all approaches across several network sizes.

Approaches compared:
    1. Full spectrum via analytical materialization + eigvalsh (DLN-specific)
    2. Full spectrum via jacfwd(jacrev(...)) + eigvalsh (general-purpose)
    3. Partial spectrum via analytical materialization + scipy partial eigh
    4. Extremal eigenvalues via implicit analytical HVPs + ARPACK eigsh
"""

import time

import numpy as np
import torch as t
from torch import nn
from torch.func import jacfwd, jacrev, functional_call
from scipy.linalg import eigh as scipy_eigh
from scipy.sparse.linalg import LinearOperator, eigsh as scipy_eigsh
from omegaconf import OmegaConf

from dln.model import DeepLinearNetwork
from dln.metrics import (
    _materialize_hessian_analytical,
    _hessian_precompute,
    _hessian_vector_product_analytical,
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


def test_partial_spectrum_correctness(sizes_list, n_samples=50, seed=42):
    """Verify partial eigenvalue methods match full spectrum."""
    t.manual_seed(seed)
    print("\n" + "=" * 70)
    print("CORRECTNESS: partial eigh and eigsh vs full spectrum")
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

        # Full spectrum as reference
        full_eigs = t.linalg.eigvalsh(H).numpy()
        H_np = H.cpu().numpy()

        # --- Scipy partial eigh (top 20%) ---
        fraction = 0.2
        k = max(1, int(P * fraction))
        top_eigs = scipy_eigh(H_np, eigvals_only=True, subset_by_index=[P - k, P - 1])
        bottom_eigs = scipy_eigh(H_np, eigvals_only=True, subset_by_index=[0, k - 1])

        top_err = np.max(np.abs(top_eigs - full_eigs[-k:]))
        bottom_err = np.max(np.abs(bottom_eigs - full_eigs[:k]))

        # --- ARPACK eigsh with analytical HVP (top/bottom 10 eigenvalues) ---
        eigsh_k = min(10, P // 2 - 1)
        device, dtype = inputs.device, inputs.dtype

        def matvec(x):
            v = t.tensor(x, device=device, dtype=dtype)
            return _hessian_vector_product_analytical(pre, v).cpu().numpy()

        op = LinearOperator((P, P), matvec=matvec, dtype="float64")
        arpack_top, _ = scipy_eigsh(op, k=eigsh_k, which="LM")
        arpack_bottom, _ = scipy_eigsh(op, k=eigsh_k, which="SA")

        arpack_top_sorted = np.sort(arpack_top)
        arpack_bottom_sorted = np.sort(arpack_bottom)
        ref_top = np.sort(full_eigs[-eigsh_k:])
        ref_bottom = np.sort(full_eigs[:eigsh_k])

        arpack_top_err = np.max(np.abs(arpack_top_sorted - ref_top))
        arpack_bottom_err = np.max(np.abs(arpack_bottom_sorted - ref_bottom))

        eig_scale = max(abs(full_eigs[0]), abs(full_eigs[-1]))
        partial_passed = top_err < 1e-4 and bottom_err < 1e-4
        arpack_passed = (arpack_top_err / eig_scale < 1e-3) and (arpack_bottom_err / eig_scale < 1e-3)
        passed = partial_passed and arpack_passed
        all_passed = all_passed and passed

        print(f"\n  {sizes}  (P={P})")
        print(f"    Partial eigh top {fraction:.0%} error:     {top_err:.2e}")
        print(f"    Partial eigh bottom {fraction:.0%} error:  {bottom_err:.2e}")
        print(f"    ARPACK eigsh top-{eigsh_k} error:       {arpack_top_err:.2e}  (rel: {arpack_top_err/eig_scale:.2e})")
        print(f"    ARPACK eigsh bottom-{eigsh_k} error:    {arpack_bottom_err:.2e}  (rel: {arpack_bottom_err/eig_scale:.2e})")
        print(f"    [{'PASS' if passed else 'FAIL'}]")

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
        3. Analytical materialization + scipy partial eigh (top/bottom 20%)
        4. Implicit analytical HVPs + ARPACK eigsh (top/bottom k)
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
        fraction = 0.2
        k_partial = max(1, int(P * fraction))
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

        # --- 3. Scipy partial eigh (top 20% + bottom 20%) ---
        H_np = H_ref.cpu().numpy()

        def partial_eigh():
            top = scipy_eigh(H_np, eigvals_only=True, subset_by_index=[P - k_partial, P - 1])
            bottom = scipy_eigh(H_np, eigvals_only=True, subset_by_index=[0, k_partial - 1])
            return top, bottom

        t_partial, _ = _time_fn(partial_eigh, device, n_warmup, n_runs)

        # --- 4. ARPACK eigsh with analytical HVPs ---
        with t.no_grad():
            pre = _hessian_precompute(model, inputs, targets)

        def make_op():
            def matvec(x):
                v = t.tensor(x, device=device, dtype=inputs.dtype)
                return _hessian_vector_product_analytical(pre, v).cpu().numpy()
            return LinearOperator((P, P), matvec=matvec, dtype="float64")

        def eigsh_implicit_small():
            op = make_op()
            top, _ = scipy_eigsh(op, k=k_extremal, which="LM")
            bottom, _ = scipy_eigsh(op, k=k_extremal, which="SA")
            return top, bottom

        t_eigsh_small, _ = _time_fn(eigsh_implicit_small, device, n_warmup, n_runs)

        # Also time k=100 (or largest feasible) for the user's use case
        k_100 = min(100, P // 2 - 1)

        def eigsh_implicit_100():
            op = make_op()
            top, _ = scipy_eigsh(op, k=k_100, which="LM")
            bottom, _ = scipy_eigsh(op, k=k_100, which="SA")
            return top, bottom

        t_eigsh_100, _ = _time_fn(eigsh_implicit_100, device, n_warmup, n_runs)

        del H_ref, H_np

        # --- 2. Autodiff materialization (skip for large P) ---
        if P > 30000:
            t_mat_autodiff = None
        else:
            def autodiff_materialize():
                return materialize_hessian_autodiff(model, inputs, targets)
            t_mat_autodiff, _ = _time_fn(autodiff_materialize, device, n_warmup, n_runs)

        # --- Report ---
        print(f"    {'Method':<45} {'Time':>10}")
        print(f"    {'-'*45} {'-'*10}")

        if t_mat_autodiff is not None:
            print(f"    Autodiff materialize                       {t_mat_autodiff:>9.4f}s")
            print(f"    Analytical materialize                     {t_mat_analytical:>9.4f}s  ({t_mat_autodiff/t_mat_analytical:.0f}x faster)")
        else:
            print(f"    Autodiff materialize                        skipped")
            print(f"    Analytical materialize                     {t_mat_analytical:>9.4f}s")

        print(f"    Full eigvalsh                              {t_eig_full:>9.4f}s")

        t_full_analytical = t_mat_analytical + t_eig_full
        print(f"    ---")
        print(f"    Full spectrum (analytical + eigvalsh)       {t_full_analytical:>9.4f}s")
        if t_mat_autodiff is not None:
            t_full_autodiff = t_mat_autodiff + t_eig_full
            print(f"    Full spectrum (autodiff + eigvalsh)         {t_full_autodiff:>9.4f}s")

        print(f"    Partial eigh top+bottom {fraction:.0%} (on H)        {t_mat_analytical + t_partial:>9.4f}s  (mat + partial)")
        print(f"    ARPACK eigsh top+bottom {k_extremal} (implicit HVP)  {t_eigsh_small:>9.4f}s  (no materialization)")
        print(f"    ARPACK eigsh top+bottom {k_100} (implicit HVP) {t_eigsh_100:>9.4f}s  (no materialization)")


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
    test_partial_spectrum_correctness(correctness_sizes)

    # --- Performance benchmarks ---
    benchmark_sizes = [
        (5, 10, 10, 10, 5),    # P=350
        (5, 20, 20, 20, 5),    # P=1300
        (5, 50, 50, 50, 5),    # P=7750
        (5, 100, 100, 100, 5), # P=21000
    ]
    benchmark(benchmark_sizes, n_samples=500)
