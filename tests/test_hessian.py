"""Test and benchmark the analytical Hessian against autodiff (jacfwd/jacrev).

Verifies correctness by comparing Hessian matrices and eigenvalues, then
times both approaches across several network sizes. Materialization and
eigendecomposition are timed separately so the shared O(P^3) eigvalsh cost
doesn't mask the materialization speedup.
"""

import time

import torch as t
from torch import nn
from torch.func import jacfwd, jacrev, functional_call
from omegaconf import OmegaConf

from dln.model import DeepLinearNetwork
from dln.metrics import _materialize_hessian_analytical


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


def test_correctness(sizes_list, n_samples=50, seed=42):
    """Compare Hessian matrices and eigenvalues from both methods."""
    t.manual_seed(seed)
    print("=" * 70)
    print("CORRECTNESS TEST")
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

        # Analytical (under no_grad, matching how the metric runs)
        with t.no_grad():
            H_analytical = _materialize_hessian_analytical(model, inputs, targets)
        eigs_analytical = t.linalg.eigvalsh(H_analytical)

        # Autodiff
        H_autodiff = materialize_hessian_autodiff(model, inputs, targets)
        eigs_autodiff = t.linalg.eigvalsh(H_autodiff)

        # Compare
        max_eig_err = (eigs_analytical - eigs_autodiff).abs().max().item()
        max_H_err = (H_analytical - H_autodiff).abs().max().item()
        H_scale = H_autodiff.abs().max().item()
        rel_H_err = max_H_err / H_scale if H_scale > 0 else 0.0
        sym_err = (H_analytical - H_analytical.T).abs().max().item()

        passed = max_eig_err < 1e-3 and rel_H_err < 1e-4
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"\n  {sizes}  (P={P})")
        print(f"    Max eigenvalue error:  {max_eig_err:.2e}")
        print(f"    Max Hessian abs error: {max_H_err:.2e}")
        print(f"    Max Hessian rel error: {rel_H_err:.2e}")
        print(f"    Symmetry error (analytical): {sym_err:.2e}")
        print(f"    [{status}]")

    print(f"\n{'=' * 70}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 70}")
    return all_passed


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
    """Time both methods across network sizes.

    Reports materialization time and eigendecomposition time separately,
    so the shared eigvalsh cost doesn't mask the materialization speedup.
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

        print(f"\n  {sizes}  (P={P}, N={n_samples})")
        print(f"  {'-' * 55}")

        # -- Analytical: materialization --
        def analytical_materialize():
            with t.no_grad():
                return _materialize_hessian_analytical(model, inputs, targets)

        t_mat_analytical, _ = _time_fn(analytical_materialize, device, n_warmup, n_runs)

        # -- Eigendecomposition (same for both, time once) --
        H_for_eig = analytical_materialize()

        def eigdecomp():
            return t.linalg.eigvalsh(H_for_eig)

        t_eig, _ = _time_fn(eigdecomp, device, n_warmup, n_runs)
        del H_for_eig

        # -- Autodiff: materialization (skip for very large P) --
        if P > 30000:
            print(f"    Analytical materialize:  {t_mat_analytical:.4f}s")
            print(f"    Autodiff materialize:    skipped (P={P} too large)")
            print(f"    Eigendecomposition:      {t_eig:.4f}s")
            print(f"    Analytical total:        {t_mat_analytical + t_eig:.4f}s")
            continue

        def autodiff_materialize():
            return materialize_hessian_autodiff(model, inputs, targets)

        t_mat_autodiff, _ = _time_fn(autodiff_materialize, device, n_warmup, n_runs)

        mat_speedup = t_mat_autodiff / t_mat_analytical if t_mat_analytical > 0 else float("inf")
        total_analytical = t_mat_analytical + t_eig
        total_autodiff = t_mat_autodiff + t_eig

        print(f"    Analytical materialize:  {t_mat_analytical:.4f}s")
        print(f"    Autodiff materialize:    {t_mat_autodiff:.4f}s")
        print(f"    Materialize speedup:     {mat_speedup:.1f}x")
        print(f"    Eigendecomposition:      {t_eig:.4f}s")
        print(f"    Analytical total:        {total_analytical:.4f}s")
        print(f"    Autodiff total:          {total_autodiff:.4f}s")
        print(f"    End-to-end speedup:      {total_autodiff / total_analytical:.1f}x")


if __name__ == "__main__":
    # Correctness: test on small networks where autodiff is feasible
    correctness_sizes = [
        (3, 4, 3),           # 2 layers, P=24
        (5, 8, 5),           # 2 layers, P=80
        (5, 10, 10, 5),      # 3 layers, P=200
        (5, 10, 10, 10, 5),  # 4 layers, P=350
        (5, 20, 20, 20, 5),  # 4 layers, P=1300
    ]
    test_correctness(correctness_sizes)

    # Performance: benchmark across sizes including the target (5,100,100,100,5)
    benchmark_sizes = [
        (5, 10, 10, 10, 5),    # P=350
        (5, 20, 20, 20, 5),    # P=1300
        (5, 50, 50, 50, 5),    # P=7750
        (5, 100, 100, 100, 5), # P=21000
    ]
    benchmark(benchmark_sizes, n_samples=500)
