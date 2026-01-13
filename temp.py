# %%
from runner import load_run
from plotting import plot
from pathlib import Path

BASE_PATH = Path("outputs/gph")
METRIC_LABELS = {
    "grad_norm_squared": "||∇L||²",
    "trace_gradient_covariance": "Tr(Σ)",
    "trace_hessian_covariance": "Tr(HΣ)",
}


def get_path(width, gamma, batch, seed, noise):
    b_str = "None" if batch is None else str(batch)
    return BASE_PATH / f"h{width}_g{gamma}_b{b_str}_s{seed}_onlineFalse_noise{noise}"


noise = 0.2
batch_size = 50
seeds = [0, 1, 2, 3, 4]

for gamma in [0.75, 1.0, 1.5]:
    for width in [10, 50, 100]:
        gd_path = get_path(width, gamma, None, 0, noise)
        if not gd_path.exists():
            continue

        gd = load_run(gd_path)

        data = {"GD": gd}
        for seed in seeds:
            sgd_path = get_path(width, gamma, batch_size, seed, noise)
            if sgd_path.exists():
                data[f"s={seed}"] = load_run(sgd_path)

        title_base = f"γ={gamma}, w={width}, b={batch_size}"

        # Loss
        plot(data, title=title_base)

        # Metrics
        for metric, label in METRIC_LABELS.items():
            plot(
                data,
                metric=metric,
                ylabel=label,
                title=f"{title_base} — {label}",
                log_scale=(metric != "trace_hessian_covariance"),
            )
# %%
