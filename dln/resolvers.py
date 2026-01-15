from omegaconf import OmegaConf

GAMMA_MAX_STEPS = {0.75: 5000, 1.0: 10000, 1.5: 27000}


def register():
    OmegaConf.register_new_resolver(
        "max_steps_for_gamma",
        lambda g: GAMMA_MAX_STEPS[float(g)],
        replace=True,
    )

    OmegaConf.register_new_resolver(
        "eval_points",
        lambda max_steps, n_points=250: max(1, int(float(max_steps) / float(n_points))),
        replace=True,
    )
