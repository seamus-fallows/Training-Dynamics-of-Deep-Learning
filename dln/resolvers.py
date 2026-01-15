from omegaconf import OmegaConf

# GAMMA_MAX_STEPS = {0.75: 5000, 1.0: 10000, 1.5: 27000}

GAMMA_MAX_STEPS = {0.75: 2000, 1.0: 4000, 1.5: 20000}


def register():
    OmegaConf.register_new_resolver(
        "max_steps_for_gamma",
        lambda g: GAMMA_MAX_STEPS[float(g)],
        replace=True,
    )
