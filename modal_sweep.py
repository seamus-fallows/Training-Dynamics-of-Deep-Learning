# """
# Run sweep.py experiments on Modal for massive parallelism.

# Usage:
#     # Same interface as sweep.py, but runs on Modal cloud workers:
#     modal run modal_sweep.py -- -cn=gph training.batch_seed=0..100 --output=outputs/my_experiment

#     # Full example matching gph_offline_loss_only.sh:
#     modal run modal_sweep.py -- -cn=gph \
#         model.model_seed=2,3 \
#         data.noise_std=0.0,0.2 \
#         model.gamma=1.5,1.0,0.75 \
#         max_steps=26000,8000,5000 \
#         model.hidden_dim=100,50,10 \
#         training.batch_size=null \
#         --zip=model.gamma,max_steps \
#         --output=outputs/gph_offline/full_batch
# """

# import modal

# app = modal.App("dln-sweep")

# image = (
#     modal.Image.debian_slim(python_version="3.12")
#     .pip_install("torch", "omegaconf", "polars", "pyyaml", "numpy", "scipy")
#     .add_local_python_source("dln")
# )


# @app.function(image=image, cpu=1, memory=128, timeout=600)
# def run_job_remote(resolved_base: dict, config_dir: str, job_overrides: dict, device: str = "cpu"):
#     import torch as t
#     t.set_num_threads(1)
#     t.set_num_interop_threads(1)

#     import traceback
#     from dln.experiment import run_experiment, run_comparative_experiment
#     from dln.utils import resolve_config

#     try:
#         cfg = resolve_config(resolved_base, config_dir, job_overrides)
#         if config_dir == "comparative":
#             result = run_comparative_experiment(cfg, device=device)
#         else:
#             result = run_experiment(cfg, device=device)
#         return True, result.history, None
#     except Exception:
#         return False, None, traceback.format_exc()


# @app.local_entrypoint()
# def main():
#     import sys
#     import time

#     from omegaconf import OmegaConf

#     from dln.overrides import parse_overrides, split_overrides, expand_sweep_params, get_output_dir
#     from dln.utils import load_base_config, resolve_config, save_sweep_config
#     from dln.results_io import SweepWriter

#     # --- Parse args (reuse sweep.py's CLI but drop --workers/--device) ---
#     import argparse

#     parser = argparse.ArgumentParser(description="Run sweep on Modal")
#     parser.add_argument("-cn", "--config-name", required=True)
#     parser.add_argument("--comparative", action="store_true")
#     parser.add_argument(
#         "--zip", action="append", dest="zip_groups", metavar="PARAMS",
#         help="Comma-separated param names to zip together",
#     )
#     parser.add_argument("--output", default=None)
#     parser.add_argument(
#         "--rerun", nargs="*", metavar="OVERRIDE",
#         help="Force re-run of matching jobs",
#     )
#     parser.add_argument("overrides", nargs="*")
#     parser.add_argument(
#         "--no-save", action="store_false", dest="save_results",
#         help="Run without saving results",
#     )
#     args = parser.parse_args()

#     # --- Expand jobs (same logic as sweep.py) ---
#     overrides = parse_overrides(args.overrides)
#     fixed_overrides, sweep_overrides = split_overrides(overrides)
#     jobs = expand_sweep_params(sweep_overrides, args.zip_groups)
#     param_keys = list(sweep_overrides.keys())

#     config_dir = "comparative" if args.comparative else "single"
#     base_config = load_base_config(args.config_name, config_dir)
#     effective_cfg = resolve_config(base_config, config_dir, fixed_overrides)
#     experiment_name = effective_cfg.experiment.name
#     resolved_base = OmegaConf.to_container(effective_cfg, resolve=True)

#     # --- Set up writer ---
#     if args.save_results:
#         output_dir = get_output_dir(experiment_name, args.output)
#         output_dir.mkdir(parents=True, exist_ok=True)
#         save_sweep_config(resolved_base, output_dir)
#         writer = SweepWriter(output_dir, param_keys)
#     else:
#         from dln.results_io import NullWriter
#         writer = NullWriter()

#     writer.consolidate_parts()

#     # --- Filter already-completed jobs ---
#     completed = writer.get_completed_params()

#     if args.rerun:
#         from sweep import _build_rerun_set
#         rerun_set = _build_rerun_set(args.rerun, param_keys, completed)
#         completed -= rerun_set

#     if completed:
#         jobs_to_run = []
#         skipped = 0
#         for job in jobs:
#             key = tuple(job.get(k) for k in param_keys)
#             if key in completed:
#                 skipped += 1
#             else:
#                 jobs_to_run.append(job)
#     else:
#         jobs_to_run = jobs
#         skipped = 0

#     total = len(jobs_to_run)
#     print(f"Running {total} jobs on Modal (skipped {skipped} already-completed)")
#     if not jobs_to_run:
#         print("Nothing to do.")
#         return

#     # --- Fan out to Modal ---
#     start_time = time.time()
#     done = 0
#     failed = 0
#     errors = []

#     inputs = [(resolved_base, config_dir, job, "cpu") for job in jobs_to_run]

#     for i, (success, history, error) in enumerate(run_job_remote.starmap(inputs)):
#         job = jobs_to_run[i]
#         if success:
#             writer.add(job, history)
#             done += 1
#         else:
#             failed += 1
#             errors.append((i, job, error))

#         finished = done + failed
#         elapsed = time.time() - start_time
#         rate = finished / elapsed if elapsed > 0 else 0
#         eta = (total - finished) / rate if rate > 0 else 0
#         print(
#             f"\rProgress: {finished}/{total} ({100 * finished / total:.1f}%) | "
#             f"{rate:.1f} jobs/s | Elapsed: {_fmt_time(elapsed)} | "
#             f"ETA: {_fmt_time(eta)} | failed: {failed}",
#             end="", flush=True,
#         )

#     writer.finalize()

#     elapsed = time.time() - start_time
#     print()
#     print("=" * 50)
#     print(f"Completed: {done}")
#     if skipped:
#         print(f"Skipped:   {skipped}")
#     print(f"Failed:    {failed}")
#     print(f"Time:      {_fmt_time(elapsed)}")

#     if errors:
#         print()
#         print("Errors:")
#         for i, job, error in errors[:10]:
#             print(f"  Job {i} {job}: {error}")
#         if len(errors) > 10:
#             print(f"  ... and {len(errors) - 10} more")


# def _fmt_time(seconds):
#     h, remainder = divmod(int(seconds), 3600)
#     m, s = divmod(remainder, 60)
#     if h:
#         return f"{h}h {m}m {s}s"
#     if m:
#         return f"{m}m {s}s"
#     return f"{s}s"
