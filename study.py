"""Publication campaigns. Run from this checkout with Python 3.10+."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ARCH = Path("experiments/04_architectures")
EXT = Path("experiments/05_external_benchmarks")
DT = Path("experiments/06_dt_sensitivity/run_dt_sensitivity.py")
TOPOLOGIES = "global,ring,grid_von_neumann,wheel,random_k4,random_k8,small_world_k4,scale_free_k4"


def run(script, *args):
    subprocess.run([sys.executable, str(script), *map(str, args)], cwd=ROOT, check=True)


def read(path):
    return json.loads(path.read_text())


def pin(path, settings):
    if path.exists() and read(path) != settings:
        raise SystemExit(f"Existing protocol differs: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(settings, indent=2) + "\n")
        temp.replace(path)
    return path


@contextmanager
def campaign_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".campaign.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another campaign is active in {folder}")
        yield


def sensitivity(base, pilot):
    directory = base / "01_dt"
    selection = read(ROOT / "configs/campaigns/cpso_dt_statistical_selection_m2.json")
    if pilot:
        selection.update(campaign_id="dt_pilot", iterations=30, seeds=list(range(810001, 810006)),
                         dt_values=[0.175, 0.38125, 1., 3.6, 30.])
    config = pin(directory / "selection.json", selection)
    for name in ("selection", "holdout"):
        if name == "holdout":
            config = directory / "holdout.json"
            if not config.exists():
                run("scripts/make_dt_validation_config.py", "--selection-config", directory / "selection.json",
                    "--selection-summary", directory / "selection/analysis/selection_summary.json",
                    "--output", config, "--first-seed", 820001 if pilot else 920001)
        raw = directory / name / "raw"
        run(DT, "--config", config, "--output", raw,
            *(["--resume"] if (raw / "study_config.json").exists() else []))
        run("analysis/analyze_dt_statistical.py", "--input", raw, "--output-dir", raw.parent / "analysis")


def replot_scaling(base):
    sys.path.insert(0, str(ROOT / ARCH))
    from run_publication_scaling import scaling_figure
    folder = base / "02_parallel"
    scaling_figure([folder / "rastrigin", folder / "inverse_diffusion"],
                   folder / "fig_strong_scaling.pdf")


def scaling(base, dt, pilot):
    for benchmark in ("rastrigin", "inverse_diffusion"):
        run(ARCH / "run_publication_scaling.py", "--benchmark", benchmark, "--dt", dt,
            "--dimensions", "30,100", "--topologies", TOPOLOGIES,
            "--seeds", "85001,85002" if pilot else ",".join(str(n) for n in range(95001, 95011)),
            "--repetitions", 2 if pilot else 5, "--iterations", 8 if pilot else 64,
            "--work", (64 if pilot else 2048) if benchmark == "inverse_diffusion" else 0,
            "--no-figures", "--output", base / "02_parallel" / benchmark)
    replot_scaling(base)


def dynamics(base, dt, pilot):
    cfg = read(ROOT / "configs/campaigns/dynamics.json")
    cfg["dt"] = dt
    if pilot:
        cfg.update(iterations=10, seeds=cfg["seeds"][:2])
    folder = base / "02_parallel/dynamics"
    config = pin(folder / "protocol.json", cfg)
    run(ARCH / "run_architecture_study.py", config, "--output", folder / "raw")


def external(base, dt, pilot, suite):
    prefix = {"bbob": "bbob", "largescale": "bbob_largescale", "cec": "cec2013"}[suite]
    cfg = read(ROOT / f"configs/benchmarks/{prefix}_confirmatory.json")
    cfg.update(cpso_dt=dt, campaign_id=f"{suite}_{base.name}")
    if pilot and suite == "cec":
        cfg.update(function_ids=[1, 4, 13, 14], seeds=[1, 2], max_evaluations=800,
                   milestones=[400, 800])
    elif pilot:
        dims = "5,20" if suite == "bbob" else "20,80"
        cfg.update(instances="instances: 1-2", suite_options=f"dimensions: {dims} function_indices: 1,8,21",
                   budget_multiplier=80)
    elif suite == "cec":
        cfg.update(seeds=list(range(1, 6)), max_evaluations=4000, milestones=[400, 2000, 4000])
    else:
        dims = "5,20" if suite == "bbob" else "20,80"
        cfg.update(instances="instances: 1-5", suite_options=f"dimensions: {dims} function_indices: 1-24",
                   budget_multiplier=400 if suite == "bbob" else 80)
    folder = base / "03_comparison" / suite
    config = pin(folder / "protocol.json", cfg)
    driver = "run_cec2013_lsgo.py" if suite == "cec" else "run_bbob.py"
    run(EXT / driver, "--config", config, "--output", folder / "raw")
    run("analysis/postprocess_external_benchmarks.py", "--results-root", folder / "raw",
        "--output", folder / "analysis")


def topology_campaign(base):
    for obj in ("rastrigin", "inverse_diffusion"):
        run(ARCH / "run_publication_scaling.py", "--benchmark", obj, "--dt", 1,
            "--topologies", TOPOLOGIES, "--dimensions", "30,100", "--population", 64,
            "--iterations", 96, "--workers", "1,2,4,8", "--architectures", "particle_parallel",
            "--seeds", ",".join(str(s) for s in range(117001, 117025)), "--repetitions", 3,
            "--work", 256 if obj == "inverse_diffusion" else 0, "--no-figures",
            "--output", base / obj)
    run(ARCH / "topology_study.py", base, base / "analysis")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["dt", "scaling", "scaling-figure", "topologies", "topology-figures", "dynamics", "bbob", "largescale", "cec"])
    ap.add_argument("--profile", choices=["pilot", "paper"], default="pilot")
    args = ap.parse_args()
    base = ROOT / "results" / args.profile
    # Timing campaigns should run separately even when different stages are invoked.
    with campaign_lock(ROOT / "results"):
        if args.stage == "topologies":
            topology_campaign(ROOT / "results/topology_m2_v1")
            return
        if args.stage == "topology-figures":
            directory = ROOT / "results/topology_m2_v1"
            run(ARCH / "topology_study.py", directory, directory / "analysis")
            return
        if args.stage == "scaling-figure":
            replot_scaling(base)
            return
        if args.stage == "dt":
            sensitivity(base, args.profile == "pilot")
            return
        summary = base / "01_dt/holdout/analysis/selection_summary.json"
        if not summary.is_file():
            ap.error(f"First run: python study.py dt --profile {args.profile}")
        dt = read(summary)["frozen_dt_if_holdout"]
        if args.stage == "scaling":
            scaling(base, dt, args.profile == "pilot")
        elif args.stage == "dynamics":
            dynamics(base, dt, args.profile == "pilot")
        else:
            external(base, dt, args.profile == "pilot", args.stage)


if __name__ == "__main__":
    main()
