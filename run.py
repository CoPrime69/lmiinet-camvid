"""One entry point for the whole project: tests, data, benchmark, every training run, figures, report.

    python run.py                          # everything (resumes: finished runs are skipped)
    python run.py --phase tests bench      # selected phases only
    python run.py --epochs 80 --seeds 42 7 123

Phases, in order:
  tests     correctness checks (src/tests.py) -> results/00_unit_tests.csv; stops on failure
  data      download CamVid into data/CamVid if it is missing
  bench     params / GMACs / FPS of every model -> results/benchmark.{json,csv}
  main      LMIINet, U-Net, U-Net-PM on the full training set (one run per seed)
  ablation  four LMIINet variants (seven with --full-ablation), seed 42
  scale     LMIINet and U-Net-PM on 25% and 50% of the training images, same number of steps
  hires     LMIINet evaluated on 720x960 inputs
  assets    figures/, results/*.csv, results/tables/*.tex, results/summary.md
  report    compile report.tex with latexmk if it is installed
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

PHASES = ["tests", "data", "bench", "main", "ablation", "scale", "hires", "assets", "report"]
MAIN = ["lmiinet", "unet", "unet_pm"]
# The four ablations that test the paper's components; --full-ablation adds the other three.
ABLATIONS = ["lmiinet_no_cc", "lmiinet_no_tf", "lmiinet_no_cru", "lmiinet_no_fe"]
EXTRA_ABLATIONS = ["lmiinet_fixed_cc", "lmiinet_no_cab", "lmiinet_no_aux"]
FRACTIONS = [0.25, 0.5]
# Reduced-data study: LMIINet against the parameter-matched U-Net (same budget of parameters).
SCALE_MODELS = ["lmiinet", "unet_pm"]
LOG = os.path.join("results", "run_log.txt")


def log(msg=""):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else ""
    print(line, flush=True)
    os.makedirs("results", exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_name(model, seed, fraction):
    return model + (f"_frac{fraction:g}" if fraction < 1 else "") + (f"_seed{seed}" if seed != 42 else "")


def train(model, epochs, seed, workers, fraction=1.0, force=False):
    name = run_name(model, seed, fraction)
    if os.path.exists(os.path.join("runs", name, "results.json")) and not force:
        log(f"skip {name} (finished)")
        return
    cmd = [sys.executable, "train.py", "--model", model, "--epochs", str(epochs), "--seed", str(seed),
           "--workers", str(workers), "--train-fraction", str(fraction)]
    if os.path.exists(os.path.join("runs", name, "last.pt")) and not force:
        cmd.append("--resume")
    t0 = time.time()
    for attempt, extra in enumerate(([], ["--no-cuda-graph"])):
        log(f"train {name}: {' '.join(cmd[1:] + extra)}")
        rc = _stream(cmd + extra)
        if rc == 0:
            break
        log(f"{name} exited with code {rc}" + ("; retrying without CUDA graphs" if attempt == 0 else ""))
    else:
        raise SystemExit(f"training {name} failed; see {LOG}")
    log(f"done {name} in {(time.time() - t0) / 60:.1f} min")


def _stream(cmd):
    with open(LOG, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
            lf.flush()
        return proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", nargs="+", default=["all"], choices=["all"] + PHASES)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--full-ablation", action="store_true", help="also run the three secondary ablations")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--force", action="store_true", help="retrain runs that already finished")
    a = p.parse_args()
    phases = PHASES if "all" in a.phase else a.phase
    log(f"run.py phases={phases} epochs={a.epochs} seeds={a.seeds} python={sys.version.split()[0]}")

    if "tests" in phases:
        from src.tests import run_tests
        log("correctness checks")
        if not run_tests(log=log):
            raise SystemExit("a correctness check failed; see results/00_unit_tests.csv")

    if "data" in phases:
        from src.download_camvid import ensure_camvid
        ensure_camvid("data/CamVid", log=log)

    if "bench" in phases:
        from src.benchmark import run_benchmark
        run_benchmark(iters=100, warmup=20, log=log)

    if "main" in phases:
        for s in a.seeds:
            for m in MAIN:
                train(m, a.epochs, s, a.workers, force=a.force)

    if "ablation" in phases:
        for m in ABLATIONS + (EXTRA_ABLATIONS if a.full_ablation else []):
            train(m, a.epochs, 42, a.workers, force=a.force)

    if "scale" in phases:
        for f in FRACTIONS:
            for m in SCALE_MODELS:
                # same number of optimisation steps as the full run: fewer images, more epochs
                train(m, int(round(a.epochs / f)), 42, a.workers, fraction=f, force=a.force)

    if "hires" in phases:
        ck = os.path.join("runs", "lmiinet", "best.pt")
        if os.path.exists(ck):
            log("evaluate LMIINet at 720x960")
            subprocess.run([sys.executable, "evaluate.py", "--ckpt", ck, "--split", "test",
                            "--eval-size", "720", "960"], check=True)

    if "assets" in phases:
        from src.report_assets import build_all
        build_all(log=log)
        cks = [os.path.join("runs", m, "best.pt") for m in MAIN]
        if all(os.path.exists(c) for c in cks):
            from src.visualize import qualitative_figure
            qualitative_figure(cks, ["LMIINet", "U-Net", "U-Net-PM"])
            log("qualitative figure written")

    if "report" in phases:
        if shutil.which("latexmk"):
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-outdir=build", "report.tex"])
            if os.path.exists(os.path.join("build", "report.pdf")):
                shutil.copy(os.path.join("build", "report.pdf"), "report.pdf")
                log("report.pdf compiled")
        else:
            log("latexmk not found: compile report.tex with any LaTeX installation or Overleaf")
    log("finished")


if __name__ == "__main__":
    main()
