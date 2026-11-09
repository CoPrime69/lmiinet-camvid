"""Turn every runs/*/results.json and results/benchmark.json into the report's numbers.

Writes
  figures/NN_name.{png,pdf} with results/NN_name.csv beside each figure (the numbers it plots)
  results/metrics.csv           one row per training run
  results/summary.md            readable summary of the main, ablation and data-scale tables
  results/tables/*.tex          table rows and \\newcommand macros that report.tex inputs

Needs numpy and matplotlib only. Runs of one (model, fraction) with several seeds are averaged
(mean and standard deviation). Anything not yet run is written as 'pending'.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

CLASSES = ["Sky", "Building", "Pole", "Road", "Sidewalk", "Tree",
           "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"]
PAPER = {"mIoU": 69.94, "fps": 160, "params_M": 0.72, "device": "RTX 2080Ti"}  # paper Table VIII

PRETTY = {
    "lmiinet": "LMIINet (our reimplementation)",
    "unet": "U-Net (base 32)",
    "unet_pm": "U-Net-PM (base 20, parameter-matched)",
    "lmiinet_no_cc": "without combination coefficients",
    "lmiinet_fixed_cc": "CC fixed at 0.5, not learned",
    "lmiinet_no_cru": "without CRU in LFIB",
    "lmiinet_no_tf": "without Flatten Transformer",
    "lmiinet_no_cab": "without CAB (plain Flatten Transformer)",
    "lmiinet_no_fe": "without FE (plain skip fusion)",
    "lmiinet_no_aux": "without auxiliary loss",
}
SHORT = {"lmiinet": "LMIINet", "unet": "U-Net", "unet_pm": "U-Net-PM"}
MAIN = ["lmiinet", "unet", "unet_pm"]
ABLATION = ["lmiinet", "lmiinet_no_cc", "lmiinet_no_tf", "lmiinet_no_cru", "lmiinet_no_fe"]
EXTRA_ABLATION = ["lmiinet_fixed_cc", "lmiinet_no_cab", "lmiinet_no_aux"]   # shown only if run
SCALE_MODELS = ["lmiinet", "unet_pm"]
FRACTIONS = [0.25, 0.5, 1.0]
COLORS = {"lmiinet": "#2a78d6", "unet": "#eb6834", "unet_pm": "#1baf7a"}  # categorical slots 1 to 3
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
PENDING = "pending"


# ----------------------------------------------------------------------------------- loading
def load_runs(runs_dir):
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        with open(p) as f:
            r = json.load(f)
        a = r.get("args", {})
        key = (r["model"], float(a.get("train_fraction", 1.0)), a.get("train_split", "train"))
        r["_dir"] = os.path.dirname(p)
        groups[key].append(r)
    return groups


def load_bench(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        b = json.load(f)
    return {(r["model"], tuple(r["input"])): r for r in b["rows"]}


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    # sample standard deviation (ddof=1): with three seeds the population formula understates spread
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None)


def ablations_run(groups):
    return ABLATION + [m for m in EXTRA_ABLATION if (m, 1.0, "train") in groups]


def summary(groups, bench, model, frac=1.0, seed=None):
    """Aggregate the runs of one (model, fraction). seed=42 restricts to the seed used by the
    single-seed ablation and reduced-data runs, so those are compared like with like."""
    rs = groups.get((model, frac, "train"), [])
    if seed is not None:
        rs = [r for r in rs if r.get("args", {}).get("seed", 42) == seed]
    b = bench.get((model, (360, 480)), {})
    b720 = bench.get((model, (720, 960)), {})
    return {
        "n": len(rs), "runs": rs,
        "miou": agg([r["test"]["mIoU"] for r in rs]),
        "pacc": agg([r["test"]["pixel_acc"] for r in rs]),
        "params": rs[0]["params_M"] if rs else b.get("params_M"),
        "gmacs": rs[0]["gmacs"] if rs else b.get("gmacs"),
        "fps": b.get("fps") if b else agg([r.get("fps") for r in rs])[0],
        "fps16": b.get("fps_fp16"),
        "fps_graph": b.get("fps_graph") if b else agg([r.get("fps_graph") for r in rs])[0],
        "fps_graph16": b.get("fps_graph_fp16"),
        "fps720": b720.get("fps_graph") or b720.get("fps"), "gmacs720": b720.get("gmacs"),
        "device": b.get("device") or (rs[0]["device"] if rs else None),
    }


def fmt(m, s=None, nd=2):
    if m is None:
        return PENDING
    return f"{m:.{nd}f}" + (f" $\\pm$ {s:.{nd}f}" if s is not None else "")


def fmt_md(m, s=None, nd=2):
    if m is None:
        return PENDING
    return f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None else "")


def train_minutes(runs):
    """Mean wall-clock training time over runs that were not resumed (a resumed run logs only the
    epochs after the restart)."""
    return agg([r.get("train_minutes") for r in runs if not r.get("args", {}).get("resume")])[0]


def hires_miou(runs):
    vals = []
    for r in runs:
        p = os.path.join(r["_dir"], "eval_test_720x960.json")
        if os.path.exists(p):
            with open(p) as f:
                vals.append(json.load(f)["mIoU"])
    return agg(vals)[0]


# ----------------------------------------------------------------------------------- tables
def write_tables(groups, bench, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def w(name, lines):
        with open(os.path.join(out_dir, name), "w") as f:
            f.write("\n".join(lines) + "\n")

    lines = []
    for m in MAIN:
        s = summary(groups, bench, m)
        lines.append(f"{PRETTY[m]} & {fmt(s['params'], nd=3)} & {fmt(s['gmacs'])} & {fmt(s['fps'], nd=1)} & "
                     f"{fmt(s['fps_graph'], nd=1)} & {fmt(s['fps_graph16'], nd=1)} & "
                     f"{fmt(*s['miou'])} & {fmt(*s['pacc'])} \\\\")
    lines += ["\\midrule", f"LMIINet as reported ({PAPER['device']}) & {PAPER['params_M']:.2f} & n/r & "
                           f"\\multicolumn{{3}}{{c}}{{{PAPER['fps']} (mode not stated)}} & {PAPER['mIoU']:.2f} & n/r \\\\"]
    w("main_table.tex", lines)

    lines = []
    base = summary(groups, bench, "lmiinet", seed=42)["miou"][0]
    for m in ablations_run(groups):
        s = summary(groups, bench, m, seed=42)
        params = s["params"] if s["params"] is not None else bench.get((m, (360, 480)), {}).get("params_M")
        gm = s["gmacs"] if s["gmacs"] is not None else bench.get((m, (360, 480)), {}).get("gmacs")
        d = ("" if m == "lmiinet" else PENDING) if (s["miou"][0] is None or base is None) else \
            ("" if m == "lmiinet" else f"{s['miou'][0] - base:+.2f}")
        name = "Full LMIINet" if m == "lmiinet" else PRETTY[m]
        lines.append(f"{name} & {fmt(params, nd=3)} & {fmt(gm)} & {fmt(*s['miou'])} & {d} \\\\")
    w("ablation_table.tex", lines)

    lines = []
    for m in SCALE_MODELS:
        lines.append(f"{PRETTY[m]} & " + " & ".join(fmt(*summary(groups, bench, m, f, seed=42)["miou"]) for f in FRACTIONS)
                     + " \\\\")
    w("scale_table.tex", lines)

    lines = []
    for m in MAIN:
        pc = per_class(groups, m)
        cells = [PENDING] * len(CLASSES) if pc is None else [f"{v:.1f}" for v in pc]
        lines.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
    w("perclass_table.tex", lines)

    lines = []
    for m in MAIN:
        s = summary(groups, bench, m)
        lines.append(f"{PRETTY[m]} & {fmt(s['gmacs720'])} & {fmt(s['fps720'], nd=1)} \\\\")
    w("hires_table.tex", lines)

    s_l, s_u, s_p = (summary(groups, bench, m) for m in MAIN)
    cc = [v for r in s_l["runs"] for v in r.get("learned_cc", {}).values()]
    epochs = sorted({r["args"]["epochs"] for k, rs in groups.items() for r in rs if k[1] == 1.0})
    macros = {
        "BenchDevice": (s_l["device"] or PENDING).replace("_", r"\_"),
        "LmiiMiou": fmt(s_l["miou"][0]), "UnetMiou": fmt(s_u["miou"][0]), "UnetPmMiou": fmt(s_p["miou"][0]),
        "LmiiParams": fmt(s_l["params"], nd=3), "UnetParams": fmt(s_u["params"], nd=3),
        "UnetPmParams": fmt(s_p["params"], nd=3),
        "LmiiFps": fmt(s_l["fps"], nd=1), "UnetFps": fmt(s_u["fps"], nd=1), "UnetPmFps": fmt(s_p["fps"], nd=1),
        "LmiiFpsG": fmt(s_l["fps_graph"], nd=1), "UnetFpsG": fmt(s_u["fps_graph"], nd=1),
        "UnetPmFpsG": fmt(s_p["fps_graph"], nd=1),
        "LmiiFpsGH": fmt(s_l["fps_graph16"], nd=1), "UnetFpsGH": fmt(s_u["fps_graph16"], nd=1),
        "UnetPmFpsGH": fmt(s_p["fps_graph16"], nd=1),
        "LmiiTrainMin": fmt(train_minutes(s_l["runs"]), nd=0),
        "UnetTrainMin": fmt(train_minutes(s_u["runs"]), nd=0),
        "UnetPmTrainMin": fmt(train_minutes(s_p["runs"]), nd=0),
        "LmiiPeakGB": fmt(agg([r.get("peak_train_memory_gb") for r in s_l["runs"]])[0]),
        "LmiiGmacs": fmt(s_l["gmacs"]), "UnetGmacs": fmt(s_u["gmacs"]), "UnetPmGmacs": fmt(s_p["gmacs"]),
        "PaperGap": fmt(None if s_l["miou"][0] is None else PAPER["mIoU"] - s_l["miou"][0]),
        "CcMean": fmt(float(np.mean(cc)) if cc else None),
        "CcMin": fmt(float(np.min(cc)) if cc else None), "CcMax": fmt(float(np.max(cc)) if cc else None),
        "NumSeeds": str(s_l["n"]) if s_l["n"] else PENDING,
        "LmiiMiouHiRes": fmt(hires_miou(s_l["runs"])), "LmiiFpsHiRes": fmt(s_l["fps720"], nd=1),
        "Epochs": ", ".join(str(e) for e in epochs) if epochs else PENDING,
        "LmiiMiouSeedFortyTwo": fmt(summary(groups, bench, "lmiinet", seed=42)["miou"][0]),
        "LmiiMiouStd": fmt(s_l["miou"][1]), "UnetMiouStd": fmt(s_u["miou"][1]),
        "UnetPmMiouStd": fmt(s_p["miou"][1]),
    }
    with open(os.path.join(out_dir, "macros.tex"), "w") as f:
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return macros


def per_class(groups, model):
    rs = groups.get((model, 1.0, "train"), [])
    if not rs:
        return None
    arr = np.array([[np.nan if v is None else v for v in r["test"]["per_class_iou"]] for r in rs], dtype=float)
    return np.nanmean(arr, axis=0)


def write_metrics_csv(groups, path):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run", "model", "train_fraction", "seed", "epochs", "params", "gmacs_360x480",
                     "fps_360x480", "selected_epoch", "val_mIoU_best", "test_mIoU", "test_pixel_acc",
                     "test_mean_acc", *[f"iou_{c}" for c in CLASSES], "device"])
        for key in sorted(groups):
            for r in groups[key]:
                a, t = r["args"], r["test"]
                wr.writerow([r["run"], r["model"], a.get("train_fraction"), a.get("seed"), a.get("epochs"),
                             r["params"], f"{r['gmacs']:.4f}", "" if r.get("fps") is None else f"{r['fps']:.2f}",
                             r.get("epoch_selected"), f"{r['best_monitor_mIoU']:.2f}", f"{t['mIoU']:.2f}",
                             f"{t['pixel_acc']:.2f}", f"{t['mean_acc']:.2f}",
                             *["" if v is None else f"{v:.2f}" for v in t["per_class_iou"]], r.get("device")])


def write_summary_md(groups, bench, path):
    L = ["# Results summary", "", "Generated by `run.py`. CamVid test set, 360x480.", "",
         "## Main comparison", "",
         "| Model | Params (M) | GMACs | FPS | mIoU | Pixel acc | runs |", "|---|---|---|---|---|---|---|"]
    for m in MAIN:
        s = summary(groups, bench, m)
        L.append(f"| {PRETTY[m]} | {fmt_md(s['params'], nd=3)} | {fmt_md(s['gmacs'])} | {fmt_md(s['fps'], nd=1)} | "
                 f"{fmt_md(*s['miou'])} | {fmt_md(*s['pacc'])} | {s['n']} |")
    L.append(f"| LMIINet as reported ({PAPER['device']}) | 0.72 | n/r | 160 | 69.94 | n/r | |")
    L += ["", "## Ablations (seed 42)", "", "| Variant | Params (M) | mIoU |", "|---|---|---|"]
    for m in ablations_run(groups):
        s = summary(groups, bench, m, seed=42)
        params = s["params"] if s["params"] is not None else bench.get((m, (360, 480)), {}).get("params_M")
        L.append(f"| {PRETTY[m]} | {fmt_md(params, nd=3)} | {fmt_md(*s['miou'])} |")
    L += ["", "## Training-data scale (test mIoU, seed 42)", "",
          "| Model | " + " | ".join(f"{int(f * 100)}%" for f in FRACTIONS) + " |", "|---|" + "---|" * len(FRACTIONS)]
    for m in SCALE_MODELS:
        L.append(f"| {PRETTY[m]} | " + " | ".join(fmt_md(*summary(groups, bench, m, f, seed=42)["miou"]) for f in FRACTIONS) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    return plt


def _style(ax, grid_axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _pending(ax):
    ax.text(0.5, 0.97, "results pending", ha="center", va="top", transform=ax.transAxes,
            color=INK2, fontsize=7, style="italic")


def _save(plt, fig, fig_dir, stem):
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, stem + ".pdf"))
    fig.savefig(os.path.join(fig_dir, stem + ".png"), dpi=200)
    plt.close(fig)


def _csv(res_dir, stem, header, rows):
    with open(os.path.join(res_dir, stem + ".csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)


def fig_tradeoff(plt, groups, bench, fig_dir, res_dir):
    stem = "01_miou_vs_fps"
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    _style(ax)
    rows, drawn = [], False
    for m in MAIN:
        s = summary(groups, bench, m)
        rows.append([m, s["params"], s["gmacs"], s["fps"], s["fps_graph"], s["miou"][0], s["miou"][1], s["device"]])
        if s["miou"][0] is None or s["fps_graph"] is None:
            continue
        drawn = True
        ax.scatter(s["fps_graph"], s["miou"][0], s=30 + 50 * (s["params"] or 0.5), color=COLORS[m],
                   edgecolor="white", linewidth=1.5, zorder=3, label=f"{SHORT[m]} ({s['params']:.2f}M)")
        ax.annotate(SHORT[m], (s["fps_graph"], s["miou"][0]), textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=INK)
    rows.append(["lmiinet_paper", PAPER["params_M"], None, None, PAPER["fps"], PAPER["mIoU"], None, PAPER["device"]])
    ax.scatter(PAPER["fps"], PAPER["mIoU"], marker="*", s=80, color=MUTED, zorder=3,
               label="LMIINet as reported (2080Ti)")
    if not drawn:
        _pending(ax)
    ax.set_xlabel("Frames per second (360×480, batch 1, fp32, CUDA graph)")
    ax.set_ylabel("CamVid test mIoU (%)")
    ax.legend(fontsize=6.5, frameon=False, loc="best")
    _save(plt, fig, fig_dir, stem)
    _csv(res_dir, stem, ["model", "params_M", "gmacs", "fps_eager", "fps_cudagraph", "test_mIoU", "test_mIoU_std",
                         "device"], rows)


def fig_scale(plt, groups, bench, fig_dir, res_dir):
    stem = "02_data_scale"
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    _style(ax)
    rows, drawn = [], False
    for m in SCALE_MODELS:
        xs, ys, es = [], [], []
        for fr in FRACTIONS:
            s = summary(groups, bench, m, fr, seed=42)
            mu, sd = s["miou"]
            rows.append([m, fr, int(round(367 * fr)), mu, sd, s["n"]])
            if mu is not None:
                xs.append(fr * 100)
                ys.append(mu)
                es.append(sd or 0)
        if xs:
            drawn = True
            ax.errorbar(xs, ys, yerr=es, color=COLORS[m], linewidth=1.5, marker="o", markersize=5,
                        markeredgecolor="white", capsize=2, label=SHORT[m])
            ax.annotate(SHORT[m], (xs[-1], ys[-1]), textcoords="offset points", xytext=(5, -2),
                        fontsize=7, color=INK)
    if not drawn:
        _pending(ax)
    ax.set_xlabel("Training images used (% of 367), equal number of steps")
    ax.set_ylabel("CamVid test mIoU (%)")
    ax.set_xticks([f * 100 for f in FRACTIONS])
    ax.set_xlim(15, 115)
    if drawn:
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    _save(plt, fig, fig_dir, stem)
    _csv(res_dir, stem, ["model", "train_fraction", "train_images", "test_mIoU", "test_mIoU_std", "runs"], rows)


def fig_curves(plt, groups, fig_dir, res_dir):
    stem = "03_training_curves"
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax in axes:
        _style(ax)
    rows, drawn = [], False
    for m in MAIN:
        for i, r in enumerate(groups.get((m, 1.0, "train"), [])):
            log = os.path.join(r["_dir"], "log.csv")
            if not os.path.exists(log):
                continue
            ep, loss, ev_ep, ev = [], [], [], []
            with open(log) as f:
                rd = csv.reader(f)
                header = next(rd)
                col = next(j for j, h in enumerate(header) if h.endswith("_mIoU"))
                for row in rd:
                    ep.append(int(row[0]))
                    loss.append(float(row[2]))
                    if len(row) > col and row[col]:
                        ev_ep.append(int(row[0]))
                        ev.append(float(row[col]))
                    rows.append([r["run"], m, row[0], row[2], row[col] if len(row) > col else ""])
            drawn = True
            kw = dict(color=COLORS[m], linewidth=1.4, alpha=1.0 if i == 0 else 0.4,
                      label=SHORT[m] if i == 0 else None)
            axes[0].plot(ep, loss, **kw)
            axes[1].plot(ev_ep, ev, **kw)
    if not drawn:
        _pending(axes[0])
        _pending(axes[1])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation mIoU (%)")
    if drawn:
        axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    _save(plt, fig, fig_dir, stem)
    _csv(res_dir, stem, ["run", "model", "epoch", "train_loss", "val_mIoU"], rows)


def fig_per_class(plt, groups, fig_dir, res_dir):
    stem = "05_per_class_iou"
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    _style(ax, "y")
    x = np.arange(len(CLASSES))
    width = 0.26
    rows = [[c] for c in CLASSES]
    drawn = False
    for j, m in enumerate(MAIN):
        pc = per_class(groups, m)
        for k in range(len(CLASSES)):
            rows[k].append(None if pc is None else round(float(pc[k]), 2))
        if pc is None:
            continue
        drawn = True
        ax.bar(x + (j - 1) * width, pc, width - 0.03, color=COLORS[m], label=SHORT[m])
    if not drawn:
        _pending(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("Test IoU (%)")
    ax.set_ylim(0, 100)
    if drawn:
        ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    _save(plt, fig, fig_dir, stem)
    _csv(res_dir, stem, ["class"] + [f"{m}_iou" for m in MAIN], rows)


def fig_confusion(plt, groups, fig_dir, res_dir):
    stem = "06_confusion_matrix"
    rs = groups.get(("lmiinet", 1.0, "train"), [])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    rows = []
    if rs and "confusion" in rs[0]["test"]:
        cm = np.sum([np.array(r["test"]["confusion"], dtype=float) for r in rs], axis=0)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(norm * 100, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                v = norm[i, j] * 100
                if v >= 5:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                            color="white" if v > 55 else INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("% of ground-truth pixels", fontsize=7)
        for i, c in enumerate(CLASSES):
            rows.append([c] + [int(v) for v in cm[i]])
    else:
        _pending(ax)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(CLASSES, fontsize=6.5)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    _save(plt, fig, fig_dir, stem)
    _csv(res_dir, stem, ["ground_truth \\ predicted"] + CLASSES, rows)


def fig_ablation(plt, groups, bench, fig_dir, res_dir):
    stem = "07_ablation"
    base = summary(groups, bench, "lmiinet", seed=42)["miou"][0]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    _style(ax, "x")
    names, deltas, rows = [], [], []
    for m in ablations_run(groups)[1:]:
        s = summary(groups, bench, m, seed=42)
        d = None if (s["miou"][0] is None or base is None) else s["miou"][0] - base
        params = s["params"] if s["params"] is not None else bench.get((m, (360, 480)), {}).get("params_M")
        rows.append([m, params, s["miou"][0], d])
        if d is not None:
            names.append(PRETTY[m])
            deltas.append(d)
    if deltas:
        y = np.arange(len(names))[::-1]
        ax.barh(y, deltas, color=[COLORS["lmiinet"] if d >= 0 else MUTED for d in deltas], height=0.6)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        for yi, d in zip(y, deltas):
            ax.text(d + (0.05 if d >= 0 else -0.05), yi, f"{d:+.2f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=6.5, color=INK)
    else:
        _pending(ax)
    ax.set_xlabel("Test mIoU change vs. full LMIINet (points, seed 42)")
    _save(plt, fig, fig_dir, stem)
    _csv(res_dir, stem, ["variant", "params_M", "test_mIoU", "delta_vs_full"], rows)


def build_all(runs_dir="runs", results_dir="results", fig_dir="figures", log=print):
    groups = load_runs(runs_dir)
    bench = load_bench(os.path.join(results_dir, "benchmark.json"))
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    log(f"report assets: {sum(len(v) for v in groups.values())} runs, {len(bench)} benchmark rows")
    macros = write_tables(groups, bench, os.path.join(results_dir, "tables"))
    write_metrics_csv(groups, os.path.join(results_dir, "metrics.csv"))
    write_summary_md(groups, bench, os.path.join(results_dir, "summary.md"))
    plt = _plt()
    fig_tradeoff(plt, groups, bench, fig_dir, results_dir)
    fig_scale(plt, groups, bench, fig_dir, results_dir)
    fig_curves(plt, groups, fig_dir, results_dir)
    fig_per_class(plt, groups, fig_dir, results_dir)
    fig_confusion(plt, groups, fig_dir, results_dir)
    fig_ablation(plt, groups, bench, fig_dir, results_dir)
    return macros


if __name__ == "__main__":
    build_all()
