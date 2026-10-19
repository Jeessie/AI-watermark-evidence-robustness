"""
C7: the four contracted figures, straight from results/<tag>/*.csv.

  python c_eprocess/make_plots.py --dir results/real --tag real

  fig1_peeking.png    the peeking demo: FPR of "check z after every token" vs alpha, and one
                      example path showing z_t crossing its threshold early while M_t does not
  fig2_length.png     detection rate vs text length (fixed-length z vs the e-processes)
  fig3_heatmap.png    detection rate over (text length x attack strength)
  fig4_stopping.png   stopping-time distribution of the e-processes

Figures are English-labelled on purpose (report / slides); the csvs keep the numbers.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eprocess as ep  # noqa: E402

ORDER = ["eprocess_betmix", "eprocess_avg", "eprocess_bet", "eprocess_wadapt", "eprocess_og",
         "eprocess_nonadapt", "fixed_z", "peeking_z"]
NICE = {"eprocess_avg": "e-process (avg)", "eprocess_bet": "e-process (betting, FTL)",
        "eprocess_betmix": "e-process (betting, mixture)",
        "eprocess_wadapt": "e-process (adaptive λ)", "eprocess_og": "e-process (online Grenander)",
        "eprocess_nonadapt": "e-process (fixed λ)", "fixed_z": "fixed-length z",
        "peeking_z": "peeking z (invalid)"}


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fig1(fpr, tag_dir, out_dir, alpha=0.01):
    rows = [r for r in fpr if r["dedup"] == "dedup" and float(r["alpha"]) == alpha]
    by = {}
    for r in rows:
        by.setdefault(r["detector"], []).append(r)
    names = [d for d in ORDER if d in by]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})

    vals = [float(by[d][0]["fpr"]) for d in names]
    hi = [float(by[d][0]["ci_hi"]) - v for d, v in zip(names, vals)]
    colors = ["#c0392b" if d == "peeking_z" else "#2c7fb8" for d in names]
    ax[0].bar(range(len(names)), vals, yerr=[np.zeros(len(vals)), hi], color=colors, capsize=3)
    ax[0].axhline(alpha, ls="--", c="k", lw=1, label=f"nominal α = {alpha:g}")
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels([NICE.get(d, d) for d in names], rotation=35, ha="right", fontsize=8)
    ax[0].set_ylabel("measured false positive rate (500 human texts)")
    ax[0].set_title("A. False positives on human text (α = 1%)\npeeking z is the only one that breaks")
    ax[0].legend(fontsize=8)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)

    # example path: z_t and M_t for one watermarked text, so the reader sees why peeking is wrong
    import json
    wm = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "flags_wm.jsonl")
    if os.path.exists(wm):
        r = json.loads(open(wm, encoding="utf-8").readline())
        flags = np.asarray(r["flags"], float)
        keep = np.asarray(r["first_occurrence"], bool)
        p = ep.pvalues(flags[keep], seed=ep.doc_seed(r["id"]))
        M = ep.eprocess_average(p)
        z = ep.z_curve(flags)
        t = np.arange(1, len(z) + 1)
        tk = np.flatnonzero(keep) + 1          # kept (deduped) positions, in original token index
        ax[1].plot(t, z, lw=1, c="#c0392b")
        thr = ep.z_threshold(alpha)
        ax[1].axhline(thr, ls=":", c="#c0392b")
        ax[1].set_ylabel("z_t, checked after every token", color="#c0392b")
        ax[1].tick_params(axis="y", colors="#c0392b")
        ax2 = ax[1].twinx()
        ax2.plot(tk, np.log10(np.maximum(M, 1e-3)), lw=1, c="#2c7fb8")
        ax2.axhline(np.log10(1 / alpha), ls=":", c="#2c7fb8")
        ax2.set_ylabel("log10 M_t (deduped tokens)", color="#2c7fb8")
        ax2.tick_params(axis="y", colors="#2c7fb8")
        st = ep.first_cross(M, alpha)
        if st:
            ax2.annotate(f"M_t ≥ 1/α at t={st}", (st, np.log10(1 / alpha)),
                         textcoords="offset points", xytext=(12, 16), fontsize=7, color="#2c7fb8")
        ax[1].set_xlabel("tokens read")
        ax[1].set_title("B. One watermarked text: z is a random walk,\nM_t is a test martingale")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_peeking.png"), dpi=160, bbox_inches="tight")
    print("  fig1_peeking.png")


def fig2(power, out_dir):
    series = {}
    for r in power:
        series.setdefault(r["detector"], []).append((int(r["n"]), float(r["tpr"])))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for d in ORDER:
        if d not in series or d == "peeking_z":   # invalid test: always rejects, not comparable
            continue
        pts = sorted(series[d])
        style = "--" if d == "fixed_z" else "-"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], style, marker="o", ms=3,
                c="#c0392b" if d == "fixed_z" else None, label=NICE.get(d, d))
    ax.set_xlabel("tokens read (fixed n for the z-test, budget t for the e-processes)")
    ax.set_ylabel("detection rate (α = 1%)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=.3)
    ax.set_title("Detection rate vs text length (500 human texts gave the FPR)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_length.png"), dpi=160)
    print("  fig2_length.png")


# 攻击臂：一个 file 一张热图。主臂是 B 组的逐句改写（200 篇/档），
# 强臂是我用 deepseek-v4.1-flash 做的改写（20 篇/档，客观强度更高）。
ATTACK_ARMS = {
    "flags_attacked_wm.jsonl": dict(
        out="fig3_heatmap.png",
        subtitle="B group: Qwen2.5-1.5B sentence paraphrase, 200 texts per level"),
    "flags_attacked_api_wm.jsonl": dict(
        out="fig3_heatmap_strong.png",
        subtitle="strong arm: deepseek-v4.1-flash rewrite, 20 texts per level"),
}


def fig3(attack, out_dir, detectors=("eprocess_betmix", "fixed_z"),
         only_file="flags_attacked_wm.jsonl", out=None, subtitle=""):
    rows = [r for r in attack if r["detector"] in detectors and r["n"].isdigit()
            and r["file"] == only_file]
    if not rows:
        print(f"  (no attack.csv rows for {only_file} -- skipped)")
        return
    fracs = sorted({float(r["attack_frac"]) for r in rows})
    ns = sorted({int(r["n"]) for r in rows})
    fig, axes = plt.subplots(1, len(detectors), figsize=(4.4 * len(detectors), 3.8), squeeze=False)
    for ax, det in zip(axes[0], detectors):
        M = np.full((len(fracs), len(ns)), np.nan)
        for r in rows:
            if r["detector"] == det:
                M[fracs.index(float(r["attack_frac"])), ns.index(int(r["n"]))] = float(r["tpr"])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto", origin="lower")
        ax.set_xticks(range(len(ns)), ns, rotation=45, fontsize=7)
        ax.set_yticks(range(len(fracs)), [f"{f:.0%}" for f in fracs])
        ax.set_xlabel("tokens read")
        if det == detectors[0]:
            ax.set_ylabel("fraction of text rewritten")
        ax.set_title(NICE.get(det, det), fontsize=9)
        for i in range(len(fracs)):
            for j in range(len(ns)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=axes[0].tolist(), label="detection rate (α = 1%)", shrink=.85, pad=0.02)
    fig.suptitle("Detection rate over text length x attack strength", fontsize=10, y=1.10)
    if subtitle:
        fig.text(0.5, 1.01, subtitle, ha="center", fontsize=8, style="italic")
    name = out or "fig3_heatmap.png"
    fig.savefig(os.path.join(out_dir, name), dpi=160, bbox_inches="tight")
    print(f"  {name}")


def fig4(stops, out_dir):
    if not stops:
        print("  (no stops.csv yet -- fig4 skipped)")
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for d in ORDER:
        v = [int(r["stop"]) for r in stops if r["detector"] == d and int(r["stop"]) > 0]
        if not v:
            continue
        v = np.sort(v)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", lw=1.3, label=NICE.get(d, d))
    ax.set_xlabel("stopping time (tokens read)")
    ax.set_ylabel("fraction of watermarked texts stopped by t")
    ax.set_xlim(0, 150)
    ax.grid(alpha=.3)
    ax.set_title("How many tokens are enough? (α = 1%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_stopping.png"), dpi=160)
    print("  fig4_stopping.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/real")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--attack_file", nargs="+", default=list(ATTACK_ARMS),
                    help="which `file` column(s) of attack.csv feed fig3 (one figure each)")
    args = ap.parse_args()
    d = args.dir
    fig1(read(os.path.join(d, "fpr.csv")), d, d, args.alpha)
    fig2(read(os.path.join(d, "power_curve.csv")), d)
    for f in args.attack_file:
        arm = ATTACK_ARMS.get(f, {})
        fig3(read(os.path.join(d, "attack.csv")), d, only_file=f,
             out=arm.get("out"), subtitle=arm.get("subtitle", ""))
    fig4(read(os.path.join(d, "stops.csv")), d)


if __name__ == "__main__":
    main()
