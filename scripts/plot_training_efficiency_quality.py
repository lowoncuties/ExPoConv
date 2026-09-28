# ruff: noqa: RUF001
"""Recreate Figure 2 from the accepted compact training-result exports.

The values below retain the full precision of the following source rows:

* ``results/training/paired_training_efficiency_summary.csv``
  (SHA-256: 48f25c36b4839b398c60bc8fd622274c962bc1b6da6552102cd70d103b6a6b3f),
  ``primary_vs_baseline`` R/F rows for ``compute_seconds`` and
  ``elapsed_seconds``;
* ``results/training/paired_quality_summary.csv``
  (SHA-256: 779e302e8fffe34f5b27f753a0d30ba857afd86582b076741db3957897e66943),
  ``fixed_final_update`` / ``official_test`` / ``primary_vs_baseline`` R/F
  rows, including their five matched-seed differences.

Both paths are relative to the root of the ``ExPoConv`` repository.
Matched-seed arrays use seed order 11, 29, 47, 71, 101. Plot
labels alone are rounded for display. No estimates or confidence intervals are
recomputed. Generation is deterministic in the recorded plotting environment:
the data, seed order, SVG hash salt, and output metadata are fixed.

Run: ./.venv/bin/python scripts/plot_training_efficiency_quality.py
Output: results/training/training_efficiency_quality.{png,svg,pdf}
Requires matplotlib and numpy.
"""
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.switch_backend("Agg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY_ROOT / "results" / "training"
SOURCE_HASHES = {
    OUTPUT_DIR / "paired_training_efficiency_summary.csv": (
        "48f25c36b4839b398c60bc8fd622274c962bc1b6da6552102cd70d103b6a6b3f"
    ),
    OUTPUT_DIR / "paired_quality_summary.csv": (
        "779e302e8fffe34f5b27f753a0d30ba857afd86582b076741db3957897e66943"
    ),
}

for source_path, expected_hash in SOURCE_HASHES.items():
    actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f"accepted source hash mismatch: {source_path}")
ACCENT = "#842C46"
GREY = "#737B82"
INK = "#24272B"

DATA = [
    dict(name="ISIC16 · Poincaré U-Net", metric="Dice × 100",
         compute=(1.8214773064529528, 1.7158596992475483, 1.9335960739552551),
         elapsed=(1.3225340459798038, 1.208213682045129, 1.4476713256673563),
         delta=(-1.0016497575623885, -3.3349421561881185,
                1.3316426410633419),
         seeds=np.array([
             -0.01065998480526853,
             -0.03714107516449394,
             0.013228289012285233,
             -0.015685765794798745,
             0.00017604887415656467,
         ]) * 100),
    dict(name="CIFAR-100 · Poincaré ResNet-20", metric="Top-1 accuracy (%)",
         compute=(1.342626806347416, 1.1919736080794054,
                  1.5123210186064584),
         elapsed=(1.0996213638428136, 0.9838078152396111,
                  1.2290684471997522),
         delta=(2.2140000000000015, -0.8101414286713945,
                5.238141428671397),
         seeds=np.array([
             0.020100000000000007,
             0.010700000000000043,
             0.06380000000000002,
             0.0007999999999999674,
             0.015300000000000036,
         ]) * 100),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#BBBBBB",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "poincare-factorization-figure2-v1",
})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7),
                         gridspec_kw={"width_ratios": [1, 1.2]})
fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.18, wspace=.34)
left, right = axes
left.set_title("A  Runtime", loc="left", fontweight="bold", pad=36)
right.set_title("B  Fixed-final task quality", loc="left", fontweight="bold", pad=36)

for ax in axes:
    ax.set_ylim(-.42, 1.57)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=3)
    ax.grid(axis="x", color="#ECECEF", linewidth=.65)
    ax.set_axisbelow(True)
    ax.axhline(.53, color="#D8D8DC", linewidth=.65)

left.axvline(1, color="#A7A7AC", linestyle=(0, (3, 3)), linewidth=1)
right.axvline(0, color="#A7A7AC", linestyle=(0, (3, 3)), linewidth=1)
left.set_xlim(.90, 2.08)
left.set_xticks([1, 1.25, 1.5, 1.75, 2])
left.set_xticklabels(["1.00×", "1.25×", "1.50×", "1.75×", "2.00×"])
left.set_xlabel("Reference time / factorized time", labelpad=9)
right.set_xlim(-4.4, 7.6)
right.set_xticks([-4, -2, 0, 2, 4, 6])
right.set_xticklabels(["−4", "−2", "0", "+2", "+4", "+6"])
right.set_xlabel("Factorized − reference (score points)", labelpad=9)

for i, row in enumerate(DATA):
    y = 1-i
    left.text(.91, y+.40, row["name"], fontsize=8.5, weight="semibold",
              bbox=dict(facecolor="white", edgecolor="none", pad=1))
    right.text(-4.35, y+.40, row["metric"], fontsize=8.5, weight="semibold",
               bbox=dict(facecolor="white", edgecolor="none", pad=1))
    for key, offset, color, marker in [("compute", .11, ACCENT, "o"),
                                        ("elapsed", -.16, GREY, "s")]:
        mid, lo, hi = row[key]
        left.errorbar(mid, y+offset, xerr=[[mid-lo], [hi-mid]],
                      fmt=marker, color=color, markersize=5, capsize=3,
                      linewidth=1.5, zorder=3)
        left.text(hi+.035, y+offset, f"{mid:.2f}×", va="center", fontsize=8,
                  color=color)
    mid, lo, hi = row["delta"]
    right.errorbar(mid, y+.11, xerr=[[mid-lo], [hi-mid]], fmt="D",
                   color=ACCENT, markersize=5, capsize=3,
                   linewidth=1.5, zorder=3)
    right.text(hi+.20, y+.11, f"{mid:+.2f}", va="center", fontsize=8,
               color=ACCENT)
    seed_y = y-.16 + np.array([-.035, .035, -.035, .035, -.035])
    right.scatter(row["seeds"], seed_y, color=GREY, s=22,
                  edgecolor="white", linewidth=.5, zorder=4)

left.legend(handles=[
    Line2D([], [], color=ACCENT, marker="o", label="Update compute"),
    Line2D([], [], color=GREY, marker="s", label="Whole worker")],
    loc="lower left", bbox_to_anchor=(-.02, 1.035), frameon=False,
    ncol=2, fontsize=8, handlelength=1.7, columnspacing=1.4)
right.legend(handles=[
    Line2D([], [], color=ACCENT, marker="D", label="Mean and 95% CI"),
    Line2D([], [], color=GREY, marker="o", linestyle="none", label="Matched seed")],
    loc="lower left", bbox_to_anchor=(-.025, 1.035), frameon=False,
    ncol=2, fontsize=8, handlelength=1.7, columnspacing=1.4)

fixed_time = datetime(2026, 9, 28, tzinfo=UTC)
metadata = {
    "png": {"Creator": "Matplotlib", "Creation Time": "2026-09-28T00:00:00Z"},
    "svg": {"Creator": "Matplotlib", "Date": "2026-09-28T00:00:00Z"},
    "pdf": {
        "Creator": "Matplotlib",
        "CreationDate": fixed_time,
        "ModDate": fixed_time,
    },
}
for ext in ("png", "svg", "pdf"):
    fig.savefig(
        OUTPUT_DIR / f"training_efficiency_quality.{ext}",
        dpi=220,
        facecolor="white",
        metadata=metadata[ext],
    )
plt.close(fig)
