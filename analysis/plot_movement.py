"""Exploratory CH movement plot: one segment per passing run, failed runs as dots."""
import matplotlib.pyplot as plt
import numpy as np

from common import ARMS, LANGS, load_runs

COLOR_IMPROVE, COLOR_WORSEN, COLOR_SAME, COLOR_PERFECT = "#0072B2", "#D55E00", "#7f7f7f", "#9467bd"

runs = load_runs()
panels = [(lang, arm) for lang in LANGS for arm in ARMS]
fig, axes = plt.subplots(len(panels), 1, figsize=(5.15, 0.85 * len(panels)), sharex=True)

for ax, (lang, arm) in zip(axes, panels):
    df = runs[(runs["language"] == lang) & (runs["arm"] == arm)]
    passed = df[df["passed"]].sort_values(["init_ch", "final_ch"]).reset_index(drop=True)
    failed = df[~df["passed"]].sort_values("init_ch").reset_index(drop=True)

    x0, x1 = passed["init_ch"].to_numpy(), passed["final_ch"].to_numpy()
    colors = np.select([np.isclose(x1, 10.0), x1 > x0, x1 < x0], [COLOR_PERFECT, COLOR_IMPROVE, COLOR_WORSEN], COLOR_SAME)
    ax.hlines(passed.index, np.minimum(x0, x1), np.maximum(x0, x1), colors=colors, alpha=0.35, linewidth=0.5)

    y_failed = failed.index - len(failed)
    ax.scatter(failed["init_ch"], y_failed, s=2, color="black", alpha=0.55, linewidths=0)
    ax.set_ylim(y_failed.min() - 5, len(passed) - 1)

    ax.axvline(10, linewidth=1)
    ax.axvline(9, linewidth=1, linestyle="--")
    ax.get_yaxis().set_visible(False)
    for side in ("right", "top", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0.9, 10.1)
    ax.tick_params(labelsize=6.0, length=2.5, pad=1.5)
    ax.set_title(f"{arm} {lang} | Tasks: {len(df):,} | Passed: {len(passed):,} ({len(passed) / len(df):.1%})",
                 fontsize=7.0, pad=3)

axes[-1].set_xlabel("CodeHealth (initial to refactored)", fontsize=7.0)
plt.tight_layout()
plt.savefig("figures/xxmovement_plot.pdf")
