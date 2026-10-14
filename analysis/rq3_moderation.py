"""RQ3: moderation by initial CH. GEE models and break-even points (tab:gee_regression,
fig:rq3), effects per CH, CH gains below the break-even point, and the file-size check."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import LANGS, Z95, break_even, fit_gee, load_runs, stars, title

OUTCOMES = {"effective": "P(effective refactoring)", "improved": "P(CH improved)", "passed": "P(pass)"}
TERMS = ["Intercept", "guided", "init_ch", "guided:init_ch"]
BASIC_COLOR, GUIDED_COLOR = "#B0642A", "#2A6BB0"


def margin(model, ch):
    """P(outcome) for Basic and Guided at initial CH = ch, and their difference with a delta-method SE."""
    rows = pd.DataFrame({"guided": [0, 1], "init_ch": [ch, ch]})
    p_b, p_g = model.predict(rows)
    # average=True divides the weighted sum by the two rows, hence weights of -2 and 2.
    diff = model.get_prediction(rows, which="mean", average=True, agg_weights=np.array([-2.0, 2.0]))
    return p_b, p_g, diff.predicted, diff.se


def draw_row(axes, runs, models, outcome):
    grid = np.linspace(1, 10, 200)
    for ax, lang in zip(axes, LANGS):
        m = models[outcome, lang]
        ch_star, lo, hi = break_even(m)

        # Region where the model expects guidance to be worse than Basic.
        has_break_even = 1 <= ch_star <= 10
        if has_break_even:
            ax.axvspan(1, ch_star, color="#999999", alpha=0.10, zorder=0)
            ax.axvspan(max(lo, 1), min(hi, 10), color="#666666", alpha=0.16, zorder=1)
            ax.axvline(ch_star, color="#444444", linestyle=":", linewidth=1.2, zorder=2)

        # Observed per-bin rates, marker area proportional to the number of tasks.
        df = runs[runs["language"] == lang]
        for arm, color, marker in (("Basic", BASIC_COLOR, "o"), ("Guided", GUIDED_COLOR, "^")):
            sub = df[df["arm"] == arm]
            rate = sub.groupby(np.floor(sub["init_ch"]).clip(1, 10))[outcome].agg(["mean", "size"])
            ax.scatter(rate.index, rate["mean"], s=3 + 60 * np.sqrt(rate["size"] / rate["size"].max()),
                       facecolors="none", edgecolors=color, marker=marker, linewidths=0.8, alpha=0.85, zorder=3)

        basic, guided = (m.predict(pd.DataFrame({"guided": g, "init_ch": grid})) for g in (0, 1))
        ax.plot(grid, basic, color=BASIC_COLOR, linewidth=1.5, label="Basic", zorder=4)
        ax.plot(grid, guided, color=GUIDED_COLOR, linewidth=1.5, linestyle="--", label="Guided", zorder=4)

        if has_break_even:
            right = ch_star < 8
            ax.annotate(f"CH*={ch_star:.1f}", xy=(ch_star, 0.985), xytext=(ch_star + (0.3 if right else -0.3), 0.985),
                        fontsize=6.0, color="#333333", va="top", ha="left" if right else "right")
        ax.set_xlim(1, 10)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, 11, 2))
        ax.tick_params(labelsize=6.0, length=2.5, pad=1.5)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    axes[0].set_ylabel(OUTCOMES[outcome], fontsize=7)


runs = load_runs()
runs[list(OUTCOMES)] = runs[list(OUTCOMES)].astype(int)

models = {}
for outcome in OUTCOMES:
    for lang in ("Overall",) + LANGS:
        df = runs if lang == "Overall" else runs[runs["language"] == lang]
        models[outcome, lang] = fit_gee(df, f"{outcome} ~ guided * init_ch")

for outcome in OUTCOMES:
    title(f"GEE: {outcome} ~ guided * init_ch, coefficient (SE) (tab:gee_regression)")
    table = {}
    for lang in ("Overall",) + LANGS:
        m = models[outcome, lang]
        col = {t: f"{m.params[t]:+.3f} ({m.bse[t]:.3f}){stars(m.pvalues[t])}" for t in TERMS}
        ch, lo, hi = break_even(m)
        col["CH*"] = f"{ch:.1f} [{lo:.1f}, {hi:.1f}]" if lang != "Overall" else "--"
        col["N"] = f"{int(m.nobs):,}"
        table[lang] = col
    print(pd.DataFrame(table).to_string())

title("Model-implied effect of guidance on P(effective) by initial CH, with tasks per floored CH bin")
for lang in LANGS:
    m = models["effective", lang]
    tasks = np.floor(runs.query("arm == 'Basic' and language == @lang")["init_ch"]).clip(1, 10).value_counts()
    rows = []
    for ch in range(1, 11):
        p_b, p_g, d, se = margin(m, ch)
        p = 2 * stats.norm.sf(abs(d / se))
        rows.append({"CH": ch, "Basic": p_b, "Guided": p_g, "Δ pp": f"{100 * d:+.1f}{stars(p)}",
                     "95% CI": f"[{100 * (d - Z95 * se):+.1f}, {100 * (d + Z95 * se):+.1f}]",
                     "tasks": tasks.get(ch, 0)})
    print(f"\n{lang}")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.2f"))

title("Median CH gain of passing runs below the break-even point CH*")
for lang in LANGS:
    ch_star = break_even(models["effective", lang])[0]
    df = runs[(runs["language"] == lang) & (runs["init_ch"] < ch_star) & (runs["passed"] == 1)]
    gain = (df["final_ch"] - df["init_ch"]).groupby(df["arm"]).median()
    print(f"{lang:<7} CH* = {ch_star:.2f}  Basic {gain['Basic']:+.2f}  Guided {gain['Guided']:+.2f}")

title("File-size check: SLoC and Guided session size vs initial CH (Spearman rho, medians below/above CH 7)")
for lang in LANGS:
    df = runs[runs["language"] == lang]
    basic, guided = df[df["arm"] == "Basic"], df[(df["arm"] == "Guided")].dropna(subset=["tokens"])
    print(f"\n{lang}")
    for label, sub, col, fmt in (("SLoC", basic, "loc", ".0f"), ("Guided tokens", guided, "tokens", ",.0f"),
                                 ("Guided turns", guided, "turns", ".0f")):
        rho, p = stats.spearmanr(sub["init_ch"], sub[col])
        below, above = sub.loc[sub["init_ch"] < 7, col], sub.loc[sub["init_ch"] >= 7, col]
        print(f"  {label:<14} rho {rho:+.3f}{stars(p)}  median CH<7 {below.median():{fmt}} (n={len(below)})"
              f"  CH>=7 {above.median():{fmt}} (n={len(above)})")
    failed = df["arm"].eq("Guided") & df["agent_status"].eq("process-failed")
    print(f"  MCP process failures: {failed.sum()}, median SLoC {df.loc[failed, 'loc'].median():.0f}"
          f" vs {df.loc[df['arm'].eq('Guided') & ~failed, 'loc'].median():.0f} for the other Guided runs")

# sn-jnl \textwidth is 372pt (~5.15in); drawn at final size so the type stays legible.
fig, axes = plt.subplots(3, 3, figsize=(5.15, 4.5), sharey=True, sharex=True)
for row, outcome in zip(axes, OUTCOMES):
    draw_row(row, runs, models, outcome)
for ax, lang in zip(axes[0], LANGS):
    ax.set_title(lang, fontsize=8, fontweight="bold", pad=3)
for ax in axes[2]:
    ax.set_xlabel("Initial CodeHealth", fontsize=7)
plt.tight_layout(rect=[0, 0, 1, 0.978])
handles, labels = axes[0][0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, frameon=True, framealpha=1.0,
                    edgecolor="#666666", handlelength=2.2, columnspacing=1.8, borderpad=0.4,
                    bbox_to_anchor=(0.5, 1.0))
legend.get_frame().set_linewidth(0.6)
plt.savefig("figures/rq3_moderation.pdf", bbox_inches="tight")
