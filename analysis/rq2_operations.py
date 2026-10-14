"""RQ2: refactoring operations by level (tab:ref_ops, tab:level_shift), smell counts
(fig:smells), and per-smell McNemar tests (tab:mcnemar_tests)."""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import ARMS, LANGS, bootstrap_ci, load_runs, pair, paired_proportions, stars, title, wilcoxon_r

# Operation levels of Horikawa et al.:
# https://github.com/Mont9165/Agent_Refactoring_Analysis/blob/main/Refactoring_Level_Classification.md
HIGH_OPS = """add class annotation|add class modifier|add attribute annotation|add attribute modifier|
add method annotation|add method modifier|add parameter|add parameter annotation|add thrown exception type|
change class access modifier|change attribute access modifier|change method access modifier|
change thrown exception type|change type declaration kind|collapse hierarchy|encapsulate attribute|
extract interface|extract superclass|merge attribute|merge class|merge package|merge parameter|
modify class annotation|modify attribute annotation|modify method annotation|modify parameter annotation|
move attribute|move class|move method|move package|move and rename attribute|move and rename class|
move and rename method|parameterize test|parameterize variable|pull up attribute|pull up method|
push down attribute|push down method|remove class annotation|remove class modifier|remove attribute annotation|
remove attribute modifier|remove method annotation|remove method modifier|remove parameter|
remove parameter annotation|remove thrown exception type|rename attribute|rename class|rename method|
rename package|reorder parameter|split attribute|split package|split parameter"""
MEDIUM_OPS = """change attribute type|change parameter type|change return type|extract attribute|extract class|
extract method|extract subclass|extract and move method|inline attribute|inline method|localize parameter|
merge method|move and inline method|parameterize attribute|replace anonymous with class|
replace attribute with variable|replace variable with attribute|split class|split method"""
LOW_OPS = """add parameter modifier|add variable annotation|add variable modifier|assert throws|assert timeout|
change variable type|extract variable|inline variable|invert condition|merge catch|merge conditional|
merge variable|modify variable annotation|move code|remove parameter modifier|remove variable annotation|
remove variable modifier|rename parameter|rename variable|replace anonymous with lambda|replace attribute|
replace conditional with ternary|replace generic with diamond|replace loop with pipeline|
replace pipeline with loop|split conditional|split variable|try with resources"""
LEVELS = ("High", "Medium", "Low")
LEVEL = {op.strip(): level
         for level, ops in zip(LEVELS, (HIGH_OPS, MEDIUM_OPS, LOW_OPS))
         for op in ops.replace("\n", "").split("|")}

SMELLS = [
    "Large Method", "Brain Method", "Global Conditionals", "Complex Conditional",
    "Complex Method", "Deep, Nested Complexity", "Bumpy Road Ahead",
    "Overall Code Complexity", "Code Duplication", "Excess Number of Function Arguments",
    "Primitive Obsession", "String Heavy Function Arguments", "Constructor Over-Injection",
]
MIN_SMELLS = 20  # smells with fewer instances before refactoring are left out
SMELL_GROUP_BOUNDARIES = (6.5, 8.5)
BASIC_COLOR, GUIDED_COLOR = "#B0642A", "#2A6BB0"
SHORT_NAMES = {"Excess Number of Function Arguments": "Excess Arguments",
               "String Heavy Function Arguments": "String-Heavy Arguments"}
MCNEMAR_SMELLS = [
    "Large Method", "Complex Conditional", "Complex Method", "Deep, Nested Complexity",
    "Bumpy Road Ahead", "Code Duplication", "Primitive Obsession",
]


def level_shares(ops):
    levels = Counter(LEVEL[op.lower()] for op in ops)
    return pd.Series({level: levels[level] / len(ops) for level in LEVELS})


runs = load_runs()
paired = pair(runs)
# Smell comparisons need a CodeScene review of the refactored file in both arms.
reviewed = paired.dropna(subset=["smells_after_b", "smells_after_g"])

op_shares, top, level_rows = {}, set(), []
for lang in ("Java", "Python"):
    df = runs[runs["language"] == lang]
    ops = df[["arm", "ops"]].explode("ops", ignore_index=True).dropna().rename(columns={"ops": "op"})
    ops["level"] = ops["op"].str.lower().map(LEVEL)
    counts = pd.crosstab([ops["level"], ops["op"]], ops["arm"])
    op_shares[lang] = 100 * counts / counts.sum()
    top |= set(counts.sum(axis=1).nlargest(10).index)
    by_level = counts.groupby("level").sum()
    level_rows.append(100 * by_level / by_level.sum())
    print(f"{lang}: {counts['Basic'].sum():,} Basic and {counts['Guided'].sum():,} Guided operations, "
          f"Cramér's V (pooled operations) {stats.contingency.association(by_level.to_numpy()):.3f}")

    # Tasks with operations in both arms.
    both = paired[(paired["language"] == lang) & (paired["ops_b"].str.len() > 0) & (paired["ops_g"].str.len() > 0)]
    share_b, share_g = both["ops_b"].apply(level_shares), both["ops_g"].apply(level_shares)
    title(f"{lang}: mean per-task level share, {len(both):,} tasks (tab:level_shift)")
    rows = []
    for level in LEVELS:
        diff = (share_g[level] - share_b[level]).to_numpy()
        ci = bootstrap_ci(np.mean, diff)
        p, r = wilcoxon_r(share_g[level], share_b[level])
        rows.append({"Level": level, "Basic %": 100 * share_b[level].mean(), "Guided %": 100 * share_g[level].mean(),
                     "Δ pp": f"{100 * diff.mean():+.1f}{stars(p)}",
                     "95% CI": f"[{100 * ci.low:+.1f}, {100 * ci.high:+.1f}]", "r": f"{r:.2f}"})
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.1f"))

title("Share of all operations per level and for the top 10 operations of each language, % (tab:ref_ops)")
shares = pd.concat(op_shares, axis=1).fillna(0)
for lang in op_shares:
    shares[lang, "Δ pp"] = shares[lang, "Guided"] - shares[lang, "Basic"]
levels = pd.concat(dict(zip(op_shares, level_rows)), axis=1)
for lang in op_shares:
    levels[lang, "Δ pp"] = levels[lang, "Guided"] - levels[lang, "Basic"]
for level in LEVELS:
    rows = shares.loc[[idx for idx in shares.index if idx in top and idx[0] == level]]
    order = rows[[(lang, "Basic") for lang in op_shares]].sum(axis=1).sort_values(ascending=False).index
    print(pd.concat([levels.loc[[level]].set_axis([f"{level}-level"]), rows.loc[order].droplevel("level")])
          [list(shares.columns.sort_values())].to_string(float_format="%.2f"))

smells = []
for lang in LANGS:
    sub = reviewed[reviewed["language"] == lang]
    before = sum(sub["smells_before_b"], Counter())
    after = {arm: sum(sub[f"smells_after_{arm[0].lower()}"], Counter()) for arm in ARMS}
    for smell in SMELLS:
        if before[smell] >= MIN_SMELLS:
            smells.append({"Language": lang, "Smell": smell, "Before": before[smell],
                           **{f"{arm} after": after[arm][smell] for arm in ARMS}})
smells = pd.DataFrame(smells)
for arm in ARMS:
    smells[f"{arm} net %"] = 100 * (smells["Before"] - smells[f"{arm} after"]) / smells["Before"]

title(f"Smell instances before and after, net reduction %, smells with >= {MIN_SMELLS} instances (fig:smells)")
print(smells.pivot(index="Smell", columns="Language").swaplevel(axis=1).sort_index(axis=1, level=0)
      .reindex(SMELLS).dropna(how="all").to_string(float_format="%.1f"))

title("Per-smell McNemar on tasks with the smell initially, Holm within language (tab:mcnemar_tests)")
rows = []
for lang in LANGS:
    df = reviewed[reviewed["language"] == lang]
    tests = []
    for smell in MCNEMAR_SMELLS:
        before = df["smells_before_b"].map(lambda c: c[smell])
        present = before > 0
        if not present.any():
            continue
        fixed_b = df.loc[present, "smells_after_b"].map(lambda c: c[smell]) < before[present]
        fixed_g = df.loc[present, "smells_after_g"].map(lambda c: c[smell]) < before[present]
        tests.append((smell, paired_proportions(fixed_b, fixed_g)))
    holm = multipletests([s["p"] for _, s in tests], method="holm")[1]
    for (smell, s), p in zip(tests, holm):
        rows.append({"Language": lang, "Smell": smell, "Basic only": f"{s['basic_only']:,}",
                     "Guided only": f"{s['guided_only']:,}{stars(p)}", "g": f"{s['g']:.2f}"})
print(pd.DataFrame(rows).pivot(index="Smell", columns="Language").swaplevel(axis=1).sort_index(axis=1, level=0)
      .reindex(MCNEMAR_SMELLS).fillna("--").to_string())

# Ratio of smell instances after to before, Basic vs Guided, one panel per language.
fig, axes = plt.subplots(1, len(LANGS), figsize=(5.15, 2.9), sharey=True, sharex=True)
y = {smell: i for i, smell in enumerate(SMELLS)}
for ax, lang in zip(axes, LANGS):
    df = smells[smells["Language"] == lang]
    ratio = {arm: df[f"{arm} after"] / df["Before"] for arm in ARMS}
    rows = df["Smell"].map(y)
    ax.hlines(rows, ratio["Basic"], ratio["Guided"], color="#999999", linewidth=0.8, zorder=1)
    ax.scatter(ratio["Basic"], rows, s=16, marker="o", facecolors="white", edgecolors=BASIC_COLOR,
               linewidths=0.9, label="Basic", zorder=2)
    ax.scatter(ratio["Guided"], rows, s=18, marker="^", color=GUIDED_COLOR, label="Guided", zorder=3)
    ax.axvline(1, color="#444444", linewidth=0.7, zorder=0)
    for boundary in SMELL_GROUP_BOUNDARIES:
        ax.axhline(boundary, color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xlim(1 / 40, 16)
    ax.set_xticks([1 / 16, 1 / 4, 1, 4, 16], ["1/16", "1/4", "1", "4", "16"])
    ax.set_title(lang, fontsize=8, fontweight="bold", pad=3)
    ax.tick_params(labelsize=6.0, length=2.5, pad=1.5)
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
axes[0].set_yticks(range(len(SMELLS)), [SHORT_NAMES.get(smell, smell) for smell in SMELLS])
axes[0].set_ylim(len(SMELLS) - 0.5, -0.5)
axes[1].set_xlabel("Smell instances after / before refactoring (log scale)", fontsize=7)
handles, labels = axes[0].get_legend_handles_labels()
plt.tight_layout(rect=[0, 0, 1, 0.94], w_pad=0.6)
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
plt.savefig("figures/smell_change.pdf", bbox_inches="tight")
