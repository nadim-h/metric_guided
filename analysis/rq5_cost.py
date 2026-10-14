"""RQ5: token and turn overhead (tab:tokens), tokens per effective refactoring, and
the routing simulation (Guided only when initial CH >= CH*)."""
import numpy as np
import pandas as pd

from common import LANGS, break_even, fit_gee, load_runs, mean_sd, pair, stars, title, wilcoxon_r

METRICS = {"tokens": "Total tokens", "tokens_in": "Input tokens", "tokens_out": "Output tokens", "turns": "Turns"}

runs = load_runs()
paired = pair(runs)

title("Tokens and interaction turns per task with session statistics in both arms, "
      "Wilcoxon r of the paired difference (tab:tokens)")
rows = []
for col, label in METRICS.items():
    for lang in LANGS:
        df = paired[paired["language"] == lang].dropna(subset=["tokens_b", "tokens_g"])
        fmt = ".1f" if col == "turns" else ",.0f"
        p, r = wilcoxon_r(df[col + "_b"], df[col + "_g"])
        rows.append({"Metric": label, "Language": lang,
                     "Basic": mean_sd(df[col + "_b"], fmt), "Guided": mean_sd(df[col + "_g"], fmt),
                     "Factor": f"{df[col + '_g'].mean() / df[col + '_b'].mean():.1f}x{stars(p)}", "r": f"{r:.2f}"})
print(pd.DataFrame(rows).to_string(index=False))

title("Tokens per effective refactoring (all tokens spent / effective runs)")
for lang in LANGS:
    df = paired[paired["language"] == lang]
    basic, guided = (df[f"tokens_{s}"].sum() / df[f"effective_{s}"].sum() for s in "bg")
    print(f"{lang:<7} Basic {basic:,.0f}  Guided {guided:,.0f}  ratio {guided / basic:.1f}x")

title("Routing simulation: Guided only when initial CH >= CH*")
rows = []
for lang in LANGS:
    df = paired[paired["language"] == lang]
    model_df = runs[runs["language"] == lang].astype({"effective": int})
    ch_star = break_even(fit_gee(model_df, "effective ~ guided * init_ch"))[0]
    routed = df["init_ch_b"] >= ch_star
    for policy, use_guided in (("all-Basic", False), ("all-Guided", True), ("routed", routed)):
        use_guided = np.broadcast_to(use_guided, len(df))
        rows.append({"Language": lang, "Policy": policy,
                     **{label: np.nanmean(np.where(use_guided, df[col + "_g"], df[col + "_b"])) * scale
                        for col, label, scale in (("effective", "Effective %", 100), ("passed", "Pass %", 100),
                                                  ("tokens", "Mean tokens", 1))},
                     "Routed to Guided %": 100 * use_guided.mean()})
print(pd.DataFrame(rows).to_string(index=False, float_format="%.1f", formatters={"Mean tokens": "{:,.0f}".format}))
