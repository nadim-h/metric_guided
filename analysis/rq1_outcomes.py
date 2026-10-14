"""RQ1: pass, improved-CH, and effective rates (tab:functional_results) and CH change
(tab:quality_results)."""
import numpy as np
import pandas as pd

from common import LANGS, bootstrap_ci, load_runs, mean_sd, pair, paired_proportions, stars, title, wilcoxon_r

runs = load_runs()
paired = pair(runs)

title("Pass, improved-CH, and effective refactoring rates (tab:functional_results)")
rows = []
for outcome in ("passed", "improved", "effective"):
    for lang in LANGS:
        df = paired[paired["language"] == lang]
        s = paired_proportions(df[f"{outcome}_b"], df[f"{outcome}_g"])
        rows.append({
            "Outcome": outcome,
            "Language": lang,
            "Basic %": f"{100 * s['basic']:.1f}",
            "Guided %": f"{100 * s['guided']:.1f}",
            "Δ pp": f"{100 * s['delta']:+.1f}{stars(s['p'])}",
            "95% CI": f"[{100 * s['ci'][0]:+.1f}, {100 * s['ci'][1]:+.1f}]",
            "Cohen's g": f"{s['g']:.2f}",
        })
print(pd.DataFrame(rows).to_string(index=False))

title("CH from initial to refactored, and Guided - Basic final CH, "
      "over tasks with a refactored file in both arms (tab:quality_results)")
rows = []
for lang in LANGS:
    df = paired[paired["language"] == lang]
    df = df.dropna(subset=["final_ch_b", "final_ch_g"])
    row = {"Language": lang, "Tasks": f"{len(df):,}", "Init CH": mean_sd(df["init_ch_b"], ".2f")}
    for arm, s in (("Basic", "_b"), ("Guided", "_g")):
        p, r = wilcoxon_r(df["init_ch" + s], df["final_ch" + s])
        row[f"{arm} CH"] = mean_sd(df["final_ch" + s], ".2f")
        row[f"{arm} r"] = f"{r:.2f}{stars(p)}"
    diff = (df["final_ch_g"] - df["final_ch_b"]).to_numpy()
    ci = bootstrap_ci(np.mean, diff)
    p, r = wilcoxon_r(df["final_ch_g"], df["final_ch_b"])
    row["Δ [95% CI]"] = f"{diff.mean():+.2f} [{ci.low:.2f}, {ci.high:.2f}]{stars(p)}"
    row["Δ r"] = f"{r:.2f}"
    row["Δ median"] = f"{np.median(diff):+.2f}"
    row["pairs Guided > 1 CH lower (%)"] = f"{100 * (diff < -1).mean():.1f}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
