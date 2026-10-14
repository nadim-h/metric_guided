"""Dataset descriptives (tab:dataset) and SLoC by initial CH bin (tab:loc_bins)."""
import numpy as np
import pandas as pd

from common import LANGS, load_runs, mean_sd, title

originals = load_runs().query("arm == 'Basic'")

title("Table 1: dataset descriptives (tab:dataset)")
rows = []
for lang in LANGS:
    df = originals[originals["language"] == lang]
    below = (df["init_ch"] < 6).sum()
    rows.append({
        "Language": lang,
        "Solutions": len(df),
        "CH mean ± SD": mean_sd(df["init_ch"], ".2f"),
        "CH median": df["init_ch"].median(),
        "CH < 6 (N, %)": f"{below} ({100 * below / len(df):.1f})",
        "SLoC mean ± SD": mean_sd(df["loc"]),
        "SLoC median": df["loc"].median(),
        "SLoC max": df["loc"].max(),
    })
print(pd.DataFrame(rows).to_string(index=False))

title("SLoC by initial CH bin, floored to 1-10 (tab:loc_bins)")
bins = np.floor(originals["init_ch"]).clip(1, 10).astype(int)
table = originals.groupby(["language", bins])["loc"].agg(
    lambda loc: f"{mean_sd(loc) if len(loc) > 1 else f'{loc.mean():.0f}'} (median {loc.median():.0f})")
print(table.unstack("language").reindex(range(1, 11)).fillna("--")[list(LANGS)].to_string())
