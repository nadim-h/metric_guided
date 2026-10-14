"""Shared data loading and statistics. Run scripts from the repository root."""
import json
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

LANGS = ("C++", "Java", "Python")
ARMS = ("Basic", "Guided")
SEED = 0
Z95 = stats.norm.ppf(0.975)

FILES = {
    ("C++", "Basic"): "data/gptme_cpp_9825_ref.jsonl",
    ("C++", "Guided"): "data/gptme_mcp_cpp_9825_ref.jsonl",
    ("Java", "Basic"): "data/gptme_java_9795_ref_miner.jsonl",
    ("Java", "Guided"): "data/gptme_mcp_java_9795_ref_miner.jsonl",
    ("Python", "Basic"): "data/gptme_python_5000_ref_miner.jsonl",
    ("Python", "Guided"): "data/gptme_mcp_python_5000_ref_miner.jsonl",
}


def count_smells(review):
    smells = [s["smell"] for s in review.get("module") or []]
    smells += [s["smell"] for fun in review.get("funs") or [] for s in fun.get("smells") or []]
    return Counter(smells)


def load_runs():
    """One row per agent run."""
    rows = []
    for (lang, arm), path in FILES.items():
        for line in open(path):
            d = json.loads(line)
            # A CH score of 0 means the refactored file could not be reviewed: missing, not zero.
            scored = d["ref-review"]["score"] > 0
            rows.append({
                "language": lang,
                "arm": arm,
                "task": d["task"],
                "ref_task": d.get("ref-task") or "",
                "status": d["status"],
                "agent_status": d["agent-status"],
                "comp_err": d.get("comp-err") or "",
                "rt_err": d.get("rt-err") or "",
                "init_ch": d["task-review"]["score"],
                "final_ch": d["ref-review"]["score"] if scored else np.nan,
                "smells_before": count_smells(d["task-review"]),
                "smells_after": count_smells(d["ref-review"]) if scored else None,
                "ops": d.get("miner"),
                "tokens_in": d.get("tokens-in") or np.nan,
                "tokens_out": d.get("tokens-out") or np.nan,
                "turns": d.get("turns") or np.nan,
            })
    runs = pd.DataFrame(rows)
    runs["guided"] = (runs["arm"] == "Guided").astype(int)
    runs["passed"] = runs["status"] == "all-passed"
    runs["improved"] = runs["final_ch"] > runs["init_ch"]  # False when final CH is missing
    runs["effective"] = runs["passed"] & runs["improved"]
    runs["tokens"] = runs["tokens_in"] + runs["tokens_out"]
    runs["loc"] = runs["task"].map(lambda src: len(src.splitlines()))
    return runs


def pair(runs):
    """One row per task, Basic columns suffixed `_b` and Guided columns `_g`."""
    basic = runs[runs["arm"] == "Basic"].set_index(["language", "task"])
    guided = runs[runs["arm"] == "Guided"].set_index(["language", "task"])
    return basic.join(guided, lsuffix="_b", rsuffix="_g", how="inner").reset_index()


def paired_proportions(basic, guided):
    """Exact McNemar test, Cohen's g, and the rate difference with a Wald 95% CI."""
    basic, guided = np.asarray(basic, bool), np.asarray(guided, bool)
    n = len(basic)
    guided_only = int((guided & ~basic).sum())
    basic_only = int((basic & ~guided).sum())
    table = [[int((basic & guided).sum()), basic_only], [guided_only, int((~basic & ~guided).sum())]]
    delta = (guided_only - basic_only) / n
    se = np.sqrt(guided_only + basic_only - (guided_only - basic_only) ** 2 / n) / n
    return {
        "basic": basic.mean(),
        "guided": guided.mean(),
        "delta": delta,
        "ci": (delta - Z95 * se, delta + Z95 * se),
        "p": mcnemar(table, exact=True).pvalue,
        "g": max(basic_only, guided_only) / (basic_only + guided_only) - 0.5,
        "basic_only": basic_only,
        "guided_only": guided_only,
    }


def wilcoxon_r(x, y):
    """Paired Wilcoxon signed-rank test, r = |z| / sqrt(non-zero pairs)."""
    res = stats.wilcoxon(x, y, method="approx")
    nonzero = np.count_nonzero(np.asarray(x, float) - np.asarray(y, float))
    return res.pvalue, abs(res.zstatistic) / np.sqrt(nonzero)


def bootstrap_ci(statistic, *samples, n_resamples=2000):
    """Percentile bootstrap CI, resampling tasks."""
    res = stats.bootstrap(samples, statistic, n_resamples=n_resamples, paired=True,
                          vectorized=False, method="percentile", rng=np.random.default_rng(SEED))
    return res.confidence_interval


def fit_gee(df, formula):
    """Logistic GEE, clustered by task."""
    return smf.gee(formula, data=df, groups="task", family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


def break_even(model):
    """CH* = -b_guided / b_interaction, with a delta-method 95% CI."""
    keys = ["guided", "guided:init_ch"]
    bp, bi = model.params[keys]
    grad = np.array([-1 / bi, bp / bi ** 2])
    se = np.sqrt(grad @ model.cov_params().loc[keys, keys].to_numpy() @ grad)
    ch = -bp / bi
    return ch, ch - Z95 * se, ch + Z95 * se


def stars(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def mean_sd(values, fmt=",.0f"):
    return f"{np.mean(values):{fmt}} ± {np.std(values, ddof=1):{fmt}}"


def title(text):
    print(f"\n=== {text} ===")
