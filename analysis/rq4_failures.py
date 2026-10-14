"""RQ4: failure modes (fig:failure_modes), failure composition (tab:failure_comp), edit
churn, and compile and runtime error causes (tab:error_causes)."""
import difflib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import ARMS, LANGS, SEED, load_runs, mean_sd, pair, stars, title, wilcoxon_r

CATEGORIES = ("Pass", "Compile", "Runtime", "Test", "Exception")
COMPILE_STATUSES = {"compile-error", "compile-timeout", "compile-error-name"}
TEST_STATUSES = {"all-test-failed", "mix-pass-failed"}
RUNTIME_TIMEOUTS = {"all-rt-timeout", "mix-pass-rt-timeout"}
FAILURE_COLORS = {"Compile": "#E69F00", "Runtime": "#D55E00", "Test": "#0072B2", "Exception": "#7f7f7f"}

# Python has no compile step: errors a compiler would reject count as compile errors.
PYTHON_COMPILE_ERRORS = {
    "SyntaxError": "Syntax/Indentation", "IndentationError": "Syntax/Indentation", "TabError": "Syntax/Indentation",
    "NameError": "Name Binding", "UnboundLocalError": "Name Binding",
    "ImportError": "Import Resolution", "ModuleNotFoundError": "Import Resolution",
}

# (cause, regex) over the lower-cased error message; the first match wins.
COMPILE_RULES = [
    ("Linker / Entrypoint", r"undefined reference|/usr/bin/ld:|collect2: error: ld returned"),
    ("Missing Symbol/Identifier", r"cannot find symbol|was not declared in this scope|does not name a type|"
                                  r"undeclared|has no member|does not have any field named"),
    ("Static Context Misuse", r"non-static (variable|method)[\s\S]*static context"),
    ("Exception Handling", r"unreported exception|is never thrown in body of corresponding try statement"),
    ("Signature Mismatch", r"cannot be applied to given types|no suitable constructor found|no matching function for call|"
                           r"candidate expects|too few arguments|too many arguments|default argument given|"
                           r"no match for 'operator|wrong number of template arguments"),
    ("Type Mismatch", r"incompatible types|invalid conversion|cannot be converted to|possible lossy conversion|"
                      r"invalid types|no type named|does not match any in class|invalid initialization"),
    ("Ambiguous Reference", r"ambiguous"),
    ("Access Control", r"private access|protected access"),
    ("Duplicate Definition", r"already defined|redefinition of"),
    ("Syntax / Parse", r"class, interface, enum, or record expected|reached end of file while parsing|expected|invalid syntax"),
    ("Missing/Invalid Include", r"fatal error: |no such file or directory"),
    ("Incomplete Type", r"incomplete type"),
    ("Invalid Return", r"return-statement with (no|a) value"),
]
RUNTIME_RULES = [
    ("Timeout", r"timeout|time limit exceeded"),
    ("Segmentation Fault", r"segmentation fault|sigsegv"),
    ("Memory Corruption/Alloc", r"double free|invalid pointer|corrupted|stack smashing|std::bad_alloc|munmap_chunk|"
                                r"malloc\(\):|free\(\):|memory limit exceeded|outofmemoryerror"),
    ("Stack Overflow / Recursion", r"stackoverflowerror|recursionerror"),
    ("Index Out of Bounds", r"indexoutofboundsexception|indexerror|std::out_of_range|string index out of range"),
    ("Missing/Null Reference", r"nullpointerexception|attributeerror|unboundlocalerror|nameerror|keyerror|nonetype"),
    ("Input Parsing", r"nosuchelementexception|inputmismatchexception|numberformatexception|eoferror|stopiteration"),
    ("Arithmetic", r"arithmeticexception|sigfpe|zerodivisionerror|floating point exception|math domain error"),
    ("Assertion Failure", r"assert"),
    ("Syntax Error", r"syntaxerror"),
    ("Type / Value Error", r"typeerror|valueerror|classcastexception|illegalargumentexception|invalid literal"),
    ("Missing File/Input", r"filenotfoundexception|filenotfounderror|no such file or directory"),
    ("Other Exception", r"traceback \(most recent call last\):|exception in thread|"
                        r"terminate called after throwing an instance|exception"),
]


def python_exception(message):
    """Exception class on the last line of a Python traceback."""
    return message.strip().splitlines()[-1].split(":", 1)[0].strip()


def match_rules(rules, message, default):
    return next((cause for cause, pattern in rules if re.search(pattern, message.lower())), default)


def classify(run):
    """Failure category, and for compile and runtime failures the error cause."""
    status = run["status"]
    if status == "all-passed":
        return "Pass", None
    # No refactored file counts as Exception, whatever the evaluator reported for it.
    if status == "exception" or not run["ref_task"].strip():
        return "Exception", None
    if status in TEST_STATUSES:
        return "Test", None
    if status == "compile-timeout":
        return "Compile", "Compile Timeout"
    if status == "compile-error-name":
        return "Compile", "Filename/Class Mismatch"
    if status in COMPILE_STATUSES:
        return "Compile", match_rules(COMPILE_RULES, run["comp_err"], "Other Compile Error")
    if run["language"] == "Python" and python_exception(run["rt_err"]) in PYTHON_COMPILE_ERRORS:
        return "Compile", PYTHON_COMPILE_ERRORS[python_exception(run["rt_err"])]
    if status in RUNTIME_TIMEOUTS:
        return "Runtime", "Timeout"
    return "Runtime", match_rules(RUNTIME_RULES, run["rt_err"], "Other Runtime Error")


def churn(original, refactored):
    """Changed lines: each differing block counts with its larger side."""
    if not refactored:
        return np.nan
    ops = difflib.SequenceMatcher(None, original.splitlines(), refactored.splitlines(), autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def share_shift(code):
    """Guided - Basic share of category `code` among failed runs (code 0 = Pass)."""
    def statistic(basic, guided, axis=-1):
        return (((guided == code).sum(axis) / (guided > 0).sum(axis))
                - ((basic == code).sum(axis) / (basic > 0).sum(axis)))
    return statistic


runs = load_runs()
runs[["category", "cause"]] = runs.apply(classify, axis=1, result_type="expand")
runs["code"] = runs["category"].map(CATEGORIES.index)
runs["churn"] = [churn(a, b) for a, b in zip(runs["task"], runs["ref_task"])]
paired = pair(runs)

title("Outcome shares of all runs, % (fig:failure_modes)")
shares = 100 * pd.crosstab([runs["language"], runs["arm"]], runs["category"], normalize="index")
print(shares[list(CATEGORIES)].to_string(float_format="%.1f"))

fig, ax = plt.subplots(figsize=(5.15, 1.75))
bars = shares.loc[[(lang, arm) for lang in LANGS for arm in ARMS]]
y = np.arange(len(bars)) + np.repeat(np.arange(len(LANGS)) * 0.5, len(ARMS))
left = np.zeros(len(bars))
for category, color in FAILURE_COLORS.items():
    width = bars[category].to_numpy()
    ax.barh(y, width, left=left, height=0.8, color=color, label=category, edgecolor="white", linewidth=0.5)
    for yi, x, w in zip(y, left, width):
        if w >= 2.5:
            ax.text(x + w / 2, yi, f"{w:.1f}", ha="center", va="center", fontsize=5.8, color="white")
    left += width
ax.set_yticks(y, [f"{lang} {arm}" for lang, arm in bars.index])
ax.invert_yaxis()
ax.set_xlim(0, 45)
ax.set_xlabel("Failed runs (% of all runs)", fontsize=7)
ax.tick_params(labelsize=6.0, length=2.5, pad=1.5)
ax.grid(axis="x", alpha=0.18, linewidth=0.5)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower right", ncol=4, fontsize=6.0, frameon=False, handlelength=1.2, columnspacing=1.0)
plt.tight_layout(pad=0.2)
plt.savefig("figures/failure_modes.pdf", bbox_inches="tight")

title("Composition of failed runs; Test is the primary contrast, the rest Holm-corrected (tab:failure_comp)")
rows = []
for lang in LANGS:
    df = paired[paired["language"] == lang]
    basic, guided = df["code_b"].to_numpy(), df["code_g"].to_numpy()
    tests = {}
    for code, category in enumerate(CATEGORIES[1:], start=1):
        tests[category] = stats.permutation_test((basic, guided), share_shift(code), permutation_type="samples",
                                                 vectorized=True, n_resamples=10_000, batch=500,
                                                 rng=np.random.default_rng(SEED))
    secondary = [c for c in tests if c != "Test"]
    p_adj = dict(zip(secondary, multipletests([tests[c].pvalue for c in secondary], method="holm")[1]))
    p_adj["Test"] = tests["Test"].pvalue
    for category, res in tests.items():
        n_b, n_g = (basic == CATEGORIES.index(category)).sum(), (guided == CATEGORIES.index(category)).sum()
        rows.append({"Language": lang, "Category": category,
                     "Basic N": n_b, "Basic %": 100 * n_b / (basic > 0).sum(),
                     "Guided N": n_g, "Guided %": 100 * n_g / (guided > 0).sum(),
                     "Δ pp": f"{100 * res.statistic:+.1f}{stars(p_adj[category])}", "p": f"{p_adj[category]:.4f}"})
print(pd.DataFrame(rows).to_string(index=False, float_format="%.1f"))

title("Exception runs: agent status, and median turns of agent timeouts vs all runs")
exceptions = runs[runs["category"] == "Exception"]
print(pd.crosstab([exceptions["language"], exceptions["arm"]], exceptions["agent_status"]).to_string())
for (lang, arm), df in runs.groupby(["language", "arm"]):
    timeouts = df[(df["category"] == "Exception") & (df["agent_status"] == "timeout")].dropna(subset=["turns"])
    if len(timeouts):
        print(f"{lang} {arm}: timeouts {timeouts['turns'].median():.1f} vs all runs {df['turns'].median():.1f} turns")

title("Changed lines per run, Wilcoxon r of the paired difference")
rows = []
for lang in LANGS:
    df = paired[paired["language"] == lang]
    both = df.dropna(subset=["churn_b", "churn_g"])
    p, r = wilcoxon_r(both["churn_g"], both["churn_b"])
    rows.append({"Language": lang,
                 "Basic": mean_sd(df["churn_b"].dropna()), "Basic median": df["churn_b"].median(),
                 "Guided": mean_sd(df["churn_g"].dropna()), "Guided median": df["churn_g"].median(),
                 "r": f"{r:.2f}{stars(p)}"})
print(pd.DataFrame(rows).to_string(index=False))

for category in ("Compile", "Runtime"):
    title(f"Top 3 {category.lower()} error causes, % of all {category.lower()} errors (tab:error_causes)")
    failed = runs[runs["category"] == category]
    for (lang, arm), causes in failed.groupby(["language", "arm"])["cause"]:
        counts = causes.value_counts()
        top = counts[~counts.index.str.startswith("Other")].head(3)
        print(f"{lang:<7} {arm:<7} " + "; ".join(f"{c} {n} ({100 * n / counts.sum():.1f}%)" for c, n in top.items()))
