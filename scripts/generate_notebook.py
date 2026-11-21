#!/usr/bin/env python3
"""Generate the course notebook from a clean cell list."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(src: str):
    cells.append(nbf.v4.new_markdown_cell(src.strip() + "\n"))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip() + "\n"))


md(
    """
# Stack Overflow Question Quality Classifier

**CSE 575 – Statistical Machine Learning** · **P11: Learning from CQA data**

**Research question:** Can *textual and structural* features of a Stack Overflow question predict community-judged quality — without confusing quality with post type?

### What changed vs an earlier draft
Earlier models largely detected Stack Overflow **opinion-type** posts (`advice`, `best-practices`, `tooling-recommendation`) rather than writing quality. This notebook fixes that:

1. **Primary task** = normal Q&A questions only (opinion-types held out)
2. **Feature sets separated**: `text` | `struct` | `tags` | `text+struct` | `all`
3. **Headline metric = ROC-AUC** (all-high dummy F1 is a strong but uninformative baseline)
4. **Model selection by CV AUC only** (no test-set peeking)
5. **Chronological split** + **drop Score==0** sensitivity + **all-posts** secondary analysis (documents the tag confound)
6. Conservative **60-day** age filter; extracted **code text** used in TF-IDF; no hard-coded user paths

### How to run
Select the project `.venv` kernel and **Run All** (~8–15 minutes), or from the repo root:

```bash
MPLBACKEND=Agg python scripts/run_rigorous_pipeline.py
```
"""
)

code(
    """
# 0. Setup
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

ROOT = Path.cwd()
if not (ROOT / "data" / "stackoverflow_questions_2023plus.csv").exists():
    # allow running from scripts/
    if (ROOT.parent / "data" / "stackoverflow_questions_2023plus.csv").exists():
        ROOT = ROOT.parent
DATA_PATH = ROOT / "data" / "stackoverflow_questions_2023plus.csv"
FIG_DIR, RES_DIR = ROOT / "figures", ROOT / "results"
assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"
print("ROOT =", ROOT)
print("DATA =", DATA_PATH)
print("Note: CSV filename is historical; rows are from 2026-04 to 2026-10.")
"""
)

md(
    """
## 1. Run the rigorous pipeline

The full experiment lives in `scripts/run_rigorous_pipeline.py` (keeps the notebook readable and avoids duplicated logic). Re-running is safe; outputs overwrite `figures/` and `results/`.
"""
)

code(
    """
# 1. Execute pipeline (skip if summary.json already fresh and SKIP_RERUN=True)
import subprocess, sys, os
SKIP_RERUN = os.environ.get("SKIP_RERUN", "0") == "1"
summary_path = RES_DIR / "summary.json"
if SKIP_RERUN and summary_path.exists():
    print("SKIP_RERUN=1 — loading existing results")
else:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_rigorous_pipeline.py")]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
print("Done.")
"""
)

md("## 2. Data reality check: opinion-type confound & temporal drift")

code(
    """
# 2. Load summary + confound tables
summary = json.loads((RES_DIR / "summary.json").read_text())
type_stats = pd.read_csv(RES_DIR / "post_type_stats.csv")
monthly = pd.read_csv(RES_DIR / "monthly_drift.csv")
display(Markdown(
    f\"\"\"**Snapshot date:** {summary['snapshot_date']}  ·  **Age filter:** ≥ {summary['min_age_days']} days  
**After filter:** {summary['n_after_age_filter']} questions  
**Normal Q&A:** {summary['n_normal']} (high-quality rate {summary['normal_high_rate']:.1%})  
**Opinion-type:** {summary['n_opinion']} (high-quality rate {summary['opinion_high_rate']:.1%})  
`FavoriteCount` is always 0 in this API export (not usable).
\"\"\"
))
display(type_stats)
display(Image(filename=str(FIG_DIR / "01_score_distribution.png")))
display(Image(filename=str(FIG_DIR / "02_temporal_drift.png")))
display(monthly.round(3))
"""
)

md(
    """
## 3. Primary task definition

- **Population:** normal questions only (exclude opinion-type tags)
- **Label:** `Score >= 1` → high, `Score <= 0` → low  
  Limitation: Score 0 often means *unvoted*, not necessarily bad — see sensitivity analysis.
- **No leakage:** score, views, answers, comments, accepted-answer, reputation, ids/dates excluded from features
- **Features**
  - `text`: TF-IDF on title + body prose + **code text**
  - `struct`: lengths, code/links/lists/style (tags **not** included)
  - `tags`: binary tag indicators alone
  - `text+struct`: content quality signal without tags
  - `all`: text + struct + tags
"""
)

code(
    """
# 3. Primary class balance / structural EDA figures
display(Image(filename=str(FIG_DIR / "03_class_balance_normal.png")))
display(Image(filename=str(FIG_DIR / "04_structural_by_class.png")))
"""
)

md("## 4. Primary results (normal questions, random stratified split)")

code(
    """
# 4. Primary metrics — headline is ROC-AUC
cv = pd.read_csv(RES_DIR / "cv_results.csv")
test = pd.read_csv(RES_DIR / "test_results.csv")
primary_cv = cv[cv.Suite == "primary_normal_random"].sort_values("CV ROC-AUC (mean)", ascending=False)
primary_test = test[test.Suite == "primary_normal_random"].sort_values("ROC-AUC", ascending=False)

best = summary["primary_best_by_cv"]
print(f"CV-selected model: {best['model']} | {best['features']}")
print("Held-out test:", {k: round(v, 3) for k, v in summary['primary_best_test'].items()})
print("Baselines:")
display(pd.DataFrame(summary["primary_baselines"]).round(3))
print("Note: all-high/majority F1 looks strong because classes are ~balanced; AUC=0.5 shows they are uninformative.")

display(Markdown("### Cross-validation (selection criterion = ROC-AUC)"))
display(primary_cv[["Model", "Features", "CV ROC-AUC (mean)", "CV ROC-AUC (std)", "Best params"]].round(3).head(15))
display(Markdown("### Held-out test"))
display(primary_test[["Model", "Features", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].round(3))

display(Image(filename=str(FIG_DIR / "07_featureset_heatmap_primary.png")))
display(Image(filename=str(FIG_DIR / "08_test_metrics_primary.png")))
display(Image(filename=str(FIG_DIR / "06_roc_primary_textstruct.png")))
display(Image(filename=str(FIG_DIR / "05_confusion_primary_best.png")))
"""
)

md(
    """
### Reading the primary table
- **Recommended content-quality model:** Logistic Regression with `text+struct` (no tags) — AUC ≈ content/writing signal.
- **`all` (with tags)** can be slightly higher; topical tags still carry some signal even on normal questions, but they are *not* the same as writing quality.
- **`struct` alone** is weaker than text once opinion-types are removed — the old claim that “structural features dominate” was mostly **tags** bundled into “structural”.
"""
)

code(
    """
# 4b. Bootstrap deltas (LR)
boot = summary["bootstrap"]
rows = []
for k, v in boot.items():
    rows.append({
        "contrast": k,
        "delta_AUC": round(v["delta_auc_mean"], 3),
        "AUC 95% CI": f"[{v['delta_auc_ci95'][0]:.3f}, {v['delta_auc_ci95'][1]:.3f}]",
        "delta_F1": round(v["delta_f1_mean"], 3),
        "F1 95% CI": f"[{v['delta_f1_ci95'][0]:.3f}, {v['delta_f1_ci95'][1]:.3f}]",
    })
display(pd.DataFrame(rows))
display(Image(filename=str(FIG_DIR / "09_lr_coefficients_primary.png")))
display(Image(filename=str(FIG_DIR / "10_lr_structural_primary.png")))
"""
)

md("## 5. Chronological robustness (normal questions)")

code(
    """
# 5. Chronological split
chrono_test = test[test.Suite == "primary_normal_chrono"].sort_values("ROC-AUC", ascending=False)
print("CV-selected:", summary["chrono_best_by_cv"])
print("Chrono held-out:", {k: round(v, 3) for k, v in summary["chrono_best_test"].items()})
display(chrono_test[["Model", "Features", "F1", "ROC-AUC"]].round(3).head(12))
display(Image(filename=str(FIG_DIR / "11_roc_chrono.png")))
print("Random-split AUC falls under distribution shift — models are not fully time-stable.")
"""
)

md(
    """
## 6. Secondary analysis: all posts (documents the tag confound)

When opinion-type posts are included, **tags alone** nearly match the full model. This is expected: those tags are almost deterministic predictors of low scores. This suite is reported for transparency, **not** as the primary quality result.
"""
)

code(
    """
# 6. All-posts confound table
all_test = test[test.Suite == "secondary_all_posts"].sort_values("ROC-AUC", ascending=False)
display(all_test[["Model", "Features", "F1", "ROC-AUC"]].round(3))
display(Image(filename=str(FIG_DIR / "12_featureset_heatmap_allposts.png")))
print("Secondary CV-selected:", summary["secondary_all_best_by_cv"])
print("Secondary test:", {k: round(v, 3) for k, v in summary["secondary_all_best_test"].items()})
"""
)

md(
    """
## 7. Sensitivity: drop Score == 0

Score 0 is ambiguous (often “no votes yet”). Here we keep only Score ≥ 1 vs Score ≤ −1 on normal questions.
"""
)

code(
    """
# 7. Drop-zero sensitivity
sens = test[test.Suite == "sensitivity_drop_zero"].sort_values("ROC-AUC", ascending=False)
display(sens[["Model", "Features", "F1", "ROC-AUC"]].round(3))
print("CV-selected:", summary["sensitivity_drop_zero_best_by_cv"])
print("Test:", {k: round(v, 3) for k, v in summary["sensitivity_drop_zero_best_test"].items()})
"""
)

md(
    """
## 8. Conclusions

1. **Primary finding:** On normal Stack Overflow questions, posting-time text + structure predict quality better than chance (**LR `text+struct` ROC-AUC ≈ 0.67**; with tags **≈ 0.69**).  
2. **Headline metric should be ROC-AUC**, not F1 vs a stratified dummy — an all-high classifier gets F1 ≈ 0.67 with AUC 0.5 when classes are balanced.  
3. **The earlier “structural dominates text” story was largely a tag/post-type effect.** With tags separated and opinion posts removed, text matters; pure structure is weaker.  
4. **All-posts models look stronger mainly because they detect opinion-type tags** (secondary AUC ≈ 0.82 with tags).  
5. **Temporal shift is real:** chronological evaluation drops AUC (≈ 0.64).  
6. **Labels are noisy** (Score 0, 60-day proxy for fetch time). Dropping zeros raises AUC into the low–mid 0.70s but changes the task.

### Outputs
- `figures/` — primary + secondary plots  
- `results/test_results.csv`, `cv_results.csv`, `summary.json`  
- `Final_Report.pdf`, `Presentation.pdf`
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
out = ROOT / "Question_Quality_Classifier.ipynb"
nbf.write(nb, out)
print("Wrote", out)
