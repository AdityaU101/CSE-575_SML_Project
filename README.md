# CSE-575_SML_Project

**P11 – Learning from Community Question Answering (CQA) data**  
Stack Overflow **question-quality classifier** (ML + NLP), revised to separate **quality** from **post-type** effects.

## Headline result (primary task)

**Population:** normal Q&A questions only (exclude `advice` / `best-practices` / `tooling-recommendation`)  
**Metric:** ROC-AUC (not F1 vs a stratified dummy)

| Model | Features | Test ROC-AUC | Test F1 |
|---|---|---:|---:|
| Logistic Regression | text + structure (no tags) | **0.671** | 0.620 |
| Logistic Regression | all (text + structure + tags) | **0.694** | 0.621 |
| Dummy (all-high) | — | 0.500 | 0.671 |

All-high F1 looks strong only because classes are ~balanced; AUC shows it is uninformative.

## Final deliverables

| File | Description |
|---|---|
| [`Final_Report.pdf`](Final_Report.pdf) | Course final report |
| [`Final_Report.md`](Final_Report.md) | Markdown report |
| [`Presentation.pdf`](Presentation.pdf) | 9-slide presentation |
| [`Question_Quality_Classifier.ipynb`](Question_Quality_Classifier.ipynb) | Narrative notebook (runs pipeline) |
| [`Question_Quality_Classifier.html`](Question_Quality_Classifier.html) | HTML export |
| [`scripts/run_rigorous_pipeline.py`](scripts/run_rigorous_pipeline.py) | Full experiment code |
| [`figures/`](figures/) | Primary + secondary plots |
| [`results/`](results/) | `summary.json`, CV/test tables |

## What the revision fixed

1. Opinion-type posts were ~30% of the filtered data and almost never high-scoring → tags dominated earlier models  
2. Tags are no longer buried inside “structural” features  
3. Model selection uses **CV ROC-AUC** only (no test peeking)  
4. Baselines include **all-high**, not only stratified dummy  
5. Chronological split + drop-Score-0 sensitivity + all-posts confound suite  
6. 60-day age filter; code text used in TF-IDF; no hard-coded user paths  

## Setup / reproduce

```bash
cd CSE-575_SML_Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

MPLBACKEND=Agg python scripts/run_rigorous_pipeline.py
python scripts/build_deliverables.py
# optional: refresh notebook outputs without re-training
SKIP_RERUN=1 jupyter nbconvert --to notebook --execute Question_Quality_Classifier.ipynb \
  --output Question_Quality_Classifier.ipynb
```

Or open the notebook, select the project kernel, and **Run All** (~8–15 minutes).

## Dataset note

`data/stackoverflow_questions_2023plus.csv` is a historical filename. The export is Stack Overflow questions from **2026-04 to 2026-10**. `FavoriteCount` is always 0 in this API export.
