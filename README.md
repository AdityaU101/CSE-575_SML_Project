# CSE-575_SML_Project

**P11 – Learning from Community Question Answering (CQA) data**  
Stack Overflow **question-quality classifier** (ML + NLP).

## Final deliverables

| File | Description |
|---|---|
| [`Final_Report.pdf`](Final_Report.pdf) | Course final report (intro → results → conclusions) |
| [`Final_Report.md`](Final_Report.md) | Same report in Markdown |
| [`Presentation.pdf`](Presentation.pdf) | ~8-slide presentation deck |
| [`Question_Quality_Classifier.ipynb`](Question_Quality_Classifier.ipynb) | Executed end-to-end pipeline |
| [`Question_Quality_Classifier.html`](Question_Quality_Classifier.html) | HTML export of the notebook |
| [`figures/`](figures/) | All experiment plots (EDA, metrics, ROC, coefficients) |
| [`results/cv_results.csv`](results/cv_results.csv) | Cross-validation scores |
| [`results/test_results.csv`](results/test_results.csv) | Held-out test metrics |

### Headline result

**Logistic Regression + combined features**  
F1 **0.669** · ROC-AUC **0.795** · Accuracy **0.691**  
(vs stratified dummy F1 **0.364**)

## Dataset

| File | Description |
|---|---|
| `data/stackoverflow_questions_2023plus.csv` | Full dataset (10k questions, HTML bodies) |
| `data/stackoverflow_questions_preview_100.csv` | Small preview for browsing in the editor |
| `data/stackoverflow_questions_metadata.csv` | All rows without `Body` |

## Notebook pipeline

1. Load & explore  
2. Preprocess / HTML parse  
3. Label quality from `Score` (no leakage)  
4. TF-IDF + structural features  
5. Logistic Regression, Naive Bayes, Random Forest  
6. Test-set evaluation + interpretation  

## Setup / reproduce

```bash
cd CSE-575_SML_Project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=cse575-sml --display-name="CSE575 SML (.venv)"
```

Then open the notebook, select the **CSE575 SML (.venv)** kernel, and **Run All**.

Or from the CLI:

```bash
MPLBACKEND=Agg jupyter nbconvert --to notebook --execute Question_Quality_Classifier.ipynb \
  --output Question_Quality_Classifier.ipynb --ExecutePreprocessor.timeout=3600
python scripts/build_deliverables.py   # rebuild Final_Report.pdf + Presentation.pdf
```

Outputs are written to `figures/` and `results/`.
