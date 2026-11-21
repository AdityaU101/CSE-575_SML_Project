# Predicting Stack Overflow Question Quality Using Machine Learning and NLP

**CSE 575 – Statistical Machine Learning** · **P11: Learning from CQA Data**  
**Instructor:** Yingzhen Yang

## Team Members

| Name | Role |
|------|------|
| Shubham Suryakant Patil | Data Collection |
| Virinchi Daram | Literature Review and Experimental Design |
| Sai Teja Pyarasani | Data Preprocessing and Feature Engineering |
| Aditya Umesh Upadhyay | Machine Learning and Model Development |
| Sai Suhaas M | Evaluation and Analysis |

---

## 1. Introduction

Stack Overflow receives a large volume of questions. Automatically estimating quality from posting-time content could support moderation and asker guidance. Among P11 directions (routing, tagging, quality, churn), this project focuses on **question quality assessment**.

**Research question:** Can textual and structural features predict community-judged quality *without confusing quality with post type*?

An important finding of this work is methodological: models trained on all recent posts largely detect Stack Overflow **opinion-type** questions (`advice`, `best-practices`, `tooling-recommendation`) rather than writing quality. We therefore redefine the **primary task** as quality prediction on **normal Q&A questions only**, keep tags as a separate feature family, and treat **ROC-AUC** as the headline metric.

### Headline results (primary: normal questions)

| Model | Features | Test ROC-AUC | Test F1 |
|-------|----------|--------------|---------|
| Logistic Regression | **text+struct** (no tags) | **0.671** | 0.620 |
| Logistic Regression | all (text+struct+tags) | **0.694** | 0.621 |
| Dummy (all-high / majority) | — | 0.500 | 0.671 |

All-high F1 looks competitive only because classes are ~balanced; **AUC shows those baselines are uninformative**. Chronological evaluation drops AUC to ~0.64–0.65.

---

## 2. Problem Description

### 2.1 Label

High if `Score >= 1`, low if `Score <= 0`. Leakage-safe features exclude Score, views, answers, comments, favorites, accepted-answer id, reputation, and ids/dates. `FavoriteCount` is always 0 in this API export and is unused.

### 2.2 Dataset

| Item | Detail |
|------|--------|
| File | `data/stackoverflow_questions_2023plus.csv` (filename is historical) |
| Actual window | 2026-04 to 2026-10 Stack Overflow questions |
| Raw size | 10,000 questions |
| Age filter | ≥ **60 days** before snapshot date 2026-10-27 → **8,077** kept |
| Opinion-type share | 2,461 / 8,077 (**30.5%**); high-rate only **4.5%** |
| Normal Q&A | 5,616; high-rate **50.5%** |

### 2.3 Temporal drift

Opinion-type share falls from ~35–38% (Apr–Jun) to ~10% (Jul–Oct), while the high-quality rate rises. A random split mixes regimes; we also evaluate a chronological 80/20 split.

---

## 3. Methodology

### 3.1 Preprocessing

Deduplicate; drop empty title/body; parse HTML into prose, **code text**, and structure counts; apply a conservative 60-day age filter (limitation: snapshot is the newest post date in the file).

### 3.2 Feature sets (separated)

| Set | Contents |
|-----|----------|
| text | TF-IDF (1–2 grams) on title + body prose + code text |
| struct | lengths, code/link/list/style counts (**no tags**) |
| tags | binary tag indicators |
| text+struct | content/writing signal |
| all | text + struct + tags |

### 3.3 Models and selection

Logistic Regression, Naive Bayes, Random Forest; hyperparameters via **5-fold stratified GridSearchCV maximizing ROC-AUC** on train only. Baselines: majority, stratified, **all-high**.

### 3.4 Suites

1. **Primary:** normal questions, random stratified 80/20  
2. **Chronological:** normal questions, oldest 80% → newest 20%  
3. **Secondary:** all posts (documents tag confound)  
4. **Sensitivity:** normal questions with Score==0 removed (≥1 vs ≤−1)

---

## 4. Results and Analysis

### 4.1 Primary (normal questions)

CV selects **Logistic Regression | all** (test AUC **0.694**, F1 0.621).  
The best **tag-free** model is **LR | text+struct** (AUC **0.671**, F1 0.620).

On normal questions, tags help modestly but no longer dominate. Pure `struct` (AUC ~0.63) is weaker than text (~0.65). The old claim that “structural features dominate text” was driven by **tags bundled into the structural set** plus opinion posts.

Bootstrap (LR): `text+struct − text` ΔAUC ≈ +0.022, 95% CI [0.002, 0.043]; `all − text+struct` ΔAUC ≈ +0.022, CI [0.001, 0.043].

### 4.2 Fair baselines

With ~50% high-quality normal questions, **all-high F1 = 0.671** beats most model F1s while AUC = 0.5. **ROC-AUC is the fair headline.**

### 4.3 Chronological split

Best chrono test AUC ≈ **0.648** (`text+struct`). Performance drops under time shift.

### 4.4 Secondary: all posts (confound)

| Features (LR) | Test ROC-AUC |
|---------------|--------------|
| tags alone | **0.806** |
| text+struct | 0.748 |
| all | **0.824** |

Tags alone nearly match the full model — the classic post-type confound.

### 4.5 Sensitivity (drop Score 0)

Removing unvoted questions raises LR `all` AUC to **0.726**. Labels remain a noisy proxy for “quality.”

---

## 5. Conclusions and Future Work

1. Posting-time text + structure predict normal-question quality above chance (**AUC ≈ 0.67** without tags; ≈ 0.69 with tags).  
2. Without separating tags and filtering opinion-types, models mostly learn **post type**, not quality.  
3. Prefer **ROC-AUC** over F1-vs-stratified-dummy for reporting.  
4. Random splits overstate stability; chronological AUC falls to ~0.64.  
5. Score-based labels are noisy (especially Score 0).

**Future work:** ordinal/closure labels, transformer embeddings, asker history with leakage control, multi-task opinion-vs-quality models.

## References

1. Anderson et al., KDD 2012.  
2. Li et al., 2010 — G-Finder.  
3. Yao et al., KDD 2014.  
4. Pudipeddi et al., WWW 2014.  
5. Stack Exchange API.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MPLBACKEND=Agg python scripts/run_rigorous_pipeline.py
python scripts/build_deliverables.py
```
