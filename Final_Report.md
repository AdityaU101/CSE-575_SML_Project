# Predicting Stack Overflow Question Quality Using Machine Learning and Natural Language Processing

**CSE 575 – Statistical Machine Learning**  
**Project Area:** P11 – Learning from Community Question Answering (CQA) Data  
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

Community Question Answering (CQA) platforms such as Stack Overflow let users ask technical questions and receive answers from other community members. Large volumes of questions are posted daily, and not all of them are equally useful to the community. Automatically estimating question quality can help with moderation, ranking, and guidance for askers.

This project focuses on one of the research directions listed under P11: **identifying good versus bad questions**. We build and evaluate machine learning classifiers that predict whether a Stack Overflow question is high- or low-quality using only information available at posting time (title, body text, tags, and structural cues). Community vote scores define labels; score and other engagement signals are **not** used as features, to avoid target leakage.

**Research question:** *Can textual and structural features of a Stack Overflow question accurately predict its community-judged quality?*

**Main findings (held-out test set):**
- Best model: **Logistic Regression** with **combined** (text + structural) features  
- **F1 = 0.669**, **ROC-AUC = 0.795**, **Accuracy = 0.691**  
- Clear lift over a stratified dummy baseline (**F1 = 0.364**, +0.306)  
- Structural/tag features alone already beat pure TF-IDF text; combining both is best overall  

Source code: `Question_Quality_Classifier.ipynb`. Figures: `figures/`. Metrics: `results/`.

---

## 2. Problem Description

### 2.1 Background

The P11 project description identifies several CQA problems (question routing, Q&A tagging, quality assessment, and user churn). Following our proposal, we concentrate on **quality assessment of questions**, using Stack Overflow data with question text and voting information.

Related work informing the design includes Anderson et al. (KDD 2012) on value in Stack Overflow activity, Yao et al. (KDD 2014) on long-term impact of CQA posts, and related CQA routing/churn studies listed in the proposal.

### 2.2 Formal task

Given a question \(x\) with title, HTML body, and tags, predict label

\[
y = \begin{cases}
1 \text{ (high quality)} & \text{if Score} \ge 1 \\
0 \text{ (low quality)} & \text{if Score} \le 0
\end{cases}
\]

**Leakage-safe feature policy:** `Score`, `ViewCount`, `AnswerCount`, `CommentCount`, `FavoriteCount`, `AcceptedAnswerId`, `LastEditDate`, asker reputation, and IDs/dates are excluded from predictors.

### 2.3 Dataset

| Item | Detail |
|------|--------|
| Source | Stack Overflow questions via Stack Exchange API |
| File | `data/stackoverflow_questions_2023plus.csv` |
| Raw size | 10,000 questions, 14 columns |
| After cleaning | 9,011 questions (removed 989 posts younger than 30 days so scores can stabilize) |
| Train / test | 7,208 / 1,803 (stratified 80/20) |
| Class balance | ~37.8% high quality, ~62.2% low quality |

Attributes used: title, body (HTML), tags, score (label only), creation date (age filter only).

---

## 3. Methodology

### 3.1 Preprocessing

1. Drop duplicate / empty title–body rows  
2. Drop questions younger than 30 days relative to the data snapshot  
3. Parse HTML with BeautifulSoup → prose text, code text, and structure counts (code blocks, links, lists, paragraphs, etc.)

### 3.2 Features

**Textual:** TF-IDF (unigrams + bigrams) on title (max 5,000 features) and body prose (max 20,000 features).

**Structural / style:** title/body length, question-mark and wh-word title cues, code presence and code ratio, links, images, paragraphs, lists, tags count, sentence length, uppercase ratio, exclamation count, error-keyword mentions, plus binary tag indicators.

**Feature sets compared:** (1) text only, (2) structural only, (3) combined.

### 3.3 Models

| Model | Notes |
|-------|--------|
| Logistic Regression | Balanced class weights; \(C\) tuned |
| Naive Bayes | MultinomialNB for text/combined; GaussianNB for structural |
| Random Forest | 200 trees, balanced subsample; depth / leaf size tuned |
| Dummy baselines | Most-frequent and stratified |

Training uses **5-fold stratified GridSearchCV** on the training set (optimize F1). Final metrics are reported on the held-out test set only.

### 3.4 Evaluation metrics

Accuracy, Precision, Recall, F1 (positive = high quality), Macro F1, ROC-AUC, confusion matrices, and ROC curves.

---

## 4. Results and Analysis

### 4.1 Cross-validation (train)

| Model | Features | CV F1 (mean) | Best params (abbrev.) |
|-------|----------|--------------|------------------------|
| Logistic Regression | text | 0.551 | C=0.01 |
| Naive Bayes | text | 0.520 | α=0.1, no prior |
| Random Forest | text | 0.520 | depth=20, leaf=5 |
| Logistic Regression | structural | 0.665 | C=0.1 |
| Naive Bayes | structural | 0.616 | var_smoothing=0.1 |
| Random Forest | structural | 0.649 | depth=20, leaf=5 |
| **Logistic Regression** | **combined** | **0.669** | **C=0.1** |
| Naive Bayes | combined | 0.601 | α=0.1, no prior |
| Random Forest | combined | 0.630 | depth=20, leaf=5 |

### 4.2 Held-out test performance

| Model | Features | Acc | Prec | Rec | F1 | ROC-AUC |
|-------|----------|-----|------|-----|-----|---------|
| Dummy (most frequent) | – | 0.622 | 0.000 | 0.000 | 0.000 | 0.500 |
| Dummy (stratified) | – | 0.516 | 0.362 | 0.365 | 0.364 | 0.487 |
| LR | text | 0.624 | 0.503 | 0.562 | 0.530 | 0.674 |
| NB | text | 0.615 | 0.491 | 0.523 | 0.507 | 0.649 |
| RF | text | 0.642 | 0.530 | 0.479 | 0.503 | 0.667 |
| LR | structural | 0.676 | 0.547 | 0.837 | 0.662 | 0.791 |
| NB | structural | 0.659 | 0.539 | 0.685 | 0.603 | 0.721 |
| RF | structural | 0.673 | 0.546 | 0.801 | 0.649 | 0.765 |
| **LR** | **combined** | **0.691** | **0.562** | **0.827** | **0.669** | **0.795** |
| NB | combined | 0.664 | 0.551 | 0.610 | 0.579 | 0.732 |
| RF | combined | 0.654 | 0.532 | 0.710 | 0.608 | 0.742 |

**Best configuration:** Logistic Regression + combined features  
- Low quality: precision 0.853, recall 0.608, F1 0.710  
- High quality: precision 0.562, recall 0.827, F1 0.669  
- Macro F1 = 0.690  

### 4.3 Comparative observations

1. **Structural features dominate text.** Pure TF-IDF tops out near F1 ≈ 0.53; structural reaches ≈ 0.66.  
2. **Combining text + structure is best** for Logistic Regression (F1 0.669, AUC 0.795).  
3. **Logistic Regression** is strongest overall and remains interpretable via coefficients.  
4. Random Forest is competitive on structural features but does not beat LR on combined.  
5. All real models beat the stratified dummy F1 (0.364) by a large margin.

### 4.4 Interpretation

Coefficient analysis for LR (combined) shows:
- Strong **negative** associations for meta / advice-style tags (`advice`, `best-practices`, `tooling-recommendation`), consistent with Stack Overflow’s preference against opinion/advice questions.  
- Strong **positive** associations for many concrete technology tags (e.g., `c`, `r`, `powershell`, `java`).  
- Structural cues (code presence, formatting) contribute alongside tags.

Error analysis shows confident false positives are often well-formatted but low-voted posts (quality ≠ formatting alone), and false negatives include high-score posts with sparse structural signals.

### 4.5 Figures

See `figures/`:
- `01_score_distribution.png`, `02_class_balance.png` – EDA / labels  
- `03_structural_by_class.png` – feature differences by class  
- `04_test_metrics.png`, `05_confusion_matrices.png`, `06_roc_combined.png` – evaluation  
- `07_featureset_heatmap.png` – text vs structural vs combined  
- `08_lr_coefficients.png`, `09_lr_structural.png` – interpretation  

---

## 5. Conclusions and Future Work

### 5.1 Conclusions

1. Textual and structural features **can** predict Stack Overflow question quality substantially better than chance.  
2. **Structural + tag features** carry most of the signal; TF-IDF alone is weaker.  
3. **Logistic Regression (combined)** is the recommended model for this dataset: best F1/AUC and clear interpretability.  
4. Quality also depends on community dynamics not captured by posting-time text alone, so performance remains moderate (~0.67 F1 / ~0.80 AUC).  
5. Strict leakage control is essential; engagement and score must not enter the feature set.

### 5.2 Limitations

- Binary threshold on Score is a proxy, not ground-truth “goodness.”  
- Subset of ~9k recent questions; may not generalize to older dumps or other sites.  
- Asker history / answer quality / topic popularity not modeled (by design for leakage safety).  
- Class imbalance (~38% high) affects precision on the high-quality class.

### 5.3 Future work

- Multi-class or ordinal quality (closed reasons, accepted answers).  
- Transformer embeddings (e.g., CodeBERT) instead of TF-IDF.  
- Calibrated ranking for moderation queues.  
- Extend to answer quality or question routing (other P11 directions).

---

## References

1. A. Anderson et al., “Discovering value from community activity on focused question answering sites: A case study of Stack Overflow,” KDD, 2012.  
2. W. Li et al., “G-Finder: Routing programming questions closer to the experts,” ACM SIGPLAN Notices, 2010.  
3. Y. Yao et al., “Predicting long-term impact of CQA posts: A comprehensive viewpoint,” KDD, 2014.  
4. J. S. Pudipeddi et al., “User churn in focused question answering sites,” WWW, 2014.  
5. Stack Exchange Data Explorer / Stack Exchange API.  

---

## Appendix: How to reproduce

```bash
cd CSE-575_SML_Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute Question_Quality_Classifier.ipynb \
  --output Question_Quality_Classifier.ipynb --ExecutePreprocessor.timeout=3600
```

Outputs are written to `figures/` and `results/cv_results.csv`, `results/test_results.csv`.
