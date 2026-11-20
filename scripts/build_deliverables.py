#!/usr/bin/env python3
"""Build Final_Report.pdf and Presentation.pdf from results + figures."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
OUT = ROOT


def ascii(s) -> str:
    return (
        str(s)
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2265", ">=")
        .replace("\u2264", "<=")
        .replace("\u00b7", "|")
    )


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "CSE 575 | Stack Overflow Question Quality Classifier", align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def h1(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 40, 80)
        self.multi_cell(0, 8, ascii(text))
        self.ln(2)

    def h2(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 60, 100)
        self.multi_cell(0, 7, ascii(text))
        self.ln(1)

    def h3(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, ascii(text))
        self.ln(0.5)

    def body(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 5.2, ascii(text))
        self.ln(1.5)

    def bullet(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 5.2, ascii(f"  -  {text}"))

    def add_fig(self, path: Path, caption: str, w: float = 180):
        if not path.exists():
            self.body(f"[Missing figure: {path.name}]")
            return
        self.ln(1)
        # keep some room for caption
        if self.get_y() > 230:
            self.add_page()
        self.set_x(self.l_margin)
        self.image(str(path), w=w)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 4.5, caption)
        self.ln(2)


def table(pdf: ReportPDF, headers: list[str], rows: list[list[str]], col_w: list[float]):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(230, 236, 245)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 6, ascii(h), border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 7.5)
    for i, row in enumerate(rows):
        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "B", 8)
            for h, w in zip(headers, col_w):
                pdf.cell(w, 6, ascii(h), border=1, fill=True, align="C")
            pdf.ln()
            pdf.set_font("Helvetica", "", 7.5)
        fill = i % 2 == 0
        pdf.set_fill_color(248, 248, 248)
        pdf.set_x(pdf.l_margin)
        for val, w in zip(row, col_w):
            pdf.cell(w, 5.5, ascii(val), border=1, fill=fill, align="C")
        pdf.ln()
    pdf.ln(2)


def build_report():
    test = pd.read_csv(RES / "test_results.csv")
    cv = pd.read_csv(RES / "cv_results.csv")
    best = test[~test.Model.str.startswith("Dummy")].sort_values("F1", ascending=False).iloc[0]
    dummy = test.loc[test.Model == "Dummy (stratified)"].iloc[0]

    pdf = ReportPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title page block
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 35, 70)
    pdf.multi_cell(0, 9, "Predicting Stack Overflow Question Quality Using Machine Learning and NLP")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(
        0,
        6,
        "CSE 575 - Statistical Machine Learning\n"
        "Project Area: P11 - Learning from Community Question Answering (CQA) Data\n"
        "Instructor: Yingzhen Yang",
    )
    pdf.ln(2)
    pdf.h3("Team Members and Roles")
    for line in [
        "Shubham Suryakant Patil - Data Collection",
        "Virinchi Daram - Literature Review and Experimental Design",
        "Sai Teja Pyarasani - Data Preprocessing and Feature Engineering",
        "Aditya Umesh Upadhyay - Machine Learning and Model Development",
        "Sai Suhaas M - Evaluation and Analysis",
    ]:
        pdf.bullet(line)
    pdf.ln(2)

    pdf.h1("1. Introduction")
    pdf.body(
        "Community Question Answering (CQA) platforms such as Stack Overflow receive large "
        "volumes of questions daily. Automatically estimating question quality can support "
        "moderation, ranking, and asker guidance. Among the P11 directions (routing, tagging, "
        "quality assessment, churn), this project focuses on quality assessment of questions."
    )
    pdf.body(
        "Research question: Can textual and structural features of a Stack Overflow question "
        "be used to accurately predict its community-judged quality?"
    )
    pdf.h3("Headline results (held-out test set)")
    pdf.bullet(
        f"Best model: {best.Model} with {best.Features} features"
    )
    pdf.bullet(
        f"F1={best.F1:.3f}, ROC-AUC={best['ROC-AUC']:.3f}, Accuracy={best.Accuracy:.3f}"
    )
    pdf.bullet(
        f"Lift vs stratified dummy F1 ({dummy.F1:.3f}): +{best.F1 - dummy.F1:.3f}"
    )
    pdf.bullet("Structural/tag features outperform pure TF-IDF; combined is best overall.")

    pdf.h1("2. Problem Description")
    pdf.body(
        "Stack Overflow does not provide explicit good/bad labels. We define high quality as "
        "Score >= 1 and low quality as Score <= 0. Score and post-hoc engagement signals are "
        "used only for labeling, never as predictive features (no target leakage)."
    )
    pdf.h3("Dataset")
    pdf.bullet("Source: Stack Overflow questions (Stack Exchange API)")
    pdf.bullet("Raw: 10,000 questions; after age filter (>=30 days): 9,011")
    pdf.bullet("Train/test: 7,208 / 1,803 stratified (80/20); ~37.8% high-quality")
    pdf.bullet("Inputs: title, HTML body, tags; label from Score")
    pdf.add_fig(FIG / "01_score_distribution.png", "Figure 1. Score distribution and class balance (EDA).", w=175)
    pdf.add_fig(FIG / "02_class_balance.png", "Figure 2. Class balance after labeling.", w=120)

    pdf.add_page()
    pdf.h1("3. Methodology")
    pdf.h3("Preprocessing")
    pdf.bullet("Deduplicate / drop empty titles and bodies")
    pdf.bullet("Drop questions younger than 30 days (unstable scores)")
    pdf.bullet("Parse HTML to prose, code text, and structure counts")
    pdf.h3("Features")
    pdf.bullet("Text: TF-IDF unigrams+bigrams on title and body")
    pdf.bullet("Structural: lengths, code/links/lists, style cues, tag indicators")
    pdf.bullet("Compared feature sets: text | structural | combined")
    pdf.h3("Models and protocol")
    pdf.bullet("Logistic Regression, Naive Bayes, Random Forest (+ dummy baselines)")
    pdf.bullet("5-fold stratified GridSearchCV on train (optimize F1); evaluate on held-out test")
    pdf.bullet("Metrics: Accuracy, Precision, Recall, F1, Macro F1, ROC-AUC, confusion matrices")
    pdf.add_fig(FIG / "03_structural_by_class.png", "Figure 3. Structural features by quality class.", w=170)

    pdf.h1("4. Results and Analysis")
    pdf.h3("Cross-validation F1 (training)")
    cv_rows = []
    for _, r in cv.iterrows():
        cv_rows.append(
            [
                r.Model.replace("Logistic Regression", "LR").replace("Random Forest", "RF").replace("Naive Bayes", "NB"),
                r.Features,
                f"{r['CV F1 (mean)']:.3f}",
                f"{r['CV F1 (std)']:.3f}",
            ]
        )
    table(pdf, ["Model", "Features", "CV F1", "Std"], cv_rows, [45, 35, 25, 25])

    pdf.h3("Held-out test metrics")
    test_rows = []
    for _, r in test.iterrows():
        name = r.Model.replace("Logistic Regression", "LR").replace("Random Forest", "RF").replace("Naive Bayes", "NB")
        test_rows.append(
            [
                name[:22],
                str(r.Features)[:10],
                f"{r.Accuracy:.3f}",
                f"{r.Precision:.3f}",
                f"{r.Recall:.3f}",
                f"{r.F1:.3f}",
                f"{r['ROC-AUC']:.3f}",
            ]
        )
    table(
        pdf,
        ["Model", "Feat", "Acc", "Prec", "Rec", "F1", "AUC"],
        test_rows,
        [42, 24, 18, 18, 18, 18, 18],
    )

    pdf.body(
        f"Best configuration is {best.Model} with {best.Features} features "
        f"(F1={best.F1:.3f}, ROC-AUC={best['ROC-AUC']:.3f}). Structural features alone "
        "already outperform text-only models; combining both yields a small further gain for LR."
    )
    pdf.add_fig(FIG / "04_test_metrics.png", "Figure 4. Test metrics by model and feature set.", w=180)
    pdf.add_fig(FIG / "07_featureset_heatmap.png", "Figure 5. F1 and ROC-AUC heatmaps across feature sets.", w=180)
    pdf.add_fig(FIG / "05_confusion_matrices.png", "Figure 6. Confusion matrices for all model x feature-set pairs.", w=170)
    pdf.add_fig(FIG / "06_roc_combined.png", "Figure 7. ROC curves for combined-feature models.", w=120)

    pdf.h3("Interpretation")
    pdf.body(
        "LR coefficients show strong negative weights for advice / best-practices style tags "
        "(often discouraged on Stack Overflow) and positive weights for concrete technology tags. "
        "This aligns with community norms: specific, answerable technical questions tend to be "
        "valued more than opinion-seeking posts."
    )
    pdf.add_fig(FIG / "08_lr_coefficients.png", "Figure 8. Top LR (combined) coefficients.", w=150)
    pdf.add_fig(FIG / "09_lr_structural.png", "Figure 9. LR structural feature coefficients.", w=140)

    pdf.h1("5. Conclusions and Future Work")
    pdf.bullet("Textual + structural features predict question quality well above chance.")
    pdf.bullet("Structural/tag features dominate; TF-IDF alone is weaker.")
    pdf.bullet("Logistic Regression (combined) is the preferred model: best F1/AUC and interpretable.")
    pdf.bullet("Limitations: Score proxy label, ~9k subset, no asker history / transformers.")
    pdf.bullet("Future: ordinal quality, CodeBERT embeddings, answer quality, routing (other P11 tasks).")
    pdf.ln(2)
    pdf.h3("References")
    for ref in [
        "Anderson et al., KDD 2012 - Discovering value from community activity on Stack Overflow.",
        "Li et al., 2010 - G-Finder: routing programming questions closer to the experts.",
        "Yao et al., KDD 2014 - Predicting long-term impact of CQA posts.",
        "Pudipeddi et al., WWW 2014 - User churn in focused question answering sites.",
        "Stack Exchange API / Data Explorer.",
    ]:
        pdf.bullet(ref)

    pdf.ln(2)
    pdf.h3("Reproducibility")
    pdf.body(
        "Run Question_Quality_Classifier.ipynb with the project virtualenv "
        "(pip install -r requirements.txt). Outputs: figures/*.png, results/cv_results.csv, "
        "results/test_results.csv."
    )

    out = OUT / "Final_Report.pdf"
    pdf.output(str(out))
    print(f"Wrote {out}")
    return out


class SlidePDF(FPDF):
    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 6, f"{self.page_no()}", align="R")


def slide_title(pdf: SlidePDF, title: str, subtitle: str = ""):
    pdf.add_page()
    pdf.set_fill_color(15, 35, 70)
    pdf.rect(0, 0, 297, 20, "F")
    pdf.set_xy(12, 6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "CSE 575 | P11 Learning from CQA Data")
    pdf.set_xy(12, 32)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(15, 35, 70)
    pdf.multi_cell(270, 10, ascii(title))
    if subtitle:
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(270, 7, ascii(subtitle))
    pdf.ln(2)


def slide_bullets(pdf: SlidePDF, items: list[str]):
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(25, 25, 25)
    for item in items:
        pdf.set_x(16)
        pdf.multi_cell(265, 7, ascii(f"-  {item}"))
        pdf.ln(1)


def build_presentation():
    test = pd.read_csv(RES / "test_results.csv")
    best = test[~test.Model.str.startswith("Dummy")].sort_values("F1", ascending=False).iloc[0]
    dummy = test.loc[test.Model == "Dummy (stratified)"].iloc[0]

    pdf = SlidePDF(orientation="L", format="Letter")
    pdf.set_auto_page_break(auto=False)

    slide_title(
        pdf,
        "Predicting Stack Overflow Question Quality",
        "Machine Learning + NLP  |  Team project final presentation",
    )
    slide_bullets(
        pdf,
        [
            "Shubham Suryakant Patil - Data Collection",
            "Virinchi Daram - Literature Review & Experimental Design",
            "Sai Teja Pyarasani - Preprocessing & Feature Engineering",
            "Aditya Umesh Upadhyay - Model Development",
            "Sai Suhaas M - Evaluation & Analysis",
        ],
    )

    slide_title(pdf, "Motivation & Problem", "P11 focus: quality assessment (not routing / churn)")
    slide_bullets(
        pdf,
        [
            "CQA sites need ways to identify good vs bad questions at scale",
            "Research question: can text + structure predict community score?",
            "Label: High if Score >= 1, Low if Score <= 0",
            "No leakage: score / views / answers / reputation excluded from features",
        ],
    )

    slide_title(pdf, "Dataset & Pipeline")
    slide_bullets(
        pdf,
        [
            "10,000 Stack Overflow questions -> 9,011 after 30-day age filter",
            "Stratified 80/20 split: 7,208 train / 1,803 test (~38% high quality)",
            "Features: TF-IDF title/body + structural counts + tags",
            "Models: Logistic Regression, Naive Bayes, Random Forest",
            "Tune with 5-fold CV on train; report held-out test metrics",
        ],
    )
    if (FIG / "02_class_balance.png").exists():
        pdf.image(str(FIG / "02_class_balance.png"), x=180, y=55, w=100)

    slide_title(pdf, "Key Results", "Held-out test set")
    slide_bullets(
        pdf,
        [
            f"Best: {best.Model} + {best.Features}",
            f"F1 = {best.F1:.3f}   |   ROC-AUC = {best['ROC-AUC']:.3f}   |   Acc = {best.Accuracy:.3f}",
            f"Stratified dummy F1 = {dummy.F1:.3f}  (gain +{best.F1 - dummy.F1:.3f})",
            "Structural features >> text-only TF-IDF",
            "Combined features give the best overall LR performance",
        ],
    )
    if (FIG / "07_featureset_heatmap.png").exists():
        pdf.image(str(FIG / "07_featureset_heatmap.png"), x=150, y=70, w=135)

    slide_title(pdf, "Model Comparison")
    if (FIG / "04_test_metrics.png").exists():
        pdf.image(str(FIG / "04_test_metrics.png"), x=15, y=40, w=260)
    if (FIG / "06_roc_combined.png").exists():
        slide_title(pdf, "ROC - Combined Features")
        pdf.image(str(FIG / "06_roc_combined.png"), x=70, y=40, w=150)

    slide_title(pdf, "What the model learns")
    slide_bullets(
        pdf,
        [
            "Negative: advice / best-practices / tooling-recommendation tags",
            "Positive: concrete technology tags (e.g., c, java, powershell)",
            "Matches Stack Overflow norms: specific answerable questions preferred",
            "Formatting helps, but community dynamics still limit ceiling performance",
        ],
    )
    if (FIG / "08_lr_coefficients.png").exists():
        pdf.image(str(FIG / "08_lr_coefficients.png"), x=155, y=45, w=125)

    slide_title(pdf, "Conclusions & Future Work")
    slide_bullets(
        pdf,
        [
            "Yes - posting-time text/structure predict quality above chance",
            "Recommend Logistic Regression with combined features",
            "Limitations: Score proxy, subset size, no transformer embeddings",
            "Next: ordinal labels, CodeBERT, answer quality / routing extensions",
            "Deliverables: notebook, figures/, results/, Final_Report.pdf",
        ],
    )

    slide_title(pdf, "Thank you", "Questions?")
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(
        0,
        7,
        "Code: Question_Quality_Classifier.ipynb\n"
        "Metrics: results/test_results.csv\n"
        "Report: Final_Report.pdf",
    )

    out = OUT / "Presentation.pdf"
    pdf.output(str(out))
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    build_report()
    build_presentation()
