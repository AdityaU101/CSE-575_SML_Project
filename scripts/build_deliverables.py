#!/usr/bin/env python3
"""Build Final_Report.pdf and Presentation.pdf from rigorous results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"


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
        .replace("\u2248", "~")
    )


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "CSE 575 | Question Quality Classifier (revised)", align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def h1(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(20, 40, 80)
        self.multi_cell(0, 8, ascii(text))
        self.ln(1)

    def h3(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, ascii(text))
        self.ln(0.5)

    def body(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 5.2, ascii(text))
        self.ln(1.2)

    def bullet(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 5.2, ascii(f"  -  {text}"))

    def add_fig(self, path: Path, caption: str, w: float = 175):
        if not path.exists():
            self.body(f"[Missing figure: {path.name}]")
            return
        if self.get_y() > 220:
            self.add_page()
        self.set_x(self.l_margin)
        self.image(str(path), w=w)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 4.5, ascii(caption))
        self.ln(2)


def table(pdf: ReportPDF, headers, rows, col_w):
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
        pdf.set_fill_color(248, 248, 248)
        pdf.set_x(pdf.l_margin)
        for val, w in zip(row, col_w):
            pdf.cell(w, 5.5, ascii(val), border=1, fill=(i % 2 == 0), align="C")
        pdf.ln()
    pdf.ln(2)


def build_report():
    summary = json.loads((RES / "summary.json").read_text())
    test = pd.read_csv(RES / "test_results.csv")
    primary = test[test.Suite == "primary_normal_random"]
    real = primary[~primary.Model.str.startswith("Dummy")].copy()
    lr_ts = real[(real.Model == "Logistic Regression") & (real.Features == "text+struct")].iloc[0]
    lr_all = real[(real.Model == "Logistic Regression") & (real.Features == "all")].iloc[0]
    secondary = test[test.Suite == "secondary_all_posts"]
    sec_tags = secondary[(secondary.Model == "Logistic Regression") & (secondary.Features == "tags")].iloc[0]
    sec_all = secondary[(secondary.Model == "Logistic Regression") & (secondary.Features == "all")].iloc[0]
    chrono = test[test.Suite == "primary_normal_chrono"]
    chrono_best = chrono[~chrono.Model.str.startswith("Dummy")].sort_values("ROC-AUC", ascending=False).iloc[0]

    pdf = ReportPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(15, 35, 70)
    pdf.multi_cell(0, 8, ascii("Predicting Stack Overflow Question Quality with ML/NLP"))
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, ascii("CSE 575 - P11 Learning from CQA Data\nInstructor: Yingzhen Yang"))
    pdf.ln(1)
    pdf.h3("Team")
    for line in [
        "Shubham Suryakant Patil - Data Collection",
        "Virinchi Daram - Literature Review & Experimental Design",
        "Sai Teja Pyarasani - Preprocessing & Feature Engineering",
        "Aditya Umesh Upadhyay - Model Development",
        "Sai Suhaas M - Evaluation & Analysis",
    ]:
        pdf.bullet(line)

    pdf.h1("1. Introduction")
    pdf.body(
        "This project predicts Stack Overflow question quality from posting-time text and structure. "
        "Revision: models on all recent posts mostly detect opinion-type questions "
        "(advice / best-practices / tooling-recommendation), not writing quality. "
        "Primary task = normal Q&A only; tags separated from structure; headline metric = ROC-AUC."
    )
    pdf.h3("Headline (primary: normal questions)")
    pdf.bullet(f"LR text+struct (no tags): AUC={lr_ts['ROC-AUC']:.3f}, F1={lr_ts.F1:.3f}")
    pdf.bullet(f"LR all (with tags): AUC={lr_all['ROC-AUC']:.3f}, F1={lr_all.F1:.3f}")
    pdf.bullet("All-high dummy: F1~0.671 but AUC=0.500 when classes are balanced.")

    pdf.h1("2. Problem and data")
    pdf.body(
        f"Export window 2026-04 to 2026-10 (CSV filename is historical). "
        f"Age filter >= {summary['min_age_days']} days from snapshot {summary['snapshot_date']} "
        f"keeps {summary['n_after_age_filter']} questions. "
        f"Opinion high-rate={summary['opinion_high_rate']:.1%}; "
        f"normal high-rate={summary['normal_high_rate']:.1%}. "
        "FavoriteCount is always 0 in this API export."
    )
    pdf.add_fig(FIG / "01_score_distribution.png", "Figure 1. Score distribution and post-type mix.")
    pdf.add_fig(FIG / "02_temporal_drift.png", "Figure 2. Temporal drift.")

    pdf.h1("3. Methodology")
    pdf.bullet("Primary population: normal questions only.")
    pdf.bullet("Label: Score>=1 high vs Score<=0 low; no leakage features.")
    pdf.bullet("Feature sets: text | struct (no tags) | tags | text+struct | all.")
    pdf.bullet("TF-IDF uses title + body prose + extracted code text.")
    pdf.bullet("Select by 5-fold CV ROC-AUC on train only.")
    pdf.bullet("Suites: random primary, chronological, all-posts confound, drop Score==0.")

    pdf.h1("4. Results")
    pdf.h3("4.1 Primary test metrics")
    rows = []
    for _, r in primary.sort_values("ROC-AUC", ascending=False).iterrows():
        name = (
            r.Model.replace("Logistic Regression", "LR")
            .replace("Random Forest", "RF")
            .replace("Naive Bayes", "NB")
            .replace("Dummy (", "D(")[:18]
        )
        rows.append([name, str(r.Features)[:12], f"{r.Accuracy:.3f}", f"{r.F1:.3f}", f"{r['ROC-AUC']:.3f}"])
    table(pdf, ["Model", "Feat", "Acc", "F1", "AUC"], rows[:12], [40, 28, 22, 22, 22])
    pdf.body(
        "CV selects LR|all. Best tag-free content model is LR|text+struct. "
        "Pure struct no longer dominates text once tags/opinion posts are removed."
    )
    pdf.add_fig(FIG / "07_featureset_heatmap_primary.png", "Figure 3. Primary AUC/F1 heatmaps.")
    pdf.add_fig(FIG / "06_roc_primary_textstruct.png", "Figure 4. ROC for text+struct.", w=120)

    pdf.h3("4.2 Bootstrap (LR)")
    boot = summary["bootstrap"]
    brows = []
    for k, v in boot.items():
        brows.append(
            [
                k[:28],
                f"{v['delta_auc_mean']:.3f}",
                f"[{v['delta_auc_ci95'][0]:.3f},{v['delta_auc_ci95'][1]:.3f}]",
                f"{v['delta_f1_mean']:.3f}",
            ]
        )
    table(pdf, ["Contrast", "dAUC", "AUC CI95", "dF1"], brows, [55, 25, 45, 25])

    pdf.h3("4.3 Chronological robustness")
    pdf.body(
        f"Best chrono: {chrono_best.Model} | {chrono_best.Features}: "
        f"AUC={chrono_best['ROC-AUC']:.3f}, F1={chrono_best.F1:.3f}."
    )
    pdf.add_fig(FIG / "11_roc_chrono.png", "Figure 5. Chronological-split ROC.", w=120)

    pdf.h3("4.4 Secondary: all posts (tag confound)")
    pdf.body(
        f"LR tags-alone AUC={sec_tags['ROC-AUC']:.3f}; LR all AUC={sec_all['ROC-AUC']:.3f}. "
        "Reported for transparency, not as the primary quality claim."
    )
    pdf.add_fig(FIG / "12_featureset_heatmap_allposts.png", "Figure 6. All-posts AUC heatmap.")
    pdf.add_fig(FIG / "09_lr_coefficients_primary.png", "Figure 7. LR coefficients (normal questions).", w=140)

    pdf.h1("5. Conclusions")
    pdf.bullet("Normal-question quality is predictable above chance (AUC ~0.67 without tags, ~0.69 with tags).")
    pdf.bullet("Earlier structural>text claims were mostly tags + opinion-type detection.")
    pdf.bullet("Lead with ROC-AUC; separate tags from structure; filter post types.")
    pdf.bullet("Chronological evaluation and Score==0 ambiguity remain limitations.")
    pdf.body("Reproduce: MPLBACKEND=Agg python scripts/run_rigorous_pipeline.py && python scripts/build_deliverables.py")

    out = ROOT / "Final_Report.pdf"
    pdf.output(str(out))
    print("Wrote", out)


class SlidePDF(FPDF):
    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 6, f"{self.page_no()}", align="R")


def slide_title(pdf, title, subtitle=""):
    pdf.add_page()
    pdf.set_fill_color(15, 35, 70)
    pdf.rect(0, 0, 297, 18, "F")
    pdf.set_xy(12, 5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "CSE 575 | P11 CQA Question Quality")
    pdf.set_xy(12, 28)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(15, 35, 70)
    pdf.multi_cell(270, 9, ascii(title))
    if subtitle:
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(270, 7, ascii(subtitle))
    pdf.ln(1)


def bullets(pdf, items):
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(25, 25, 25)
    for item in items:
        pdf.set_x(16)
        pdf.multi_cell(265, 7, ascii(f"-  {item}"))
        pdf.ln(0.8)


def build_presentation():
    summary = json.loads((RES / "summary.json").read_text())
    test = pd.read_csv(RES / "test_results.csv")
    primary = test[test.Suite == "primary_normal_random"]
    real = primary[~primary.Model.str.startswith("Dummy")]
    lr_ts = real[(real.Model == "Logistic Regression") & (real.Features == "text+struct")].iloc[0]
    lr_all = real[(real.Model == "Logistic Regression") & (real.Features == "all")].iloc[0]
    secondary = test[test.Suite == "secondary_all_posts"]
    sec_tags = secondary[(secondary.Model == "Logistic Regression") & (secondary.Features == "tags")].iloc[0]

    pdf = SlidePDF(orientation="L", format="Letter")
    pdf.set_auto_page_break(auto=False)

    slide_title(pdf, "Predicting Stack Overflow Question Quality", "Revised: quality vs post-type confound")
    bullets(
        pdf,
        [
            "Shubham Suryakant Patil - Data Collection",
            "Virinchi Daram - Literature & Design",
            "Sai Teja Pyarasani - Features",
            "Aditya Umesh Upadhyay - Models",
            "Sai Suhaas M - Evaluation",
        ],
    )

    slide_title(pdf, "The confound we fixed")
    bullets(
        pdf,
        [
            f"Opinion-type posts: {summary['n_opinion']} / {summary['n_after_age_filter']}",
            f"High-rate opinion {summary['opinion_high_rate']:.1%} vs normal {summary['normal_high_rate']:.1%}",
            "On all posts, tags-alone nearly matches the full model",
            "Primary task: normal Q&A only; tags separated from struct",
        ],
    )
    if (FIG / "02_temporal_drift.png").exists():
        pdf.image(str(FIG / "02_temporal_drift.png"), x=155, y=55, w=125)

    slide_title(pdf, "Primary protocol")
    bullets(
        pdf,
        [
            "60-day age filter; label Score>=1 vs <=0",
            "Features: text | struct | tags | text+struct | all",
            "Select by CV ROC-AUC (no test peeking)",
            "Baselines include all-high; headline metric = ROC-AUC",
        ],
    )

    slide_title(pdf, "Primary results (normal questions)")
    bullets(
        pdf,
        [
            f"LR text+struct: AUC={lr_ts['ROC-AUC']:.3f}, F1={lr_ts.F1:.3f}",
            f"LR all: AUC={lr_all['ROC-AUC']:.3f}, F1={lr_all.F1:.3f}",
            "All-high dummy F1~0.671 with AUC=0.5",
            "Struct alone no longer beats text once tags are removed",
        ],
    )
    if (FIG / "07_featureset_heatmap_primary.png").exists():
        pdf.image(str(FIG / "07_featureset_heatmap_primary.png"), x=140, y=70, w=145)

    slide_title(pdf, "ROC - content signal without tags")
    if (FIG / "06_roc_primary_textstruct.png").exists():
        pdf.image(str(FIG / "06_roc_primary_textstruct.png"), x=70, y=40, w=150)

    slide_title(pdf, "Robustness checks")
    bullets(
        pdf,
        [
            f"Chronological best AUC ~ {test.loc[test.Suite=='primary_normal_chrono','ROC-AUC'].max():.3f}",
            f"All-posts tags-alone AUC={sec_tags['ROC-AUC']:.3f}",
            "Drop Score==0 raises AUC but changes the task",
        ],
    )
    if (FIG / "11_roc_chrono.png").exists():
        pdf.image(str(FIG / "11_roc_chrono.png"), x=160, y=50, w=120)

    slide_title(pdf, "What the model learns")
    if (FIG / "09_lr_coefficients_primary.png").exists():
        pdf.image(str(FIG / "09_lr_coefficients_primary.png"), x=40, y=35, w=210)

    slide_title(pdf, "Conclusions")
    bullets(
        pdf,
        [
            "Text+structure predict normal-question quality above chance",
            "Earlier high scores were mostly opinion-type detection",
            "Lead with ROC-AUC; separate tags; filter post types",
            "Deliverables: notebook, figures/, results/, Final_Report.pdf",
        ],
    )

    slide_title(pdf, "Thank you", "Questions?")
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(
        0,
        7,
        ascii(
            "Code: Question_Quality_Classifier.ipynb + scripts/run_rigorous_pipeline.py\n"
            "Metrics: results/summary.json\n"
            "Report: Final_Report.pdf"
        ),
    )

    out = ROOT / "Presentation.pdf"
    pdf.output(str(out))
    print("Wrote", out)
    print("Presentation pages: 9")


if __name__ == "__main__":
    build_report()
    build_presentation()
