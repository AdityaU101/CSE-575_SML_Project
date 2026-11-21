#!/usr/bin/env python3
"""Rigorous Stack Overflow question-quality experiments.

Primary task: predict quality on *normal* (non-opinion) questions, with
tags separated from structural features. Secondary analyses document the
opinion-tag confound, chronological shift, and bootstrap uncertainty.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
sns.set_theme(style="whitegrid", context="notebook")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "stackoverflow_questions_2023plus.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MIN_AGE_DAYS = 60  # conservative: scores need time to stabilize
HIGH_Q_THRESHOLD = 1
OPINION_TAGS = {"advice", "best-practices", "tooling-recommendation"}
# Documented collection window for this export (filename is historical).
DATA_NOTE = "Export covers Stack Overflow questions from 2026-04 to 2026-10 (filename is historical)."

np.random.seed(RANDOM_STATE)


def savefig(name: str):
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def split_tags(s: str) -> list[str]:
    return [t for t in str(s).split("|") if t]


def is_opinion(tags: list[str]) -> bool:
    return bool(set(tags) & OPINION_TAGS)


WS = re.compile(r"\s+")
WH_START = re.compile(
    r"^\s*(how|what|why|when|where|which|who|is|are|can|could|should|does|do|did|will|would)\b",
    re.I,
)
ERROR_RE = re.compile(
    r"\b(?:error|exception|traceback|stack ?trace|fails?|failed|crash(?:es|ed)?|warning)\b",
    re.I,
)
SENT_SPLIT = re.compile(r"[.!?]+\s")


def parse_body(html: str) -> dict:
    soup = BeautifulSoup(html if isinstance(html, str) else "", "html.parser")
    pre_blocks = soup.find_all("pre")
    code_text = "\n".join(p.get_text("\n") for p in pre_blocks)
    for p in pre_blocks:
        p.decompose()
    inline = soup.find_all("code")
    inline_text = " ".join(c.get_text(" ") for c in inline)
    for c in inline:
        c.decompose()
    body_text = WS.sub(" ", soup.get_text(" ")).strip()
    # re-parse for structure counts on original
    soup2 = BeautifulSoup(html if isinstance(html, str) else "", "html.parser")
    return {
        "body_text": body_text,
        "code_text": code_text,
        "inline_code_text": inline_text,
        "n_code_blocks": len(soup2.find_all("pre")),
        "n_code_lines": sum(len(p.get_text().splitlines()) for p in soup2.find_all("pre")),
        "n_inline_code": len(soup2.find_all("code")),
        "n_links": len(soup2.find_all("a")),
        "n_images": len(soup2.find_all("img")),
        "n_paragraphs": len(soup2.find_all("p")),
        "n_list_items": len(soup2.find_all("li")),
        "n_blockquotes": len(soup2.find_all("blockquote")),
        "n_headers": len(soup2.find_all(re.compile(r"^h[1-6]$"))),
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tags_list"] = out.Tags.apply(split_tags)
    out["tags_str"] = out.tags_list.apply(" ".join)
    out["is_opinion"] = out.tags_list.apply(is_opinion)
    out["title_chars"] = out.Title.str.len()
    out["title_words"] = out.Title.str.split().str.len()
    out["title_has_qmark"] = out.Title.str.contains(r"\?").astype(int)
    out["title_starts_wh"] = out.Title.str.match(WH_START).astype(int)
    out["body_chars"] = out.body_text.str.len()
    out["body_words"] = out.body_text.str.split().str.len().fillna(0)
    out["code_chars"] = out.code_text.str.len().fillna(0)
    out["code_words"] = out.code_text.str.split().str.len().fillna(0)
    out["has_code"] = (out.n_code_blocks > 0).astype(int)
    out["code_ratio"] = out.code_chars / (out.code_chars + out.body_chars + 1)
    out["n_tags"] = out.tags_list.str.len()
    out["avg_sentence_len"] = out.body_text.apply(
        lambda t: len(t.split()) / max(1, len(SENT_SPLIT.split(t)))
    )
    out["upper_ratio"] = out.body_text.apply(
        lambda t: sum(c.isupper() for c in t) / max(1, sum(c.isalpha() for c in t))
    )
    out["n_exclaim"] = out.body_text.str.count("!")
    out["mentions_error"] = out.body_text.str.contains(ERROR_RE).astype(int)
    # Combine prose + code into one text field used by TF-IDF (uses extracted code_text).
    out["full_text"] = (
        out.Title.fillna("")
        + "\n"
        + out.body_text.fillna("")
        + "\n"
        + out.code_text.fillna("")
    )
    out["label"] = (out.Score >= HIGH_Q_THRESHOLD).astype(int)
    return out


STRUCT_FEATURES = [
    "title_chars",
    "title_words",
    "title_has_qmark",
    "title_starts_wh",
    "body_chars",
    "body_words",
    "code_chars",
    "code_words",
    "n_code_blocks",
    "n_code_lines",
    "n_inline_code",
    "has_code",
    "code_ratio",
    "n_links",
    "n_images",
    "n_paragraphs",
    "n_list_items",
    "n_blockquotes",
    "n_headers",
    "n_tags",
    "avg_sentence_len",
    "upper_ratio",
    "n_exclaim",
    "mentions_error",
]


def make_features(feature_set: str, numeric_scaler="standard", dense=False):
    parts = []
    if feature_set in ("text", "text+struct", "all"):
        parts.append(
            (
                "text_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    sublinear_tf=True,
                    stop_words="english",
                    max_features=25_000,
                ),
                "full_text",
            )
        )
    if feature_set in ("struct", "text+struct", "all"):
        scaler = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(clip=True),
            None: "passthrough",
        }[numeric_scaler]
        parts.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", scaler),
                    ]
                ),
                STRUCT_FEATURES,
            )
        )
    if feature_set in ("tags", "all"):
        parts.append(
            (
                "tags",
                CountVectorizer(
                    token_pattern=r"\S+", binary=True, min_df=20, lowercase=False
                ),
                "tags_str",
            )
        )
    return ColumnTransformer(parts, sparse_threshold=0.0 if dense else 0.3)


def get_experiments(feature_sets):
    exps = {}
    for fs in feature_sets:
        exps[("Logistic Regression", fs)] = (
            Pipeline(
                [
                    ("features", make_features(fs, "standard")),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=4000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.01, 0.1, 0.5, 1.0, 5.0]},
        )
        if fs == "struct":
            exps[("Naive Bayes", fs)] = (
                Pipeline(
                    [
                        ("features", make_features(fs, "standard", dense=True)),
                        ("clf", GaussianNB()),
                    ]
                ),
                {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
            )
        else:
            exps[("Naive Bayes", fs)] = (
                Pipeline(
                    [
                        ("features", make_features(fs, "minmax")),
                        ("clf", MultinomialNB()),
                    ]
                ),
                {
                    "clf__alpha": [0.01, 0.1, 0.5, 1.0],
                    "clf__fit_prior": [True, False],
                },
            )
        exps[("Random Forest", fs)] = (
            Pipeline(
                [
                    ("features", make_features(fs, None)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "clf__max_depth": [20, None],
                "clf__min_samples_leaf": [1, 5],
                "clf__max_features": ["sqrt"],
            },
        )
    return exps


FEATURE_COLS = ["full_text", "tags_str"] + STRUCT_FEATURES


def evaluate(model, X_, y_):
    pred = model.predict(X_)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_)[:, 1]
    else:
        proba = pred.astype(float)
    return {
        "Accuracy": accuracy_score(y_, pred),
        "Precision": precision_score(y_, pred, zero_division=0),
        "Recall": recall_score(y_, pred, zero_division=0),
        "F1": f1_score(y_, pred, zero_division=0),
        "Macro F1": f1_score(y_, pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_, proba) if len(np.unique(y_)) > 1 else np.nan,
    }


def run_suite(name, X_train, X_test, y_train, y_test, feature_sets):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    experiments = get_experiments(feature_sets)
    fitted, cv_rows = {}, []
    for (model_name, fs), (pipe, grid) in experiments.items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        fitted[(model_name, fs)] = gs.best_estimator_
        cv_rows.append(
            {
                "Suite": name,
                "Model": model_name,
                "Features": fs,
                "CV ROC-AUC (mean)": gs.best_score_,
                "CV ROC-AUC (std)": gs.cv_results_["std_test_score"][gs.best_index_],
                "Best params": {
                    k.replace("clf__", ""): v for k, v in gs.best_params_.items()
                },
                "Time (s)": round(time.time() - t0, 1),
            }
        )
        print(
            f"[{name}] {model_name:20s} | {fs:12s} | "
            f"CV AUC={gs.best_score_:.4f} | {time.time()-t0:.1f}s"
        )

    cv_df = pd.DataFrame(cv_rows)
    # Select best model by CV AUC only (no test peeking).
    best_row = cv_df.sort_values("CV ROC-AUC (mean)", ascending=False).iloc[0]
    best_key = (best_row.Model, best_row.Features)
    best_model = fitted[best_key]

    rows = []
    # Stronger baselines
    for strategy, label in [
        ("most_frequent", "Dummy (majority)"),
        ("stratified", "Dummy (stratified)"),
        ("constant", "Dummy (all-high)"),
    ]:
        kwargs = {"strategy": strategy, "random_state": RANDOM_STATE}
        if strategy == "constant":
            kwargs = {"strategy": "constant", "constant": 1}
        dummy = DummyClassifier(**kwargs).fit(X_train, y_train)
        rows.append({"Suite": name, "Model": label, "Features": "-", **evaluate(dummy, X_test, y_test)})

    for key, model in fitted.items():
        rows.append(
            {
                "Suite": name,
                "Model": key[0],
                "Features": key[1],
                **evaluate(model, X_test, y_test),
            }
        )
    test_df = pd.DataFrame(rows)
    return fitted, cv_df, test_df, best_key, best_model


def bootstrap_delta(y_true, proba_a, proba_b, pred_a, pred_b, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n_obs = len(y_true)
    d_auc, d_f1 = [], []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        d_auc.append(roc_auc_score(yt, proba_a[idx]) - roc_auc_score(yt, proba_b[idx]))
        d_f1.append(f1_score(yt, pred_a[idx], zero_division=0) - f1_score(yt, pred_b[idx], zero_division=0))
    return {
        "delta_auc_mean": float(np.mean(d_auc)),
        "delta_auc_ci95": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))],
        "delta_f1_mean": float(np.mean(d_f1)),
        "delta_f1_ci95": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
    }


def main():
    assert DATA_PATH.exists(), DATA_PATH
    print(DATA_NOTE)
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["CreationDate", "LastEditDate"])
    print("Raw shape:", df_raw.shape)
    print("Date range:", df_raw.CreationDate.min(), "->", df_raw.CreationDate.max())
    print("FavoriteCount all zero:", (df_raw.FavoriteCount.fillna(0) == 0).all())

    # Snapshot date: documented export day = max CreationDate in file.
    # Limitation: true API fetch time may be later; we use a conservative 60-day filter.
    snapshot = df_raw.CreationDate.max()
    df = df_raw.drop_duplicates(subset="QuestionId")
    df = df.drop_duplicates(subset=["Title", "Body"])
    df = df.dropna(subset=["Title", "Body"])
    df = df[(df.Title.str.len() > 0) & (df.Body.str.len() > 0)]
    age_days = (snapshot - df.CreationDate).dt.days
    df = df[age_days >= MIN_AGE_DAYS].copy()
    print(f"After {MIN_AGE_DAYS}d age filter from snapshot {snapshot.date()}: {len(df)}")

    parsed = df.Body.apply(parse_body).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    df = engineer_features(df)

    # --- EDA figures ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df.Score.clip(-10, 15), discrete=True, ax=axes[0], color="#4C72B0")
    axes[0].set(title="Score distribution (clipped)", xlabel="Score", ylabel="Questions")
    counts = df.is_opinion.value_counts().rename({True: "Opinion-type", False: "Normal Q&A"})
    counts.plot.bar(ax=axes[1], color=["#55A868", "#C44E52"])
    axes[1].set(title="Post type mix (after age filter)", ylabel="Questions")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    savefig("01_score_distribution")

    type_stats = (
        df.groupby("is_opinion")
        .agg(n=("label", "size"), high_rate=("label", "mean"), mean_score=("Score", "mean"))
        .rename(index={True: "Opinion-type", False: "Normal Q&A"})
    )
    type_stats.to_csv(RES_DIR / "post_type_stats.csv")
    print(type_stats)

    monthly = (
        df.assign(month=df.CreationDate.dt.to_period("M").astype(str))
        .groupby("month")
        .agg(n=("label", "size"), opinion_rate=("is_opinion", "mean"), high_rate=("label", "mean"))
        .reset_index()
    )
    monthly.to_csv(RES_DIR / "monthly_drift.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.month, monthly.opinion_rate, marker="o", label="Opinion-type share")
    ax.plot(monthly.month, monthly.high_rate, marker="s", label="High-quality rate")
    ax.set(title="Temporal drift", xlabel="Month", ylabel="Rate")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    plt.tight_layout()
    savefig("02_temporal_drift")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bal = df.loc[~df.is_opinion, "label"].value_counts().rename({0: "Low (Score<=0)", 1: "High (Score>=1)"})
    bal.plot.bar(ax=ax, color=["#C44E52", "#55A868"])
    ax.set(title="Primary task class balance (normal questions)", ylabel="Questions")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    savefig("03_class_balance_normal")

    # Structural EDA on normal questions
    normal = df[~df.is_opinion].copy()
    plot_feats = ["title_words", "body_words", "n_code_lines", "n_paragraphs", "n_links", "code_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, f in zip(axes.ravel(), plot_feats):
        sns.boxplot(data=normal, x="label", y=f, ax=ax, showfliers=False)
        ax.set(title=f, xlabel="label (0=low,1=high)")
    plt.tight_layout()
    savefig("04_structural_by_class")

    feature_sets_primary = ["text", "struct", "tags", "text+struct", "all"]

    # ========== PRIMARY: normal questions, random stratified split ==========
    X = normal[FEATURE_COLS]
    y = normal["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"Primary normal: train={len(X_train)} test={len(X_test)} "
        f"high_train={y_train.mean():.3f} high_test={y_test.mean():.3f}"
    )
    fitted_p, cv_p, test_p, best_key_p, best_p = run_suite(
        "primary_normal_random", X_train, X_test, y_train, y_test, feature_sets_primary
    )

    # Confusion + ROC for CV-selected best
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_p.predict(X_test),
        display_labels=["Low", "High"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Primary best by CV: {best_key_p[0]} | {best_key_p[1]}")
    plt.tight_layout()
    savefig("05_confusion_primary_best")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model_name in ["Logistic Regression", "Naive Bayes", "Random Forest"]:
        m = fitted_p[(model_name, "text+struct")]
        RocCurveDisplay.from_estimator(m, X_test, y_test, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="Chance")
    ax.set_title("ROC – primary normal questions (text+struct, no tags)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    savefig("06_roc_primary_textstruct")

    # Heatmaps for primary
    real = test_p[~test_p.Model.str.startswith("Dummy")]
    pivot_auc = real.pivot(index="Model", columns="Features", values="ROC-AUC")[
        feature_sets_primary
    ]
    pivot_f1 = real.pivot(index="Model", columns="Features", values="F1")[feature_sets_primary]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Primary test ROC-AUC (normal questions)")
    sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="Greens", ax=axes[1], cbar=False)
    axes[1].set_title("Primary test F1 (normal questions)")
    plt.tight_layout()
    savefig("07_featureset_heatmap_primary")

    # Metrics bars excluding tags-dominated story; include baselines annotation
    plot_df = real.melt(
        id_vars=["Model", "Features"],
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        var_name="Metric",
        value_name="Value",
    )
    g = sns.catplot(
        data=plot_df,
        x="Features",
        y="Value",
        hue="Model",
        col="Metric",
        kind="bar",
        col_wrap=5,
        height=3.5,
        aspect=0.85,
        palette="Set2",
        order=feature_sets_primary,
    )
    g.set(ylim=(0, 1))
    all_high_f1 = test_p.loc[test_p.Model == "Dummy (all-high)", "F1"].item()
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(all_high_f1, ls=":", c="grey", lw=1)
    g.figure.suptitle("Primary metrics (dotted = all-high dummy F1)", y=1.05)
    g.savefig(FIG_DIR / "08_test_metrics_primary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # LR coefficients for best CV model if LR, else LR text+struct
    lr_key = best_key_p if best_key_p[0] == "Logistic Regression" else ("Logistic Regression", "text+struct")
    lr = fitted_p[lr_key]
    feat_names = lr.named_steps["features"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["clf"].coef_.ravel(), index=feat_names)
    coefs.index = (
        coefs.index.str.replace("text_tfidf__", "[text] ")
        .str.replace("numeric__", "[struct] ")
        .str.replace("tags__", "[tag] ")
    )
    top = pd.concat([coefs.nsmallest(12), coefs.nlargest(12).iloc[::-1]])
    fig, ax = plt.subplots(figsize=(9, 8))
    top.sort_values().plot.barh(
        ax=ax, color=["#C44E52" if v < 0 else "#55A868" for v in top.sort_values()]
    )
    ax.set(title=f"LR coefficients ({lr_key[1]}) on normal questions", xlabel="Coefficient")
    plt.tight_layout()
    savefig("09_lr_coefficients_primary")

    struct_coefs = coefs[coefs.index.str.startswith("[struct]")].sort_values()
    if len(struct_coefs):
        fig, ax = plt.subplots(figsize=(8, 6))
        struct_coefs.plot.barh(
            ax=ax, color=["#C44E52" if v < 0 else "#55A868" for v in struct_coefs]
        )
        ax.set(title="LR structural coefficients (normal questions)", xlabel="Coefficient")
        plt.tight_layout()
        savefig("10_lr_structural_primary")

    # Bootstrap: text+struct vs text; all vs text+struct
    def proba_pred(model):
        return model.predict_proba(X_test)[:, 1], model.predict(X_test)

    boot = {}
    if ("Logistic Regression", "text+struct") in fitted_p and ("Logistic Regression", "text") in fitted_p:
        pa, ya = proba_pred(fitted_p[("Logistic Regression", "text+struct")])
        pb, yb = proba_pred(fitted_p[("Logistic Regression", "text")])
        boot["textstruct_minus_text"] = bootstrap_delta(y_test, pa, pb, ya, yb)
    if ("Logistic Regression", "all") in fitted_p and ("Logistic Regression", "text+struct") in fitted_p:
        pa, ya = proba_pred(fitted_p[("Logistic Regression", "all")])
        pb, yb = proba_pred(fitted_p[("Logistic Regression", "text+struct")])
        boot["all_minus_textstruct"] = bootstrap_delta(y_test, pa, pb, ya, yb)
    if ("Logistic Regression", "tags") in fitted_p and ("Logistic Regression", "text+struct") in fitted_p:
        pa, ya = proba_pred(fitted_p[("Logistic Regression", "tags")])
        pb, yb = proba_pred(fitted_p[("Logistic Regression", "text+struct")])
        boot["tags_minus_textstruct"] = bootstrap_delta(y_test, pa, pb, ya, yb)

    # ========== CHRONOLOGICAL SPLIT on normal questions ==========
    normal_sorted = normal.sort_values("CreationDate")
    cut = int(len(normal_sorted) * 0.8)
    train_c = normal_sorted.iloc[:cut]
    test_c = normal_sorted.iloc[cut:]
    Xtr, Xte = train_c[FEATURE_COLS], test_c[FEATURE_COLS]
    ytr, yte = train_c["label"], test_c["label"]
    print(
        f"Chrono normal: train={len(Xtr)} ({train_c.CreationDate.min().date()}->"
        f"{train_c.CreationDate.max().date()}) test={len(Xte)} "
        f"({test_c.CreationDate.min().date()}->{test_c.CreationDate.max().date()})"
    )
    # Smaller feature grid for chrono (LR focus + RF/NB on text+struct and all)
    fitted_c, cv_c, test_c_df, best_key_c, best_c = run_suite(
        "primary_normal_chrono",
        Xtr,
        Xte,
        ytr,
        yte,
        ["text", "struct", "tags", "text+struct", "all"],
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for fs, label in [("text+struct", "text+struct"), ("all", "all"), ("tags", "tags")]:
        RocCurveDisplay.from_estimator(
            fitted_c[("Logistic Regression", fs)], Xte, yte, ax=ax, name=f"LR {label}"
        )
    ax.plot([0, 1], [0, 1], ls="--", c="grey")
    ax.set_title("ROC – chronological split (normal questions)")
    plt.tight_layout()
    savefig("11_roc_chrono")

    # ========== SECONDARY: full data (documents tag confound) ==========
    Xf = df[FEATURE_COLS]
    yf = df["label"]
    Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(
        Xf, yf, test_size=TEST_SIZE, stratify=yf, random_state=RANDOM_STATE
    )
    fitted_f, cv_f, test_f, best_key_f, best_f = run_suite(
        "secondary_all_posts",
        Xtr_f,
        Xte_f,
        ytr_f,
        yte_f,
        ["text", "struct", "tags", "text+struct", "all"],
    )
    real_f = test_f[~test_f.Model.str.startswith("Dummy")]
    pivot_f = real_f.pivot(index="Model", columns="Features", values="ROC-AUC")[
        feature_sets_primary
    ]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sns.heatmap(pivot_f, annot=True, fmt=".3f", cmap="Oranges", ax=ax, cbar=False)
    ax.set_title("Secondary: ROC-AUC on ALL posts (includes opinion-type confound)")
    plt.tight_layout()
    savefig("12_featureset_heatmap_allposts")

    # Sensitivity: drop Score==0 (ambiguous) on normal questions
    clear = normal[normal.Score != 0].copy()
    clear["label"] = (clear.Score >= 1).astype(int)  # >=1 high, <=-1 low
    Xcl = clear[FEATURE_COLS]
    ycl = clear["label"]
    Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
        Xcl, ycl, test_size=TEST_SIZE, stratify=ycl, random_state=RANDOM_STATE
    )
    print(f"Sensitivity drop-zero: n={len(clear)} high={ycl.mean():.3f}")
    fitted_s, cv_s, test_s, best_key_s, best_s = run_suite(
        "sensitivity_drop_zero",
        Xtr_s,
        Xte_s,
        ytr_s,
        yte_s,
        ["text", "struct", "text+struct", "all"],
    )

    # Save tables
    cv_all = pd.concat([cv_p, cv_c, cv_f, cv_s], ignore_index=True)
    test_all = pd.concat([test_p, test_c_df, test_f, test_s], ignore_index=True)
    cv_all.to_csv(RES_DIR / "cv_results.csv", index=False)
    test_all.to_csv(RES_DIR / "test_results.csv", index=False)

    summary = {
        "data_note": DATA_NOTE,
        "snapshot_date": str(snapshot.date()),
        "min_age_days": MIN_AGE_DAYS,
        "n_after_age_filter": int(len(df)),
        "n_normal": int((~df.is_opinion).sum()),
        "n_opinion": int(df.is_opinion.sum()),
        "opinion_high_rate": float(df.loc[df.is_opinion, "label"].mean()),
        "normal_high_rate": float(df.loc[~df.is_opinion, "label"].mean()),
        "favorite_count_always_zero": True,
        "primary_best_by_cv": {"model": best_key_p[0], "features": best_key_p[1]},
        "primary_best_test": evaluate(best_p, X_test, y_test),
        "primary_baselines": test_p[test_p.Model.str.startswith("Dummy")][
            ["Model", "F1", "ROC-AUC"]
        ].to_dict(orient="records"),
        "chrono_best_by_cv": {"model": best_key_c[0], "features": best_key_c[1]},
        "chrono_best_test": evaluate(best_c, Xte, yte),
        "secondary_all_best_by_cv": {"model": best_key_f[0], "features": best_key_f[1]},
        "secondary_all_best_test": evaluate(best_f, Xte_f, yte_f),
        "sensitivity_drop_zero_best_by_cv": {"model": best_key_s[0], "features": best_key_s[1]},
        "sensitivity_drop_zero_best_test": evaluate(best_s, Xte_s, yte_s),
        "bootstrap": boot,
        "type_stats": type_stats.reset_index().to_dict(orient="records"),
    }
    with open(RES_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== PRIMARY best (CV-selected) ===")
    print(best_key_p, summary["primary_best_test"])
    print("Baselines:", summary["primary_baselines"])
    print("Bootstrap:", json.dumps(boot, indent=2))
    print("Saved figures:", sorted(p.name for p in FIG_DIR.glob("*.png")))
    print("Saved results:", sorted(p.name for p in RES_DIR.glob("*")))


if __name__ == "__main__":
    main()
