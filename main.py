"""
main.py — MarineReAI: AI-Powered Marine Excess-of-Loss Reinsurance Pricing Workbench
─────────────────────────────────────────────────────────────────────────
Single-file Streamlit deployment build, merged from the original
multi-agent Colab notebook (core_pricing.py + agents.py + app.py).

5-Agent system:
  Agent 1 — Data Analyst        : cleans data, engineers features, splits train/test
  Agent 2 — ML Pricing          : XGBoost Poisson (frequency) + Gamma (severity)
  Agent 3 — Actuarial Pricing   : Pure Premium, Gross Premium, Rate on Line
  Agent 4 — Report & Visualization : charts + downloadable PDF report
  Agent 5 — Response (Q&A)      : conversational assistant (Gemini 2.5 Flash, optional)

Run locally:
    pip install -r requirements.txt
    streamlit run main.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub (main.py + requirements.txt at the repo root)
    2. On share.streamlit.io, create a new app pointing at this repo, branch `main`,
       and main file path `main.py`
    3. Gemini API key is entered by the end user in the sidebar at runtime — no
       secrets need to be configured for the AI Assistant / Q&A tabs to be optional.
"""

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1 — core_pricing.py  (pricing engine, UI-agnostic)
# ═══════════════════════════════════════════════════════════════════════


import io
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER

# ─────────────────────────────────────────────────────────────
#  Chart styling (dark theme, matches the original CLI tool)
# ─────────────────────────────────────────────────────────────
S = {
    "bg": "#0d1117", "fg": "#e6edf3", "grid": "#21262d",
    "a1": "#58a6ff", "a2": "#f78166", "a3": "#3fb950", "a4": "#d2a8ff",
}

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(S["bg"])
    ax.tick_params(colors=S["fg"], labelsize=8)
    ax.xaxis.label.set_color(S["fg"]); ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_color(S["fg"]); ax.yaxis.label.set_fontsize(9)
    ax.title.set_color(S["a1"]); ax.title.set_fontsize(10)
    ax.grid(True, color=S["grid"], linewidth=0.5, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_edgecolor(S["grid"])
    if title:  ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


# ─────────────────────────────────────────────────────────────
#  Column-mapping helpers
# ─────────────────────────────────────────────────────────────
def col(colmap, role):
    """Real dataset column name for a generic role, or None."""
    v = colmap.get(role)
    return v if v else None

def has(colmap, df, role):
    c = col(colmap, role)
    return bool(c) and c in df.columns


# ─────────────────────────────────────────────────────────────
#  STEP 1 – Clean
# ─────────────────────────────────────────────────────────────
def clean_data(df, colmap):
    """Drop obviously invalid rows for whichever columns are mapped.

    Defensive by design: a single bad column mapping should never be able to
    silently wipe out the entire dataset. Each filter is checked before it is
    applied; if applying it would remove every remaining row, the filter is
    skipped (with a note) rather than zeroing out the DataFrame. If, after all
    filters, zero rows remain, a clear actionable error is raised instead of
    letting an empty DataFrame flow downstream into model training.
    """
    notes = []
    n0 = len(df)
    if n0 == 0:
        raise ValueError("The uploaded dataset has no rows to price.")

    def safe_filter(frame, mask, description):
        candidate = frame[mask]
        if len(candidate) == 0:
            notes.append(
                f"Skipped cleaning rule '{description}' — it would have removed "
                f"all {len(frame):,} remaining rows. This usually means a column "
                f"was auto-mapped incorrectly; please check the column mapping."
            )
            return frame
        return candidate

    if has(colmap, df, "premium"):
        pcol = col(colmap, "premium")
        df = safe_filter(df, df[pcol] > 0, f"{pcol} > 0")
    if has(colmap, df, "exposure_value"):
        ecol = col(colmap, "exposure_value")
        df = safe_filter(df, df[ecol] > 0, f"{ecol} > 0")
    if has(colmap, df, "deductible") and has(colmap, df, "sum_insured"):
        dcol, scol = col(colmap, "deductible"), col(colmap, "sum_insured")
        df = safe_filter(df, df[dcol] < df[scol], f"{dcol} < {scol}")

    # ── Loss_Ratio outlier cap (Improvement: fix #4) ────────────────────
    # Loss ratios above 5.0 (500%) are almost always data errors — corrupted
    # claims amounts or mis-mapped columns.  We cap rather than drop so the
    # row still contributes to frequency modelling.
    LOSS_RATIO_CAP = 5.0
    lr_candidates = [c for c in df.columns
                     if "loss" in c.lower() and "ratio" in c.lower()
                     and pd.api.types.is_numeric_dtype(df[c])]
    for lrc in lr_candidates:
        n_outliers = int((df[lrc] > LOSS_RATIO_CAP).sum())
        if n_outliers > 0:
            df[lrc] = df[lrc].clip(upper=LOSS_RATIO_CAP)
            notes.append(
                f"Loss_Ratio outliers capped at {LOSS_RATIO_CAP:.0f}× "
                f"({n_outliers:,} rows had values above {LOSS_RATIO_CAP:.0f}×; "
                f"these were capped, not dropped, to preserve frequency signal)."
            )

    n1 = len(df)
    notes.append(f"Removed {n0 - n1} anomalous rows ({n0:,} → {n1:,}).")

    if n1 == 0:
        raise ValueError(
            "All rows were removed during data cleaning, leaving an empty "
            "dataset. This almost always means the automatic column mapping "
            "picked the wrong column for premium, exposure, deductible, or sum "
            "insured. Please review the 'Auto-detected column mapping' panel "
            "and correct it before re-running."
        )
    return df.reset_index(drop=True), notes


# ─────────────────────────────────────────────────────────────
#  STEP 2 – Feature engineering
# ─────────────────────────────────────────────────────────────
def engineer_features(df, colmap, retention, limit):
    df = df.copy()
    engineered = []

    if has(colmap, df, "premium") and has(colmap, df, "sum_insured"):
        df["Premium_Rate"] = df[col(colmap, "premium")] / df[col(colmap, "sum_insured")]
        engineered.append("Premium_Rate")

    if has(colmap, df, "exposure_value"):
        if has(colmap, df, "unit_count"):
            df["Value_per_Unit"] = df[col(colmap, "exposure_value")] / (df[col(colmap, "unit_count")] + 1)
        else:
            df["Value_per_Unit"] = df[col(colmap, "exposure_value")]
        engineered.append("Value_per_Unit")

    if has(colmap, df, "risk_score"):
        claims_term = (1 + df[col(colmap, "claims_count")] * 0.2) if has(colmap, df, "claims_count") else 1.0
        protective_term = (df[col(colmap, "protective_factor")] + 1) if has(colmap, df, "protective_factor") else 1.0
        df["Risk_Score_Compound"] = df[col(colmap, "risk_score")] * claims_term / protective_term
        engineered.append("Risk_Score_Compound")

    if has(colmap, df, "sum_insured") and has(colmap, df, "deductible"):
        df["Net_Sum_Insured"] = df[col(colmap, "sum_insured")] - df[col(colmap, "deductible")]
        engineered.append("Net_Sum_Insured")

    if retention is not None and has(colmap, df, "sum_insured"):
        si = df[col(colmap, "sum_insured")]
        df["Layer_Exposed"] = (si > retention).astype(int)
        df["Layer_Amount"]  = np.clip(si - retention, 0, limit)
        engineered += ["Layer_Exposed", "Layer_Amount"]

    return df, engineered


def encode_categoricals(df, colmap):
    df = df.copy()
    cat_cols = [c for c in (colmap.get("categorical") or []) if c in df.columns]
    for c in cat_cols:
        le = LabelEncoder()
        df[f"{c}_enc"] = le.fit_transform(df[c].astype(str))
    return df, cat_cols


def build_feature_list(df, colmap, engineered, cat_cols):
    feature_cols = []
    for role in ("age_years", "unit_count", "exposure_value", "sum_insured",
                 "deductible", "deductible_pct", "risk_score",
                 "protective_factor", "claims_count", "coastal_distance"):
        c = col(colmap, role)
        if c and c in df.columns:
            feature_cols.append(c)

    feature_cols += [f for f in engineered if f in df.columns]

    for flag in (colmap.get("seasonal_flags") or []):
        if flag in df.columns:
            feature_cols.append(flag)

    feature_cols += [f"{c}_enc" for c in cat_cols]

    seen, deduped = set(), []
    for f in feature_cols:
        if f not in seen:
            deduped.append(f)
            seen.add(f)
    return deduped


# ─────────────────────────────────────────────────────────────
#  STEP 3 – Train / test split
# ─────────────────────────────────────────────────────────────
def prepare_targets(df, freq_target_col, sev_target_col, feature_cols,
                     exclude_targets_from_features=True):
    notes = []
    feats = list(feature_cols)
    if exclude_targets_from_features:
        before = len(feats)
        feats = [f for f in feats if f not in (freq_target_col, sev_target_col)]
        if len(feats) != before:
            notes.append("Target column(s) were excluded from the feature list to avoid leakage.")
    else:
        if freq_target_col in feats or sev_target_col in feats:
            notes.append(
                "Warning: a target column also appears as a model feature. "
                "This typically means the model will learn to copy that column "
                "almost perfectly — results may look artificially good."
            )
    return feats, notes


def train_test_prepare(df, feature_cols, freq_target_col, sev_target_col, test_size=0.2, random_state=42):
    """
    Build train/test splits defensively:
      - coerces feature/target columns to numeric
      - IMPUTES missing values (median for features, 0 for target cols) instead
        of silently dropping rows — preserving signal from smaller/older vessels
        where missingness is non-random (Improvement 4)
      - guards against having too few usable rows to split at all
      - shrinks test_size automatically for small datasets so neither the
        train nor the test split ends up empty
      - SEVERITY SPLIT IS CLAIMS-ONLY: a Gamma model requires strictly
        positive labels, and most insurance portfolios have plenty of
        zero-claim policies. So the frequency split uses every row (it's
        predicting a count, including zero), while the severity split is
        built only from rows where the claim-amount column is > 0 — i.e.
        "what does a claim cost, given that one happened". This mirrors
        standard actuarial frequency-severity modelling practice.
    """
    needed_cols = list(dict.fromkeys(list(feature_cols) + [freq_target_col, sev_target_col]))
    work = df[needed_cols].apply(pd.to_numeric, errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)

    # ── Imputation (Improvement 4) ──────────────────────────────────────────
    # Fill missing feature values with column median (robust to skew).
    # Target columns get 0-fill since NaN claims = no claim recorded.
    n_before = len(work)
    imputed_counts = {}
    for c in work.columns:
        n_miss = int(work[c].isna().sum())
        if n_miss > 0:
            if c in (freq_target_col, sev_target_col):
                work[c] = work[c].fillna(0)
            else:
                work[c] = work[c].fillna(work[c].median())
            imputed_counts[c] = n_miss
    if imputed_counts:
        summary = ", ".join(f"{c}: {v}" for c, v in imputed_counts.items())
        print(f"[Imputation] Filled missing values (median/0): {summary}")
    work = work.dropna()  # catch any remaining (e.g. all-NaN columns)

    n = len(work)
    if n == 0:
        raise ValueError(
            "No valid rows remain after removing missing/non-numeric/infinite "
            "values from the feature and target columns. This usually means "
            "the column mapping picked a non-numeric column as a feature or "
            "target, or the data has too many blank cells. Check the "
            "auto-detected column mapping and the data quality of your CSV."
        )
    if n < 4:
        raise ValueError(
            f"Only {n} usable row(s) remain after cleaning — at least 4 rows "
            "are needed to train and evaluate a model. Please upload a larger "
            "dataset, or check for excessive missing values."
        )

    X = work[list(feature_cols)]
    y_freq = work[freq_target_col]

    # Shrink/grow test_size so both the train and test splits get at least
    # one row, regardless of how small the (cleaned) dataset is.
    min_frac = 1.0 / n
    eff_test_size = min(max(test_size, min_frac), 1 - min_frac)

    X_tr, X_te, yf_tr, yf_te = train_test_split(
        X, y_freq, test_size=eff_test_size, random_state=random_state)

    # Severity: claims-only subset (Gamma needs strictly positive labels).
    work_pos = work[work[sev_target_col] > 0]
    n_pos = len(work_pos)
    if n_pos == 0:
        raise ValueError(
            f"The severity target column '{sev_target_col}' has no positive "
            "values (every row is 0 or missing) after cleaning. A severity "
            "(Gamma) model needs at least some rows where a claim amount "
            "greater than zero was recorded. Please check that the column "
            "mapping picked the correct claim-amount column, and that the "
            "dataset actually contains claims data."
        )
    if n_pos < 4:
        raise ValueError(
            f"Only {n_pos} row(s) have a positive value in the severity "
            f"target column '{sev_target_col}' — at least 4 are needed to "
            "train and evaluate the severity model. Please upload a larger "
            "dataset with more claims, or check the column mapping."
        )

    Xs = work_pos[list(feature_cols)]
    ys = work_pos[sev_target_col]
    min_frac_pos = 1.0 / n_pos
    eff_test_size_pos = min(max(test_size, min_frac_pos), 1 - min_frac_pos)

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        Xs, ys, test_size=eff_test_size_pos, random_state=random_state)

    return X_tr, X_te, yf_tr, yf_te, Xs_tr, Xs_te, ys_tr, ys_te


# ─────────────────────────────────────────────────────────────
#  STEP 4 – Train models
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  OPTIONAL: Optuna hyperparameter tuning (50 trials)
# ─────────────────────────────────────────────────────────────
def tune_freq_severity(X_tr, yf_tr, Xs_tr, ys_tr, n_trials=50):
    """
    Run a 50-trial Optuna search on both frequency and severity models.

    Uses 3-fold CV internally (fast enough for Colab with n_trials=50).
    Returns best_params dict with keys 'freq' and 'sev', each a dict
    of XGBoost hyperparameters ready to unpack into the final model.

    Called only when the user ticks "Tune model (Optuna)" in the sidebar.
    Adds ~60-120 seconds on a Colab T4 GPU for 10k rows.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import cross_val_score, KFold

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # ── Frequency objective (Poisson) ────────────────────────────────────
    def freq_objective(trial):
        params = dict(
            objective="count:poisson",
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0),
            random_state=42, verbosity=0,
        )
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(
            model, X_tr, yf_tr, cv=kf,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        return -scores.mean()

    freq_study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    freq_study.optimize(freq_objective, n_trials=n_trials, show_progress_bar=False)
    best_freq = freq_study.best_params
    best_freq["objective"] = "count:poisson"
    best_freq["random_state"] = 42
    best_freq["verbosity"] = 0

    # ── Severity objective (Gamma) ────────────────────────────────────────
    def sev_objective(trial):
        params = dict(
            objective="reg:gamma",
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 3, 20),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0),
            random_state=42, verbosity=0,
        )
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(
            model, Xs_tr, ys_tr, cv=kf,
            scoring="neg_mean_squared_log_error", n_jobs=-1,
        )
        return -scores.mean()

    sev_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    sev_study.optimize(sev_objective, n_trials=n_trials, show_progress_bar=False)
    best_sev = sev_study.best_params
    best_sev["objective"] = "reg:gamma"
    best_sev["random_state"] = 42
    best_sev["verbosity"] = 0

    return {"freq": best_freq, "sev": best_sev}

def train_freq_severity(X_tr, X_te, yf_tr, yf_te, Xs_tr, Xs_te, ys_tr, ys_te, best_params=None):
    """
    Train frequency (Poisson XGBoost) and severity (Gamma XGBoost) models.

    V7 improvements over V6:
    - Severity model uses stronger regularisation + more estimators to handle
      the heavy-tailed claim distribution (skewness ~3.2, kurtosis ~14.6).
    - Adds RMSLE (Root Mean Squared Log Error) and Relative RMSE (RMSE/mean)
      to sev_metrics — these are the actuarially correct ways to judge a Gamma
      model because the Gamma objective minimises deviance in *log-space*, not
      dollar-space.  Raw dollar RMSE will always be large (~1M) when the mean
      claim is ~1M and CV=1.37; RMSLE puts the performance in context.
    - Severity CV scoring switches to neg_mean_squared_log_error so the CV
      metric is consistent with the model's own objective.
    """
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import mean_squared_log_error

    # ── Resolve hyperparameters (Optuna best or sensible defaults) ──────
    _default_freq = dict(
        objective="count:poisson", n_estimators=300, max_depth=5,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    _default_sev = dict(
        objective="reg:gamma", n_estimators=500, max_depth=4,
        learning_rate=0.03, subsample=0.7, colsample_bytree=0.7,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=2.0,
        random_state=42, verbosity=0,
    )
    freq_hp = (best_params or {}).get("freq", _default_freq)
    sev_hp  = (best_params or {}).get("sev",  _default_sev)

    # ── Frequency: 5-fold CV ──────────────────────────────────────────────
    _freq_proto = xgb.XGBRegressor(**freq_hp)
    _X_all = pd.concat([X_tr, X_te])
    _y_all = pd.concat([yf_tr, yf_te])
    freq_cv_scores = cross_val_score(
        _freq_proto, _X_all, _y_all,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    freq_cv_rmse = float(-freq_cv_scores.mean())

    # ── Severity: 5-fold CV in LOG space (consistent with Gamma objective) ─
    # Using neg_mean_squared_log_error so CV score matches what the model
    # actually optimises — deviance in log-space, not dollar-space RMSE.
    _sev_proto = xgb.XGBRegressor(**sev_hp)
    _Xs_all = pd.concat([Xs_tr, Xs_te])
    _ys_all = pd.concat([ys_tr, ys_te])
    # CV RMSLE — the correct metric for a log-scale model
    sev_cv_rmsle_scores = cross_val_score(
        _sev_proto, _Xs_all, _ys_all,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_log_error",
        n_jobs=-1,
    )
    sev_cv_rmsle = float(np.sqrt(-sev_cv_rmsle_scores.mean()))

    # Also keep dollar RMSE CV for legacy display
    sev_cv_dollar_scores = cross_val_score(
        _sev_proto, _Xs_all, _ys_all,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    sev_cv_rmse = float(-sev_cv_dollar_scores.mean())

    # ── Final frequency model ─────────────────────────────────────────────
    freq_model = xgb.XGBRegressor(**freq_hp)
    freq_model.fit(X_tr, yf_tr, eval_set=[(X_te, yf_te)], verbose=False)
    yf_pred = np.maximum(freq_model.predict(X_te), 0)
    freq_metrics = {
        "RMSE":    float(np.sqrt(mean_squared_error(yf_te, yf_pred))),
        "MAE":     float(mean_absolute_error(yf_te, yf_pred)),
        "CV_RMSE": freq_cv_rmse,
        "Relative_RMSE_pct": float(np.sqrt(mean_squared_error(yf_te, yf_pred)) / max(yf_te.mean(), 1e-9) * 100),
    }

    # ── Final severity model ──────────────────────────────────────────────
    sev_model = xgb.XGBRegressor(**sev_hp)
    sev_model.fit(Xs_tr, ys_tr, eval_set=[(Xs_te, ys_te)], verbose=False)
    ys_pred = np.maximum(sev_model.predict(Xs_te), 1e-6)

    # Dollar RMSE (always large relative to mean for insurance severity)
    sev_dollar_rmse = float(np.sqrt(mean_squared_error(ys_te, ys_pred)))

    # RMSLE — the right metric for a log-scale Gamma model
    # Clamp predictions to be strictly positive before log
    ys_pred_clamped = np.maximum(ys_pred, 1.0)
    ys_te_clamped   = np.maximum(ys_te.values, 1.0)
    sev_rmsle = float(np.sqrt(mean_squared_log_error(ys_te_clamped, ys_pred_clamped)))

    sev_metrics = {
        "RMSE":             sev_dollar_rmse,
        "MAE":              float(mean_absolute_error(ys_te, ys_pred)),
        "CV_RMSE":          sev_cv_rmse,          # dollar CV RMSE (legacy)
        "RMSLE":            sev_rmsle,             # log-space hold-out RMSE
        "CV_RMSLE":         sev_cv_rmsle,          # log-space 5-fold CV RMSE
        "Relative_RMSE_pct": float(sev_dollar_rmse / max(ys_te.mean(), 1.0) * 100),
        "Mean_Actual":      float(ys_te.mean()),
    }
    return freq_model, sev_model, freq_metrics, sev_metrics


def feature_importance(freq_model, sev_model, feature_cols):
    fi_freq = pd.Series(freq_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fi_sev  = pd.Series(sev_model.feature_importances_,  index=feature_cols).sort_values(ascending=False)
    return {"frequency": fi_freq, "severity": fi_sev}


# ─────────────────────────────────────────────────────────────
#  STEP 5 – Actuarial pricing
# ─────────────────────────────────────────────────────────────
def price_portfolio(df, feature_cols, freq_model, sev_model, colmap,
                     retention, limit, loading_pct, base_ccy="USD"):
    df = df.copy()
    X = df[feature_cols]
    pred_freq = np.maximum(freq_model.predict(X), 0)
    pred_sev  = np.maximum(sev_model.predict(X), 1e-6)
    df["pred_freq"] = pred_freq
    df["pred_sev"]  = pred_sev
    df["expected_loss"] = pred_freq * pred_sev

    si_col = col(colmap, "sum_insured")
    pricing_mode = "Excess-of-Loss Layer" if retention is not None else "Ground-Up / Primary"

    if retention is not None and si_col:
        def layer_factor(row):
            si = row[si_col]
            if si <= retention:
                return 0.0
            return min(si - retention, limit) / si
        df["layer_factor"] = df.apply(layer_factor, axis=1)
        df["layer_expected_loss"] = df["expected_loss"] * df["layer_factor"]

        layer_df = df[df["Layer_Exposed"] == 1]
        n_layer = len(layer_df)
        layer_si = layer_df[si_col].sum()
        pure_premium = df["layer_expected_loss"].sum()
        total_limit = layer_df["Layer_Amount"].sum()
        denom = layer_si
    else:
        n_layer = len(df)
        layer_si = df[si_col].sum() if si_col else np.nan
        total_limit = layer_si
        pure_premium = df["expected_loss"].sum()
        denom = layer_si

    total_exp_loss = df["expected_loss"].sum()
    avg_freq = float(pred_freq.mean())
    avg_sev  = float(pred_sev.mean())
    gross_premium = pure_premium * (1 + loading_pct)
    pure_rol  = pure_premium / denom if denom else 0
    gross_rol = gross_premium / denom if denom else 0
    rol_on_limit = gross_premium / total_limit if total_limit else 0

    ps = {
        "pricing_mode": pricing_mode,
        "total_policies": len(df),
        "layer_exposed_policies": n_layer,
        "total_si": df[si_col].sum() if si_col else np.nan,
        "layer_si_exposed": layer_si,
        "total_limit_written": total_limit,
        "avg_freq": avg_freq,
        "avg_sev": avg_sev,
        "total_expected_loss": float(total_exp_loss),
        "pure_premium": float(pure_premium),
        "pure_premium_rol": float(pure_rol),
        "loading_pct": loading_pct,
        "gross_premium": float(gross_premium),
        "gross_premium_rol": float(gross_rol),
        "rol_on_limit": float(rol_on_limit),
        "retention": retention,
        "limit": limit,
        "base_ccy": base_ccy,
    }

    # ── Improvement 1: Historical loss ratio validation ───────────────────
    hist_prem_col   = col(colmap, "premium")
    hist_claims_col = col(colmap, "claim_amount")
    hist_lr = np.nan
    model_lr = np.nan
    if hist_prem_col and hist_prem_col in df.columns and hist_claims_col and hist_claims_col in df.columns:
        total_hist_prem   = df[hist_prem_col].sum()
        total_hist_claims = df[hist_claims_col].sum()
        if total_hist_prem > 0:
            hist_lr = total_hist_claims / total_hist_prem
        if gross_premium > 0:
            model_lr = total_exp_loss / total_hist_prem
    ps["hist_loss_ratio"]  = float(hist_lr)  if not np.isnan(hist_lr)  else None
    ps["model_loss_ratio"] = float(model_lr) if not np.isnan(model_lr) else None

    # ── Improvement 3: CAT correlation loading for XoL pricing ────────────
    # Cyclone events hit many policies simultaneously — summing independent
    # per-policy expected losses understates aggregate risk for XoL treaties.
    # We apply a variance-loaded margin: simulate correlated cyclone-year
    # losses (a single multiplier hits ALL cyclone-season policies), compute
    # the 90th-pct aggregate loss, and add a loading for the excess over the
    # mean. This turns the flat "1+loading_pct" into a reinsurance-correct number.
    cat_loading_amt = 0.0
    cat_loading_pct_applied = 0.0
    if retention is not None:
        cyclone_flag_col = next(
            (c for c in df.columns if "cyclone" in c.lower() and "flag" in c.lower()), None
        )
        if cyclone_flag_col and cyclone_flag_col in df.columns:
            rng = np.random.default_rng(42)
            N_SIM = 5000
            # Cyclone-season policies share a correlated year multiplier.
            cyclone_mask = df[cyclone_flag_col].astype(bool).values
            base_losses  = df["layer_expected_loss"].values if "layer_expected_loss" in df.columns else df["expected_loss"].values
            agg_losses = np.zeros(N_SIM)
            for sim in range(N_SIM):
                year_mult = rng.lognormal(mean=0.0, sigma=0.45)  # correlated peril multiplier
                sim_losses = base_losses.copy()
                sim_losses[cyclone_mask] *= year_mult
                agg_losses[sim] = sim_losses.sum()
            mean_sim   = agg_losses.mean()
            pct90_sim  = np.percentile(agg_losses, 90)
            cat_loading_amt        = max(pct90_sim - mean_sim, 0.0)
            cat_loading_pct_applied = cat_loading_amt / pure_premium if pure_premium > 0 else 0.0
            # Add cat loading on top of existing expense loading
            gross_premium += cat_loading_amt
            gross_rol      = gross_premium / denom if denom else 0
            rol_on_limit   = gross_premium / total_limit if total_limit else 0
            ps["cat_loading_amount"]  = float(cat_loading_amt)
            ps["cat_loading_pct"]     = float(cat_loading_pct_applied)
            ps["gross_premium"]       = float(gross_premium)
            ps["gross_premium_rol"]   = float(gross_rol)
            ps["rol_on_limit"]        = float(rol_on_limit)
        else:
            ps["cat_loading_amount"] = 0.0
            ps["cat_loading_pct"]    = 0.0
    else:
        ps["cat_loading_amount"] = 0.0
        ps["cat_loading_pct"]    = 0.0

    if retention is not None:
        notes = [
            f"Layer: {base_ccy} {limit:,.2f} xs {base_ccy} {retention:,.2f} per risk/occurrence",
            f"Total policies/risks: {len(df):,}",
            f"Layer-exposed policies (SI > {base_ccy}{retention:,.2f}): {n_layer:,} ({n_layer/len(df)*100:.1f}%)",
            f"Total limit written: {base_ccy} {total_limit:,.2f}",
        ]
    else:
        notes = [
            "Pricing mode: Ground-up / primary (no excess-of-loss layer)",
            f"Total policies/risks: {len(df):,}",
        ]
    notes += [
        f"Pure Premium: {base_ccy} {pure_premium:,.4f}",
        f"Gross Premium ({loading_pct*100:.0f}% loading + CAT load): {base_ccy} {gross_premium:,.4f}",
        f"Rate on Line (limit basis): {rol_on_limit*100:.4f}%",
    ]
    # ── Improvement 1: append loss ratio sanity check to notes ─────────────
    if ps.get("hist_loss_ratio") is not None:
        notes.append(
            f"⚖ Loss Ratio — Historical portfolio: {ps['hist_loss_ratio']*100:.1f}%  |  "
            f"Model-implied (loss / gross premium): {ps['model_loss_ratio']*100:.1f}%"
        )
    if ps.get("cat_loading_amount", 0) > 0:
        notes.append(
            f"🌀 CAT Correlation Loading (cyclone): {base_ccy} {ps['cat_loading_amount']:,.4f} "
            f"(+{ps['cat_loading_pct']*100:.1f}% on pure premium) — 5,000-sim correlated aggregate"
        )
    return df, ps, notes


# ─────────────────────────────────────────────────────────────
#  Charts (return matplotlib Figures — caller decides how to show/save them)
# ─────────────────────────────────────────────────────────────
def chart_category_geography(df, colmap, labels):
    cat_col = col(colmap, "category"); geo_col = col(colmap, "geography")
    if not (has(colmap, df, "category") or has(colmap, df, "geography")):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=S["bg"])
    clrs = [S["a1"], S["a2"], S["a3"], S["a4"], "#f0883e", "#79c0ff"]

    if has(colmap, df, "category"):
        counts = df[cat_col].value_counts()
        axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
                    colors=clrs[:len(counts)], textprops={"color": S["fg"], "fontsize": 8})
        axes[0].set_facecolor(S["bg"])
        axes[0].set_title(f"{labels['category']} Distribution", color=S["a1"], fontsize=10)
    else:
        axes[0].axis("off")

    if has(colmap, df, "category") and has(colmap, df, "geography"):
        sp = df.groupby([geo_col, cat_col]).size().unstack(fill_value=0)
        sp.plot(kind="bar", ax=axes[1], color=clrs[:sp.shape[1]], edgecolor="none")
        _style_ax(axes[1], f"{labels['category']} by {labels['geography']}", labels["geography"], "Count")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].legend(fontsize=6, labelcolor=S["fg"], facecolor=S["grid"], edgecolor=S["grid"])
    elif has(colmap, df, "geography"):
        gc = df[geo_col].value_counts()
        axes[1].barh(gc.index, gc.values, color=S["a1"])
        _style_ax(axes[1], f"{labels['geography']} Distribution", "Count", labels["geography"])
    else:
        axes[1].axis("off")

    fig.suptitle(f"{labels['category']} & {labels['geography']} Distribution", color=S["fg"], fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def chart_premium(df, colmap, labels, base_ccy):
    prem_col = col(colmap, "premium"); exp_col = col(colmap, "exposure_value"); risk_col = col(colmap, "risk_score")
    if not has(colmap, df, "premium"):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=S["bg"])
    axes[0].hist(df[prem_col], bins=40, color=S["a1"], edgecolor=S["bg"], alpha=0.85)
    _style_ax(axes[0], f"{labels['premium']} Distribution", f"{labels['premium']} ({base_ccy})", "Count")

    if has(colmap, df, "exposure_value"):
        if has(colmap, df, "risk_score"):
            axes[1].scatter(df[exp_col], df[prem_col], c=df[risk_col], cmap="plasma", alpha=0.4, s=8, linewidths=0)
        else:
            axes[1].scatter(df[exp_col], df[prem_col], color=S["a1"], alpha=0.4, s=8, linewidths=0)
        _style_ax(axes[1], f"{labels['premium']} vs {labels['exposure']}",
                  f"{labels['exposure']} ({base_ccy})", f"{labels['premium']} ({base_ccy})")
    else:
        axes[1].axis("off")

    if has(colmap, df, "entity_type"):
        et_col = col(colmap, "entity_type")
        vt = df[et_col].unique()
        bp = axes[2].boxplot([df[df[et_col]==v][prem_col].values for v in vt],
                              patch_artist=True, medianprops={"color": S["a2"], "linewidth": 2})
        for patch, c in zip(bp["boxes"], [S["a1"],S["a3"],S["a4"],S["a2"],"#f0883e","#79c0ff"]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        axes[2].set_xticks(range(1, len(vt)+1))
        axes[2].set_xticklabels(vt, rotation=30, ha="right", fontsize=7)
        _style_ax(axes[2], f"{labels['premium']} by {labels['entity_type']}", labels["entity_type"], f"{labels['premium']} ({base_ccy})")
    else:
        axes[2].axis("off")

    fig.suptitle(f"{labels['premium']} Analysis", color=S["fg"], fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def chart_feature_importance(fi, df=None, colmap=None, labels=None, base_ccy="USD"):
    """Combined Risk Drivers panel (Improvement 5):
    Left col — frequency/severity importance side-by-side.
    Right col — risk-score heatmap by geography (if available).
    This replaces the old separate chart_feature_importance + chart_risk_score calls.
    """
    has_geo = (colmap is not None and df is not None and
               has(colmap, df, "geography") and has(colmap, df, "risk_score"))

    n_cols = 3 if has_geo else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), facecolor=S["bg"])
    axes = list(axes)

    # Frequency importance
    top_f = fi["frequency"].head(10)
    clrs_f = [S["a2"]] + [S["a1"]] * (len(top_f) - 1)
    axes[0].barh(top_f.index[::-1], top_f.values[::-1], color=clrs_f[::-1], edgecolor=S["bg"])
    _style_ax(axes[0], "Frequency Drivers (Poisson)", "Importance", "Feature")

    # Severity importance
    top_s = fi["severity"].head(10)
    clrs_s = [S["a2"]] + [S["a3"]] * (len(top_s) - 1)
    axes[1].barh(top_s.index[::-1], top_s.values[::-1], color=clrs_s[::-1], edgecolor=S["bg"])
    _style_ax(axes[1], "Severity Drivers (Gamma)", "Importance", "Feature")

    # Geography risk heatmap (replaces chart_risk_score)
    if has_geo:
        risk_col = col(colmap, "risk_score")
        geo_col  = col(colmap, "geography")
        sr = df.groupby(geo_col)[risk_col].mean().sort_values()
        norm = (sr.values - sr.min()) / (sr.max() - sr.min() + 1e-9)
        axes[2].barh(sr.index, sr.values, color=plt.cm.RdYlGn_r(norm))
        _style_ax(axes[2],
                  f"Avg {(labels or {}).get('risk_score', 'Risk Score')} by Geography",
                  "Risk Score", "Geography")

    fig.suptitle("Risk Drivers — Feature Importance & Geography", color=S["fg"], fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def chart_risk_score(df, colmap, labels, base_ccy):
    # Kept for backward-compatibility; the combined panel is preferred.
    return None


def chart_layer(df, colmap, ps, retention, limit, loading_pct, base_ccy):
    si_col = col(colmap, "sum_insured")
    n_panels = 3 if (retention is not None and si_col) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 4.5), facecolor=S["bg"])
    axes = [axes] if n_panels == 1 else list(axes)

    idx = 0
    if retention is not None and si_col:
        axes[idx].hist(df[si_col], bins=50, color=S["a4"], edgecolor=S["bg"], alpha=0.8)
        axes[idx].axvline(retention, color=S["a2"], lw=2, ls="--", label=f"Retention {base_ccy}{retention:,.0f}")
        axes[idx].axvline(retention+limit, color=S["a3"], lw=2, ls="--", label=f"Limit {base_ccy}{retention+limit:,.0f}")
        _style_ax(axes[idx], "Sum Insured vs Layer", f"Sum Insured ({base_ccy})", "Count")
        axes[idx].legend(fontsize=7, labelcolor=S["fg"], facecolor=S["grid"], edgecolor=S["grid"])
        idx += 1

        layer_df = df[df["Layer_Amount"] > 0]
        axes[idx].hist(layer_df["Layer_Amount"], bins=40, color=S["a3"], edgecolor=S["bg"], alpha=0.8)
        _style_ax(axes[idx], "Layer Exposure per Policy", f"Layer Amount ({base_ccy})", "Count")
        idx += 1

    labels_ = ["Pure\nPremium", f"+{int(loading_pct*100)}%\nLoading", "Gross\nPremium"]
    vals = [ps["pure_premium"], ps["gross_premium"] - ps["pure_premium"], ps["gross_premium"]]
    axes[idx].bar(labels_, vals, color=[S["a1"], S["a2"], S["a3"]], edgecolor=S["bg"], width=0.5)
    for i, v in enumerate(vals):
        axes[idx].text(i, v, f"{base_ccy}{v:,.2f}", ha="center", color=S["fg"], fontsize=8)
    _style_ax(axes[idx], "Premium Build-Up", "", f"Amount ({base_ccy})")

    fig.suptitle("Layer Analysis & Pricing Build-Up" if retention is not None else "Pricing Build-Up",
                 color=S["fg"], fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def chart_risk_score(df, colmap, labels, base_ccy):
    risk_col = col(colmap, "risk_score"); geo_col = col(colmap, "geography"); prem_col = col(colmap, "premium")
    if not has(colmap, df, "risk_score"):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=S["bg"])

    if has(colmap, df, "geography"):
        sr = df.groupby(geo_col)[risk_col].mean().sort_values()
        norm = (sr.values - sr.min()) / (sr.max() - sr.min() + 1e-9)
        axes[0].barh(sr.index, sr.values, color=plt.cm.RdYlGn_r(norm))
        _style_ax(axes[0], f"Avg {labels['risk_score']} by {labels['geography']}", labels["risk_score"], labels["geography"])
    else:
        axes[0].hist(df[risk_col], bins=30, color=S["a1"], edgecolor=S["bg"])
        _style_ax(axes[0], f"{labels['risk_score']} Distribution", labels["risk_score"], "Count")

    flag_col = next((f for f in (colmap.get("seasonal_flags") or []) if f in df.columns), None)
    if has(colmap, df, "premium"):
        if flag_col:
            axes[1].scatter(df[risk_col], df[prem_col], c=df[flag_col], cmap="bwr", alpha=0.4, s=6)
            handles = [mpatches.Patch(color="blue", label="Off-peak"), mpatches.Patch(color="red", label="Peak")]
            axes[1].legend(handles=handles, fontsize=7, labelcolor=S["fg"], facecolor=S["grid"], edgecolor=S["grid"])
        else:
            axes[1].scatter(df[risk_col], df[prem_col], color=S["a1"], alpha=0.4, s=6)
        _style_ax(axes[1], f"{labels['risk_score']} vs {labels['premium']}", labels["risk_score"], f"{labels['premium']} ({base_ccy})")
    else:
        axes[1].axis("off")

    fig.suptitle(f"{labels['risk_score']} Profile", color=S["fg"], fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def make_all_charts(df, colmap, labels, fi, ps, retention, limit, loading_pct,
                     base_ccy):
    """Returns an ordered dict {name: matplotlib.Figure or None}."""
    charts = {}
    charts["Category & Geography"] = chart_category_geography(df, colmap, labels)
    charts["Premium Analysis"]     = chart_premium(df, colmap, labels, base_ccy)
    # Combined Risk Drivers panel (Improvement 5: replaces separate FI + Risk Score charts)
    charts["Risk Drivers"] = chart_feature_importance(
        fi, df=df, colmap=colmap, labels=labels, base_ccy=base_ccy
    )
    # Layer / Pricing Build-Up folded into pricing table for non-XoL use cases (Improvement 5)
    if retention is not None:
        charts["Layer / Pricing Build-Up"] = chart_layer(df, colmap, ps, retention, limit, loading_pct, base_ccy)
    return {k: v for k, v in charts.items() if v is not None}


# ─────────────────────────────────────────────────────────────
#  PDF report
# ─────────────────────────────────────────────────────────────
def build_pdf_bytes(meta, ps, notes, charts, base_ccy, key_findings=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    title_st = ParagraphStyle("TT", parent=styles["Title"], fontSize=20, spaceAfter=6,
                               textColor=colors.HexColor("#1a1f6e"), alignment=TA_CENTER)
    h1_st = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=4,
                            spaceBefore=10, textColor=colors.HexColor("#1a1f6e"))
    h2_st = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceAfter=3,
                            spaceBefore=6, textColor=colors.HexColor("#2d5016"))
    body_st = ParagraphStyle("B", parent=styles["Normal"], fontSize=9, spaceAfter=3, leading=14)
    grey_st = ParagraphStyle("G", parent=styles["Normal"], fontSize=8, textColor=colors.grey)

    story += [
        Spacer(1, 1*cm),
        Paragraph("MarineReAI", title_st),
        Paragraph(f"{meta['business_line']} – {ps['pricing_mode']} Pricing Report",
                  ParagraphStyle("S", parent=styles["Normal"], fontSize=13, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#2d5016"))),
        Spacer(1, 0.4*cm),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1f6e")),
        Spacer(1, 0.3*cm),
    ]

    layer_desc = (f"{base_ccy} {ps['limit']:,.2f} xs {base_ccy} {ps['retention']:,.2f} per risk"
                  if ps["retention"] is not None else "Ground-up / primary policy pricing")
    meta_rows = [
        ["Client", meta.get("client_name", "—")],
        ["Business Line", meta["business_line"]],
        ["Pricing Mode", ps["pricing_mode"]],
        ["Layer / Structure", layer_desc],
        ["Report Date", datetime.date.today().strftime("%d %B %Y")],
        ["Model", "XGBoost Poisson-Gamma Frequency-Severity"],
    ]
    mt = Table(meta_rows, colWidths=[4.5*cm, 12*cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1), colors.HexColor("#e8edf8")),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("GRID",(0,0),(-1,-1),0.3, colors.HexColor("#c0c8d8")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, colors.HexColor("#f5f7fb")]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story += [mt, PageBreak()]

    story.append(Paragraph("Executive Pricing Summary", h1_st))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1f6e")))
    story.append(Spacer(1, 0.3*cm))

    # Updated: Removed Rate on Line (exposure basis)
    prows = [
        ["Metric", f"Value ({base_ccy})"],
        ["Pure Premium (Expected Loss)", f"{base_ccy} {ps['pure_premium']:,.4f}"],
        [f"Gross Premium ({int(ps['loading_pct']*100)}% loading)", f"{base_ccy} {ps['gross_premium']:,.4f}"],
        ["Rate on Line (limit basis)", f"{ps['rol_on_limit']*100:.4f}%"],
    ]
    pt = Table(prows, colWidths=[9*cm, 8*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#1a1f6e")),
        ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("GRID",(0,0),(-1,-1),0.3, colors.HexColor("#c0c8d8")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f5e8"), colors.white]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story += [pt, Spacer(1, 0.4*cm)]

    if key_findings:
        story.append(Paragraph("Key Findings (AI-Generated Summary)", h1_st))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1f6e")))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(key_findings.replace("\n", "<br/>"), body_st))
        story.append(Spacer(1, 0.4*cm))

    # New: Detailed Analysis Report based on dataset
    story.append(Paragraph("Detailed Portfolio Analysis", h1_st))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1f6e")))
    story.append(Spacer(1, 0.3*cm))

    analysis_rows = [
        ["Portfolio Metric", "Value"],
        ["Total Policies / Risks", f"{ps['total_policies']:,}"],
        ["Total Sum Insured", f"{base_ccy} {ps['total_si']:,.2f}" if not pd.isna(ps['total_si']) else "N/A"]

    ]

    at = Table(analysis_rows, colWidths=[9*cm, 8*cm])
    at.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#1a1f6e")),
        ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("GRID",(0,0),(-1,-1),0.3, colors.HexColor("#c0c8d8")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f5f7fb"), colors.white]),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story += [at, Spacer(1, 0.4*cm), PageBreak()]

    for fig in charts.values():
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight", facecolor=S["bg"])
        img_buf.seek(0)
        story.append(Spacer(1, 0.3*cm))
        story.append(KeepTogether([RLImage(img_buf, width=16.5*cm, height=6*cm), Spacer(1, 0.2*cm)]))

    story.append(PageBreak())
    story.append(Paragraph("Actuarial Pricing Notes", h1_st))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1f6e")))
    story.append(Spacer(1, 0.2*cm))
    for n in notes:
        story.append(Paragraph(f"• {n}", body_st))

    # New: Pricing Methodology Section
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Pricing Methodology", h1_st))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1f6e")))
    story.append(Spacer(1, 0.2*cm))
    methodology_text = (
        "The pricing in this project utilizes a Machine Learning-driven Frequency-Severity approach. "
        "Two independent XGBoost regression models are trained on the underlying portfolio data:<br/><br/>"
        "<b>1. Frequency Model (Poisson):</b> Predicts the expected number of claims per policy per period based on historical risk features. <br/>"
        "<b>2. Severity Model (Gamma):</b> Predicts the expected cost of a claim (given that a claim has occurred) trained exclusively on positive claim instances. <br/><br/>"
        "The <i>Expected Loss</i> for each individual policy is calculated as the product of its predicted frequency and predicted severity. "
        "For Excess-of-Loss structures, a policy-specific layer factor scales the expected loss based on how much of the policy's Sum Insured penetrates the specified retention and limit boundaries. "
        "Finally, the <i>Gross Premium</i> is derived by aggregating the portfolio's pure expected losses and applying the configured expense and profit loading percentage."
    )
    story.append(Paragraph(methodology_text, body_st))

    story += [Spacer(1, 0.5*cm),
              Paragraph("Disclaimer", h2_st),
              Paragraph(
                  "MarineReAI automated pricing workbench – indicative purposes only. "
                  "Final pricing requires review by a qualified actuary and senior underwriter. "
                  "All values are illustrative based on the supplied dataset and configured assumptions.",
                  grey_st)]

    doc.build(story)
    return buf.getvalue()



# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2 — agents.py  (multi-agent orchestration layer, Gemini powered)
# ═══════════════════════════════════════════════════════════════════════



# ── Gemini SDK — supports both the new google-genai and legacy
#    google-generativeai packages, same compatibility shim as app.py ──
try:
    from google import genai
    _NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai
        _NEW_SDK = False
    except ImportError:
        genai = None
        _NEW_SDK = None

GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_client(api_key):
    """Build a Gemini client/model from an API key, or None if unavailable."""
    if _NEW_SDK is None or not api_key:
        return None
    try:
        if _NEW_SDK:
            return genai.Client(api_key=api_key)
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception:
        return None


def ask_gemini(client, prompt, context=""):
    """Call Gemini 2.5 Flash. Returns None if no client is configured."""
    if client is None:
        return None
    full = (context + "\n\n" + prompt) if context else prompt
    try:
        if _NEW_SDK:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=full)
            return resp.text.strip()
        resp = client.generate_content(full)
        return resp.text.strip()
    except Exception as e:
        return f"[Gemini unavailable: {e}]"


class BaseAgent:
    """Shared plumbing for every MarineReAI agent."""
    name = "Agent"
    role = ""

    def __init__(self, gemini_client=None):
        self.client = gemini_client

    def _llm(self, prompt, context="", fallback=""):
        out = ask_gemini(self.client, prompt, context)
        return out if out else fallback


# ─────────────────────────────────────────────────────────────
#  Agent 1 — Data Analyst
# ─────────────────────────────────────────────────────────────
class DataAnalystAgent(BaseAgent):
    """
    Loads the raw portfolio, cleans it (drops rows with premium<=0,
    exposure<=0, or deductible>=sum_insured), engineers actuarial
    features (Premium_Rate, Risk_Score_Compound, Layer_Exposed, etc.),
    selects the feature set used by the ML agent, and prepares the
    train/test splits (frequency uses all rows; severity uses
    claims-only rows, since Gamma needs strictly positive labels).
    """
    name = "Agent 1 — Data Analyst"
    role = "Cleans data, flags missing/invalid entries, selects features, splits train/test."

    def run(self, raw_df, colmap, retention, limit, auto_cfg):
        df, notes1 = clean_data(raw_df.copy(), colmap)
        df, engineered = engineer_features(df, colmap, retention, limit)
        df, cat_cols = encode_categoricals(df, colmap)
        feature_cols = build_feature_list(df, colmap, engineered, cat_cols)
        feature_cols, notes2 = prepare_targets(
            df, auto_cfg["freq_target"], auto_cfg["sev_target"],
            feature_cols, auto_cfg["exclude_targets"])
        splits = train_test_prepare(
            df, feature_cols, auto_cfg["freq_target"], auto_cfg["sev_target"])
        return dict(
            df=df, feature_cols=feature_cols, engineered=engineered,
            cat_cols=cat_cols, notes=notes1 + notes2, splits=splits,
        )

    def narrate(self, result, auto_cfg):
        prompt = (
            "You are the Data Analyst agent in MarineReAI, a marine reinsurance "
            "pricing system. In 2-3 short sentences, summarise the data cleaning "
            "and feature-selection outcome below for an underwriter. Be specific, "
            "plain-English, no jargon.\n\n"
            f"Frequency target column: {auto_cfg['freq_target']}\n"
            f"Severity target column: {auto_cfg['sev_target']}\n"
            f"Features selected ({len(result['feature_cols'])}): {result['feature_cols']}\n"
            f"Engineered features: {result['engineered']}\n"
            f"Cleaning notes: {result['notes']}"
        )
        fallback = (
            f"Cleaned the portfolio and selected {len(result['feature_cols'])} features "
            f"for modelling, including {', '.join(result['engineered']) or 'no engineered fields'}."
        )
        return self._llm(prompt, fallback=fallback)


# ─────────────────────────────────────────────────────────────
#  Agent 2 — ML Pricing
# ─────────────────────────────────────────────────────────────
class MLPricingAgent(BaseAgent):
    """
    Trains the two-part frequency-severity model:
      • XGBoost Poisson regressor  -> expected claim frequency
      • XGBoost Gamma regressor    -> expected claim severity (claims-only)
    """
    name = "Agent 2 — ML Pricing"
    role = "Trains XGBoost Poisson (frequency) and Gamma (severity) models."

    def run(self, feature_cols, splits, best_params=None):
        X_tr, X_te, yf_tr, yf_te, Xs_tr, Xs_te, ys_tr, ys_te = splits
        freq_model, sev_model, freq_metrics, sev_metrics = train_freq_severity(
            X_tr, X_te, yf_tr, yf_te, Xs_tr, Xs_te, ys_tr, ys_te,
            best_params=best_params)
        fi = feature_importance(freq_model, sev_model, feature_cols)
        return dict(
            freq_model=freq_model, sev_model=sev_model,
            freq_metrics=freq_metrics, sev_metrics=sev_metrics, fi=fi,
        )

    def narrate(self, result):
        top_freq = result["fi"]["frequency"].head(5).index.tolist()
        top_sev = result["fi"]["severity"].head(5).index.tolist()
        prompt = (
            "You are the ML Pricing agent in MarineReAI. In 2-3 short sentences, "
            "explain for a non-technical underwriter what drives claim frequency "
            "and what drives claim severity in this portfolio, based on the XGBoost "
            "feature importances below. Do not mention error metrics or accuracy numbers.\n\n"
            f"Top frequency drivers: {top_freq}\n"
            f"Top severity drivers: {top_sev}"
        )
        fallback = (
            f"Claim frequency is mainly driven by {', '.join(top_freq[:3])}, "
            f"while claim severity is mainly driven by {', '.join(top_sev[:3])}."
        )
        return self._llm(prompt, fallback=fallback)


# ─────────────────────────────────────────────────────────────
#  Agent 3 — Actuarial Pricing
# ─────────────────────────────────────────────────────────────
class ActuarialPricingAgent(BaseAgent):
    """
    Converts model predictions into actuarial premiums: applies the
    excess-of-loss layer factor (if any), aggregates to Pure Premium,
    applies the expense/profit loading to get Gross Premium, and
    derives the Rate-on-Line.
    """
    name = "Agent 3 — Actuarial Pricing"
    role = "Computes Pure Premium, Gross Premium and Rate-on-Line."

    def run(self, df, feature_cols, freq_model, sev_model, colmap,
            retention, limit, loading_pct, base_ccy):
        return price_portfolio(
            df, feature_cols, freq_model, sev_model, colmap,
            retention, limit, loading_pct, base_ccy=base_ccy)

    def narrate(self, ps):
        prompt = (
            "You are the Actuarial Pricing agent in MarineReAI. In 2-3 short "
            "sentences, give a plain-English pricing recommendation an underwriter "
            "could act on, based on the figures below.\n\n"
            f"Pricing mode: {ps['pricing_mode']}\n"
            f"Pure Premium: {ps['base_ccy']} {ps['pure_premium']:,.4f}\n"
            f"Gross Premium ({ps['loading_pct']*100:.0f}% loading): {ps['base_ccy']} {ps['gross_premium']:,.4f}\n"
            f"Rate on Line (limit basis): {ps['rol_on_limit']*100:.4f}%"
        )
        fallback = (
            f"Recommended gross premium is {ps['base_ccy']} {ps['gross_premium']:,.2f} "
            f"({ps['loading_pct']*100:.0f}% loading over pure premium), implying a "
            f"Rate on Line of {ps['rol_on_limit']*100:.2f}%."
        )
        return self._llm(prompt, fallback=fallback)


# ─────────────────────────────────────────────────────────────
#  Agent 4 — Report & Visualization
# ─────────────────────────────────────────────────────────────
class ReportAgent(BaseAgent):
    """
    Generates all matplotlib charts and the downloadable PDF report,
    including an AI-written "Key Findings" paragraph summarising the
    pricing run for a reinsurance underwriting committee.
    """
    name = "Agent 4 — Report & Visualization"
    role = "Builds charts and the downloadable PDF pricing report."

    def run(self, df_priced, colmap, labels, fi, ps, retention, limit, loading_pct, base_ccy):
        return make_all_charts(
            df_priced, colmap, labels, fi, ps, retention, limit, loading_pct, base_ccy)

    def narrate(self, ps, notes):
        prompt = (
            "You are the Report agent in MarineReAI. Write a 'Key Findings' "
            "paragraph (3-4 sentences) for an actuarial pricing PDF report, "
            "summarising the portfolio and pricing outcome below for a reinsurance "
            "underwriting committee. Plain English, no markdown.\n\n"
            f"Pricing summary: {ps}\n"
            f"Notes: {notes}"
        )
        fallback = (
            "This report summarises the frequency-severity pricing analysis for the "
            "portfolio above, derived from the XGBoost Poisson-Gamma model and the "
            "configured excess-of-loss layer."
        )
        return self._llm(prompt, fallback=fallback)

    def build_pdf(self, meta, ps, notes, charts, base_ccy, key_findings=None):
        return build_pdf_bytes(meta, ps, notes, charts, base_ccy, key_findings=key_findings)


# ─────────────────────────────────────────────────────────────
#  Agent 5 — Response (Q&A with memory)
# ─────────────────────────────────────────────────────────────
class ResponseAgent(BaseAgent):
    """
    Answers natural-language questions about the pricing run. Memory is
    the running chat_history list (passed in by the caller / Streamlit
    session state) — the full conversation so far is prepended to every
    prompt so Gemini can give coherent multi-turn answers.
    """
    name = "Agent 5 — Response"
    role = "Answers questions about the pricing run, with conversation memory."

    def answer(self, user_q, chat_history, system_ctx):
        hist_str = "\n".join(f"{r}: {m}" for r, m in chat_history[:-1])
        prompt = (
            (("Conversation history:\n" + hist_str + "\n") if hist_str else "")
            + f"\nUser question: {user_q}\nAnswer as the MarineReAI Response Agent:"
        )
        return self._llm(
            prompt, context=system_ctx,
            fallback="Enter a Gemini API key in the sidebar so I can answer questions.",
        )


# ─────────────────────────────────────────────────────────────
#  Orchestrator — runs Agents 1-4 in sequence
# ─────────────────────────────────────────────────────────────
class MarineReAIPipeline:
    """
    Coordinates Agents 1-4 end to end. Agent 5 (ResponseAgent) is used
    separately, on demand, from the Q&A tab — it depends on the output
    of this pipeline but isn't part of the linear pricing run.
    """

    def __init__(self, gemini_client=None):
        self.data_analyst = DataAnalystAgent(gemini_client)
        self.ml_pricing = MLPricingAgent(gemini_client)
        self.actuarial_pricing = ActuarialPricingAgent(gemini_client)
        self.report = ReportAgent(gemini_client)
        self.response = ResponseAgent(gemini_client)

    def run(self, raw_df, colmap, retention, limit, loading_pct, base_ccy, auto_cfg,
            log=None, tune_model=False):
        """
        log: optional callable(str) used to stream progress messages
             (e.g. st.write) to the caller as each agent finishes.
        tune_model: if True, run Optuna (50 trials) before the final fit.
        """
        def _log(msg):
            if log:
                log(msg)

        _log(f"🧹 [{self.data_analyst.name}] Cleaning data & selecting features…")
        da_result = self.data_analyst.run(raw_df, colmap, retention, limit, auto_cfg)
        da_insight = self.data_analyst.narrate(da_result, auto_cfg)

        best_params = None
        if tune_model:
            _log("🔍 [Optuna] Tuning hyperparameters (50 trials each — ~90 s on Colab T4)…")
            X_tr, _, yf_tr, _, Xs_tr, _, ys_tr, _ = da_result["splits"]
            best_params = tune_freq_severity(
                X_tr, yf_tr, Xs_tr, ys_tr, n_trials=50)
            _log(
                f"✅ [Optuna] Best freq params: n_est={best_params['freq']['n_estimators']}, "
                f"depth={best_params['freq']['max_depth']}, lr={best_params['freq']['learning_rate']:.4f} | "
                f"Best sev params: n_est={best_params['sev']['n_estimators']}, "
                f"depth={best_params['sev']['max_depth']}, lr={best_params['sev']['learning_rate']:.4f}"
            )

        _log(f"🤖 [{self.ml_pricing.name}] Training Poisson (frequency) + Gamma (severity) models…")
        ml_result = self.ml_pricing.run(
            da_result["feature_cols"], da_result["splits"], best_params=best_params)
        ml_insight = self.ml_pricing.narrate(ml_result)

        _log(f"💰 [{self.actuarial_pricing.name}] Computing Pure Premium & Gross Premium…")
        df_priced, ps, notes3 = self.actuarial_pricing.run(
            da_result["df"], da_result["feature_cols"],
            ml_result["freq_model"], ml_result["sev_model"],
            colmap, retention, limit, loading_pct, base_ccy)
        ap_insight = self.actuarial_pricing.narrate(ps)

        all_notes = da_result["notes"] + notes3

        _log(f"📊 [{self.report.name}] Generating charts & report…")
        charts = self.report.run(
            df_priced, colmap, auto_cfg["labels"], ml_result["fi"], ps,
            retention, limit, loading_pct, base_ccy)
        report_insight = self.report.narrate(ps, all_notes)

        return dict(
            df_priced=df_priced, ps=ps, notes=all_notes,
            feature_cols=da_result["feature_cols"],
            freq_metrics=ml_result["freq_metrics"], sev_metrics=ml_result["sev_metrics"],
            fi=ml_result["fi"], charts=charts,
            insights={
                "data_analyst": da_insight,
                "ml_pricing": ml_insight,
                "actuarial_pricing": ap_insight,
                "report": report_insight,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3 — app.py  (Streamlit dashboard UI)
# ═══════════════════════════════════════════════════════════════════════

import streamlit as st

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MarineReAI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  Futuristic CSS Injection
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=optional');

/* ── Root palette ── */
:root {
    --bg-base:      #050a14;
    --bg-surface:   #0a1628;
    --bg-card:      #0d1f3c;
    --border:       rgba(0,200,255,0.15);
    --border-glow:  rgba(0,200,255,0.5);
    --cyan:         #00c8ff;
    --cyan-dim:     rgba(0,200,255,0.12);
    --violet:       #7c3aed;
    --violet-dim:   rgba(124,58,237,0.15);
    --green:        #00e5a0;
    --amber:        #f59e0b;
    --red:          #f43f5e;
    --text-primary: #e8f4ff;
    --text-muted:   #6b8cad;
    --text-code:    #00c8ff;
    --radius:       12px;
    --radius-lg:    18px;
}

/* ── Base & body ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #050a14 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span:not(.material-icons):not(.material-symbols-rounded):not([data-testid]),
[data-testid="stSidebar"] div:not([class*="css"]) { font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

/* ── Sidebar inputs ── */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select {
    background: rgba(0,200,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 2px rgba(0,200,255,0.1) !important;
}

/* ── Sidebar labels & text ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: var(--text-primary) !important;
}
/* Only target semantic text spans, not icon spans */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .sidebar-logo-sub { color: var(--text-muted) !important; }

/* ── Sidebar radio buttons ── */
[data-testid="stSidebar"] [data-testid="stRadio"] label { color: var(--text-primary) !important; }

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--cyan) !important;
    border-color: var(--cyan) !important;
}
.stSlider .st-bd { background: var(--cyan) !important; }

/* ── Main text ── */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] div {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
/* Streamlit body text - avoid icon spans */
.stApp p, .stApp label { color: var(--text-primary) !important; }

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #07111f 0%, #0a1a33 50%, #06101e 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,200,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 300px; height: 150px;
    background: radial-gradient(ellipse, rgba(124,58,237,0.06) 0%, transparent 70%);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #e8f4ff 0%, #00c8ff 60%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero-subtitle {
    color: var(--text-muted) !important;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--cyan-dim);
    border: 1px solid rgba(0,200,255,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--cyan) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.pulse-dot {
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.4); }
}

/* ── Glassmorphism metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: linear-gradient(135deg, rgba(13,31,60,0.9) 0%, rgba(10,22,40,0.95) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: 0.6;
}
.metric-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted) !important;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--cyan) !important;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.metric-sub {
    font-size: 0.75rem;
    color: var(--text-muted) !important;
}
.metric-icon {
    position: absolute;
    top: 1rem; right: 1rem;
    font-size: 1.4rem;
    opacity: 0.25;
}

/* ── Data preview table & column-mapping grid ──
   Plain HTML (not st.dataframe / st.table) so the global
   `div,span,p{color/font-family !important}` rule above can't collide with
   Streamlit's internal virtualized-grid positioning — that collision is
   what was causing rows to render on top of each other. `contain: layout`
   gives each block its own layout/paint boundary so nothing here can
   affect, or be affected by, sibling sections; no position:sticky is used
   since that's the other common source of this kind of overlap glitch
   inside collapsible/animated containers like st.expander. */
.data-preview-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: auto;
    max-height: 420px;
    contain: layout paint;
    isolation: isolate;
}
.data-preview-wrap table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem;
    white-space: nowrap;
}
.data-preview-wrap thead th {
    background: #0a1a33 !important;
    color: var(--cyan) !important;
    text-align: left;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
}
.data-preview-wrap tbody td {
    padding: 6px 14px;
    color: var(--text-primary) !important;
    border-bottom: 1px solid rgba(0,200,255,0.06);
}
.data-preview-wrap tbody tr:nth-child(even) { background: rgba(0,200,255,0.025); }
.data-preview-wrap tbody tr:hover { background: rgba(0,200,255,0.07); }

.colmap-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 0.6rem;
    contain: layout paint;
}
.colmap-chip {
    background: rgba(0,200,255,0.04);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    min-width: 0;
}
.colmap-chip.unmapped { opacity: 0.45; }
.colmap-role {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted) !important;
    margin-bottom: 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.colmap-col {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem;
    color: var(--cyan) !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.75rem 0 0.75rem 0;
}
.section-line {
    height: 1px;
    flex: 1;
    background: linear-gradient(90deg, var(--border), transparent);
}
.section-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted) !important;
    white-space: nowrap;
}

/* ── Info / success / warning banners ── */
.banner {
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 0.88rem;
    line-height: 1.5;
}
.banner-info {
    background: rgba(0,200,255,0.06);
    border: 1px solid rgba(0,200,255,0.2);
    color: #a8d8f0 !important;
}
.banner-success {
    background: rgba(0,229,160,0.06);
    border: 1px solid rgba(0,229,160,0.2);
    color: #7de8c4 !important;
}
.banner-warn {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    color: #fbbf6a !important;
}
.banner-icon { font-size: 1rem; margin-top: 1px; flex-shrink: 0; }

/* ── Data table ── */
[data-testid="stDataFrame"], .stDataFrame {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.75rem;
    contain: layout paint;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}
[data-testid="stExpander"] summary p {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stExpander"] svg { fill: var(--cyan) !important; }
/* Hide the icon text fallback that bleeds in some Streamlit builds */
[data-testid="stExpander"] summary button span[class*="icon"],
[data-testid="stExpander"] summary [data-testid*="Icon"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    font-size: 1.2rem !important;
    line-height: 1 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,200,255,0.15), rgba(124,58,237,0.1)) !important;
    color: var(--cyan) !important;
    border: 1px solid rgba(0,200,255,0.25) !important;
}

/* ── Status / progress ── */
[data-testid="stStatus"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stStatus"] summary {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
}
[data-testid="stStatus"] summary span {
    font-family: 'Space Grotesk', sans-serif !important;
}
/* Prevent icon ligature from showing as text in status */
[data-testid="stStatus"] summary [data-testid*="Icon"],
[data-testid="stStatus"] summary span[class*="icon"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,200,255,0.12) 0%, rgba(124,58,237,0.1) 100%) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    color: var(--cyan) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,200,255,0.22) 0%, rgba(124,58,237,0.18) 100%) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.15) !important;
    transform: translateY(-1px) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(0,229,160,0.1) 0%, rgba(0,200,255,0.08) 100%) !important;
    border: 1px solid rgba(0,229,160,0.3) !important;
    color: var(--green) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    width: 100% !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--green) !important;
    box-shadow: 0 0 20px rgba(0,229,160,0.12) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(0,200,255,0.04) !important;
    border: 1px dashed rgba(0,200,255,0.25) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--border-glow) !important;
    background: rgba(0,200,255,0.07) !important;
}
/* Fix: prevent the collapsed label from reappearing via the broad color rule */
[data-testid="stFileUploader"] label[data-testid="stWidgetLabel"] {
    display: none !important;
}
/* Fix: style the upload button text cleanly */
[data-testid="stFileUploaderDropzone"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-muted) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
}

/* ── Text inputs (main area) ── */
.stTextInput input {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Chat ── */
[data-testid="stChatMessageContent"] {
    background: rgba(13,31,60,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.9rem;
}
[data-testid="stChatInput"] textarea {
    background: rgba(0,200,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Notes / bullet list ── */
.notes-list {
    background: rgba(13,31,60,0.6);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
}
.note-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 5px 0;
    border-bottom: 1px solid rgba(0,200,255,0.05);
    font-size: 0.86rem;
    color: #a8c4d8 !important;
    line-height: 1.4;
}
.note-item:last-child { border-bottom: none; }
.note-dot {
    color: var(--cyan) !important;
    font-size: 0.9rem;
    margin-top: 1px;
    flex-shrink: 0;
}

/* ── Chart containers ── */
.stPlotlyChart, .stPyplot {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Sidebar title ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.sidebar-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, rgba(0,200,255,0.2), rgba(124,58,237,0.2));
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}
.sidebar-logo-text {
    font-size: 1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e8f4ff, #00c8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-logo-sub {
    font-size: 0.7rem;
    color: var(--text-muted) !important;
}
.sidebar-section {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--cyan) !important;
    margin: 1.25rem 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,200,255,0.4); }

/* ── Metric (native st.metric) override ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(13,31,60,0.9), rgba(10,22,40,0.95)) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--cyan) !important;
    font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Info/success/warning native elements override ── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    font-size: 0.88rem;
}

/* ── Divider / HR ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────
if "priced" not in st.session_state:
    st.session_state.priced = False
if "insights" not in st.session_state:
    st.session_state.insights = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

NONE_OPTION = "— none —"

def options_with_none(cols):
    return [NONE_OPTION] + list(cols)

def resolve(choice):
    return None if choice == NONE_OPTION else choice


# ─────────────────────────────────────────────────────────────
#  Column mapping helpers
# ─────────────────────────────────────────────────────────────
ROLE_KEYWORDS = {
    "sum_insured":      ["sum_insured", "sum insured", "policy_limit", "policy limit",
                         "tsi", "insured_value", "insured value", "si", "limit", "coverage"],
    "exposure_value":   ["exposure_value", "exposure value", "exposure", "asset_value",
                         "asset value", "property_value", "insured_amount"],
    "deductible":       ["deductible", "excess", "retention", "ded", "self_insured_retention"],
    "deductible_pct":   ["deductible_pct", "deductible_percent", "excess_pct", "ded_pct"],
    "premium":          ["premium", "gwp", "written_premium", "earned_premium",
                         "gross_premium", "net_premium", "price"],
    "claims_count":     ["claims_count", "claim_count", "num_claims", "number_of_claims",
                         "claimcount", "freq", "frequency", "incurred_count"],
    "coastal_distance":  ["coastal_distance_km", "coastal_distance", "distance_to_coast",
                          "coast_dist_km", "coast_distance"],
    "claim_amount":     ["claims_historical_usd", "claims_historical", "claim_amount",
                         "claim_amount_usd", "claim_size", "claim_value", "claim_severity",
                         "incurred_amount", "incurred_loss", "loss_amount", "losses",
                         "ground_up_loss", "indemnity", "paid_amount", "paid_loss",
                         "gross_incurred", "claim_cost"],
    "risk_score":       ["risk_score", "risk score", "hazard_score", "hazard score",
                         "score", "risk_rating", "risk_index"],
    "protective_factor":["protective_factor", "protective factor", "safety_score",
                         "ncb", "no_claim_bonus", "credit_score", "discount_factor"],
    "unit_count":       ["unit_count", "unit count", "num_units", "number_of_units",
                         "crew", "drivers", "lives", "employees", "headcount", "units"],
    "age_years":        ["age_years", "age years", "age", "vehicle_age", "building_age",
                         "asset_age", "years_old", "policy_age"],
    "geography":        ["geography", "region", "location", "state", "country",
                         "zone", "territory", "area", "district", "city"],
    "category":         ["category", "peril", "cause", "class", "product_type",
                         "policy_type", "line_of_business", "lob", "cover_type"],
    "entity_type":      ["entity_type", "entity type", "vehicle_type", "vessel_type",
                         "occupancy", "construction_type", "building_type", "segment"],
}

import re

def _tokenize(name):
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]

def auto_map_columns(df_cols):
    col_tokens = {c: set(_tokenize(c)) for c in df_cols}
    used = set()
    colmap = {}
    for role, keywords in ROLE_KEYWORDS.items():
        matched = None
        for kw in keywords:
            kw_norm = kw.lower().replace("-", "_").replace(" ", "_")
            for c in df_cols:
                if c in used: continue
                c_norm = c.lower().replace("-", "_").replace(" ", "_")
                if c_norm == kw_norm:
                    matched = c
                    break
            if matched: break
        if not matched:
            for kw in keywords:
                kw_tokens = set(_tokenize(kw))
                if not kw_tokens: continue
                for c in df_cols:
                    if c in used: continue
                    if kw_tokens <= col_tokens[c]:
                        matched = c
                        break
                if matched: break
        if matched: used.add(matched)
        colmap[role] = matched
    colmap["categorical"] = [
        v for v in [colmap.get("geography"), colmap.get("category"), colmap.get("entity_type")]
        if v
    ]
    colmap["seasonal_flags"] = []
    return colmap


def pick_target(colmap, df_cols, role_preference, fallback_roles, numeric_cols):
    numeric_set = set(numeric_cols)
    pref = colmap.get(role_preference)
    if pref and pref in df_cols and pref in numeric_set:
        return pref
    for r in fallback_roles:
        v = colmap.get(r)
        if v and v in df_cols and v in numeric_set:
            return v
    if numeric_cols:
        return numeric_cols[0]
    raise ValueError(
        "No numeric columns were found in the dataset to use as a modelling target."
    )


def build_auto_config(raw_df, business_line, client_name, base_ccy,
                      retention, limit, loading_pct):
    all_cols = list(raw_df.columns)
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(raw_df[c])]
    colmap = auto_map_columns(all_cols)
    freq_target = pick_target(colmap, all_cols, "claims_count", ["premium"], numeric_cols)
    sev_target  = pick_target(colmap, all_cols, "claim_amount", ["premium"], numeric_cols)
    if freq_target == sev_target:
        alt = next((c for c in numeric_cols if c != freq_target), None)
        if alt is None:
            raise ValueError("Could not find two distinct numeric columns for freq/severity targets.")
        sev_target = alt
    labels = {
        "exposure": "Exposure Value", "sum_insured": "Sum Insured",
        "risk_score": "Risk Score", "category": "Category",
        "geography": "Geography", "entity_type": "Entity Type",
        "unit_count": "Unit Count", "premium": "Premium",
    }
    return dict(
        colmap=colmap, labels=labels,
        freq_target=freq_target, sev_target=sev_target,
        exclude_targets=True,
        retention=retention, limit=limit,
        loading_pct=loading_pct,
        business_line=business_line, client_name=client_name,
        base_ccy=base_ccy,
    )


# ─────────────────────────────────────────────────────────────
#  Sidebar — futuristic layout
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">⬡</div>
        <div>
            <div class="sidebar-logo-text">MarineReAI</div>
            <div class="sidebar-logo-sub">Frequency-Severity Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Portfolio</div>', unsafe_allow_html=True)
    business_line = st.text_input(
        "Business line",
        value="General Actuarial Pricing",
        help="e.g. Marine Hull, Marine Cargo, Property"
    )
    client_name = st.text_input(
        "Client / entity name",
        value="Indian Marine Insurance"
    )

    st.markdown('<div class="sidebar-section">Currency</div>', unsafe_allow_html=True)
    # Auto-detect currency from uploaded CSV if available, else let user pick
    _ccy_options = ["USD M", "USD", "GBP M", "GBP", "EUR M", "EUR", "INR Cr", "SGD M"]
    base_ccy = st.selectbox(
        "Currency / unit",
        options=_ccy_options,
        index=0,
        help="Select the currency and scale used in your CSV (e.g. USD M = millions of USD)."
    )

    st.markdown('<div class="sidebar-section">Pricing Structure</div>', unsafe_allow_html=True)
    pricing_mode_choice = st.radio(
        "Pricing mode",
        options=["Excess-of-Loss Layer", "Ground-Up / Primary"],
        index=0,
        horizontal=True,
    )
    if pricing_mode_choice == "Excess-of-Loss Layer":
        retention = st.number_input(
            "Retention / attachment point",
            min_value=0.0, value=30.0, step=5.0,
            help="Per-risk or per-occurrence retention in the same unit as your CSV (e.g. M USD)."
        )
        limit = st.number_input(
            "Limit (layer width)",
            min_value=1.0, value=30.0, step=5.0,
            help="Layer width above the retention."
        )
    else:
        retention = None
        limit = 0.0
    loading_pct = st.slider(
        "Expense + profit loading (%)",
        min_value=0, max_value=60, value=25, step=1,
        format="%d%%",
        help="Applied to pure premium to get gross premium."
    ) / 100.0

    st.markdown('<div class="sidebar-section">AI Assistant</div>', unsafe_allow_html=True)
    gemini_key = st.text_input(
        "Gemini API key (optional)",
        type="password",
        help="Only needed for the Ask AI tab.",
        label_visibility="visible"
    )

    st.markdown('<div class="sidebar-section">Model Options</div>', unsafe_allow_html=True)
    tune_model = st.checkbox(
        "Tune model (Optuna — 50 trials)",
        value=False,
        help=(
            "Run a 50-trial Optuna hyperparameter search on both the frequency "
            "and severity models before the final fit. Adds ~90 s on Colab T4. "
            "Recommended for final pricing runs; skip for quick exploration."
        ),
    )

    st.markdown('<div class="sidebar-section">Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"], label_visibility="collapsed")


# ─────────────────────────────────────────────────────────────
#  Main area — hero header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge"><span class="pulse-dot"></span>Live Pricing Engine</div>
    <div class="hero-title">MarineReAI</div>
    <p class="hero-subtitle">
        Upload any portfolio CSV — automatic column detection, XGBoost frequency-severity modelling,
        and instant actuarial pricing across any line of business.
    </p>
</div>
""", unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div class="banner banner-info">
        <span class="banner-icon">⬅</span>
        <span>Upload a portfolio CSV in the sidebar to begin. Motor, property, health, marine,
        liability, crop, cyber — any line of business. Column roles are detected automatically.</span>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🤖</div>
            <div class="metric-label">ML Engine</div>
            <div class="metric-value" style="font-size:1rem; color:#a78bfa !important;">XGBoost</div>
            <div class="metric-sub">Poisson frequency + Gamma severity</div>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">Automation</div>
            <div class="metric-value" style="font-size:1rem; color:#00e5a0 !important;">Auto-Map</div>
            <div class="metric-sub">Zero manual column configuration</div>
        </div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📄</div>
            <div class="metric-label">Output</div>
            <div class="metric-value" style="font-size:1rem; color:#f59e0b !important;">PDF + CSV</div>
            <div class="metric-sub">Full report + priced dataset export</div>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────────────────────
raw_df    = pd.read_csv(uploaded)
all_cols  = list(raw_df.columns)
file_name = uploaded.name

st.markdown(f"""
<div class="banner banner-success">
    <span class="banner-icon">✓</span>
    <span>Loaded <strong>{raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns</strong>
    from <code style="background:rgba(0,200,255,0.1);padding:1px 6px;border-radius:4px;font-size:0.82rem;">{file_name}</code></span>
</div>
""", unsafe_allow_html=True)

with st.expander("Preview raw data", expanded=False):
    _preview_df = raw_df.head(20)
    st.markdown(
        f'<div class="data-preview-wrap">{_preview_df.to_html(index=False, border=0, escape=True)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Showing first {len(_preview_df)} of {raw_df.shape[0]:,} rows.")

# ── Auto-detect column mapping ───────────────────────────────
auto_cfg = build_auto_config(raw_df, business_line, client_name, base_ccy,
                              retention, limit, loading_pct)
colmap   = auto_cfg["colmap"]

with st.expander("Column mapping — auto-detected (expand to review)", expanded=False):
    role_labels = {
        "sum_insured": "Sum Insured / Policy Limit",
        "exposure_value": "Exposure Value",
        "deductible": "Deductible / Excess",
        "deductible_pct": "Deductible %",
        "premium": "Premium (historical)",
        "claims_count": "Claims Count",
        "claim_amount": "Claim Amount (severity)",
        "risk_score": "Risk Score",
        "protective_factor": "Protective Factor",
        "unit_count": "Unit Count",
        "age_years": "Age (years)",
        "geography": "Geography",
        "category": "Category",
        "entity_type": "Entity Type",
        "categorical": "Categorical features",
        "seasonal_flags": "Seasonal / binary flags",
    }
    chips = []
    for role, label in role_labels.items():
        v = colmap.get(role)
        if isinstance(v, list):
            v = ", ".join(v) if v else None
        unmapped_cls = "" if v else " unmapped"
        chips.append(
            f'<div class="colmap-chip{unmapped_cls}">'
            f'<div class="colmap-role">{label}</div>'
            f'<div class="colmap-col">{v or "— not detected —"}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="colmap-grid">{"".join(chips)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-sub" style="margin-top:0.5rem;">'
        f'Frequency target: <code style="color:var(--cyan)">{auto_cfg["freq_target"]}</code> &nbsp;|&nbsp; '
        f'Severity target: <code style="color:#a78bfa">{auto_cfg["sev_target"]}</code>'
        f'</div>', unsafe_allow_html=True
    )


# ── Run pipeline ─────────────────────────────────────────────
new_file = (file_name != st.session_state.last_uploaded_name)
if new_file:
    st.session_state.priced = False
    st.session_state.last_uploaded_name = file_name

# Invalidate cached results if any pricing parameter changed
_current_cfg_key = (retention, limit, loading_pct, business_line, base_ccy, tune_model)
if st.session_state.get("last_cfg_key") != _current_cfg_key:
    st.session_state.priced = False
    st.session_state["last_cfg_key"] = _current_cfg_key

# ── Run Pricing button ────────────────────────────────────────
# Pipeline only runs when the user explicitly clicks the button.
# Once results are cached in session_state, the button re-appears
# as "Re-run Pricing" so the user can trigger a fresh run at any time.
_btn_label = "🔄  Re-run Pricing" if st.session_state.priced else "▶  Run Pricing"
_run_clicked = st.button(
    _btn_label,
    type="primary",
    use_container_width=True,
    help="Run the full 4-agent pricing pipeline on the uploaded CSV with the current settings.",
)

if _run_clicked:
    st.session_state.priced = False  # force fresh run on each click

if _run_clicked and not st.session_state.priced:
    with st.status("Running MarineReAI multi-agent pipeline…", expanded=True) as status:
        try:
            gemini_client = get_gemini_client(gemini_key)
            pipeline = MarineReAIPipeline(gemini_client)

            # ── Agents 1-4 run end-to-end via the orchestrator ───────
            # Agent 1 – Data Analyst       : clean data, engineer/select
            #                                features, prepare train/test splits.
            # Agent 2 – ML Pricing         : XGBoost Poisson (frequency) +
            #                                XGBoost Gamma (severity) models.
            # Agent 3 – Actuarial Pricing  : Pure Premium, Gross Premium, RoL,
            #                                applying the XoL layer if configured.
            # Agent 4 – Report & Visualization : charts + AI "Key Findings" text
            #                                     used in the PDF report.
            # Each agent calls Gemini 2.5 Flash to turn its structured output
            # into a short plain-English insight (falls back to a rule-based
            # sentence if no Gemini API key is supplied).
            result = pipeline.run(
                raw_df, colmap, retention, limit, loading_pct, base_ccy, auto_cfg,
                log=st.write,
                tune_model=tune_model,
            )

            st.session_state.update(dict(
                priced=True,
                df_priced=result["df_priced"], ps=result["ps"],
                notes=result["notes"],
                freq_metrics=result["freq_metrics"], sev_metrics=result["sev_metrics"],
                fi=result["fi"], charts=result["charts"],
                feature_cols=result["feature_cols"],
                config=auto_cfg,
                insights=result["insights"],
            ))
            status.update(label="✅ Pricing complete!", state="complete", expanded=False)

        except Exception as e:
            status.update(label="❌ Pricing failed", state="error")
            st.markdown(f"""
            <div class="banner banner-warn">
                <span class="banner-icon">⚠</span>
                <span>Pricing failed: {e}</span>
            </div>""", unsafe_allow_html=True)
            st.exception(e)
            st.stop()


# ─────────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────────
tab_results, tab_ask = st.tabs(["📈  Results & Report", "🤖  Ask AI"])


# ───────────── Results tab ───────────────────────────────────
with tab_results:
    if not st.session_state.get("priced"):
        st.markdown('''
        <div class="banner banner-info">
            <span class="banner-icon">▶</span>
            <span>Click <strong>Run Pricing</strong> above to run the pipeline and see results here.</span>
        </div>
        ''', unsafe_allow_html=True)
    else:
        cfg      = st.session_state.config
        ps       = st.session_state.ps
        base_ccy = cfg["base_ccy"]

        # ── Section: Pricing Summary ──
        st.markdown("""
        <div class="section-header">
        <span class="section-title">Executive Pricing Summary</span>
        <div class="section-line"></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-icon">💠</div>
            <div class="metric-label">Pure Premium</div>
            <div class="metric-value">{ps['pure_premium']:,.4f}</div>
            <div class="metric-sub">{base_ccy}</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">💰</div>
            <div class="metric-label">Gross Premium</div>
            <div class="metric-value">{ps['gross_premium']:,.4f}</div>
            <div class="metric-sub">{base_ccy} incl. {ps['loading_pct']*100:.0f}% loading + CAT load</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📐</div>
            <div class="metric-label">Rate on Line</div>
            <div class="metric-value">{ps['rol_on_limit']*100:.4f}%</div>
            <div class="metric-sub">of limit</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">⬡</div>
            <div class="metric-label">Pricing Mode</div>
            <div class="metric-value" style="font-size:0.95rem;">{ps['pricing_mode']}</div>
            <div class="metric-sub">Active structure</div>
        </div>
        {f'''<div class="metric-card" style="border-color:rgba(63,185,80,0.4)">
            <div class="metric-icon">⚖</div>
            <div class="metric-label">Loss Ratio Check</div>
            <div class="metric-value" style="font-size:1rem;">{ps["hist_loss_ratio"]*100:.1f}% hist</div>
            <div class="metric-sub">Model-implied: {ps["model_loss_ratio"]*100:.1f}% of gross premium</div>
        </div>''' if ps.get("hist_loss_ratio") is not None else ''}
        {f'''<div class="metric-card" style="border-color:rgba(247,129,102,0.4)">
            <div class="metric-icon">🌀</div>
            <div class="metric-label">CAT Loading</div>
            <div class="metric-value" style="font-size:1rem;">{ps["cat_loading_pct"]*100:.1f}%</div>
            <div class="metric-sub">Cyclone correlation (5k sims, 90th pct)</div>
        </div>''' if ps.get("cat_loading_amount", 0) > 0 else ''}
        </div>
        """, unsafe_allow_html=True)

        # ── Section: Charts ──
        st.markdown("""
        <div class="section-header">
        <span class="section-title">Analytics & Charts</span>
        <div class="section-line"></div>
        </div>""", unsafe_allow_html=True)

        charts = st.session_state.charts
        chart_items = [(k, v) for k, v in charts.items() if v is not None]

        for i in range(0, len(chart_items), 2):
            cols = st.columns(2)
            for j, col_ui in enumerate(cols):
                if i + j < len(chart_items):
                    name, fig = chart_items[i + j]
                    with col_ui:
                        st.markdown(f'<div class="metric-label" style="margin-bottom:0.4rem;">{name}</div>', unsafe_allow_html=True)
                        st.pyplot(fig, use_container_width=True)

        # ── Section: Actuarial Notes ──
        st.markdown("""
        <div class="section-header">
        <span class="section-title">Actuarial Notes</span>
        <div class="section-line"></div>
        </div>""", unsafe_allow_html=True)

        notes_html = '<div class="notes-list">' + "".join(
        f'<div class="note-item"><span class="note-dot">›</span><span>{n}</span></div>'
        for n in st.session_state.notes
        ) + '</div>'
        st.markdown(notes_html, unsafe_allow_html=True)

        # ── Section: Model Validation (Improvement 2) ──────────────────────────
        st.markdown("""
        <div class="section-header">
        <span class="section-title">Model Validation</span>
        <div class="section-line"></div>
        </div>""", unsafe_allow_html=True)

        fm = st.session_state.freq_metrics
        sm = st.session_state.sev_metrics
        cv_freq_str = f"{fm['CV_RMSE']:.4f}" if "CV_RMSE" in fm else "—"
        cv_sev_str  = f"{sm['CV_RMSE']:,.0f}" if "CV_RMSE" in sm else "—"
        sev_rmsle_str     = f"{sm['RMSLE']:.4f}"     if "RMSLE"    in sm else "—"
        sev_cv_rmsle_str  = f"{sm['CV_RMSLE']:.4f}"  if "CV_RMSLE" in sm else "—"
        sev_rel_rmse_str  = f"{sm['Relative_RMSE_pct']:.1f}%" if "Relative_RMSE_pct" in sm else "—"
        freq_rel_rmse_str = f"{fm['Relative_RMSE_pct']:.1f}%" if "Relative_RMSE_pct" in fm else "—"
        sev_mean_str = f"${sm['Mean_Actual']:,.0f}" if "Mean_Actual" in sm else "—"
        st.markdown(f"""
        <div class="metric-grid" style="grid-template-columns:repeat(4,1fr);">
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-label">Freq Hold-out RMSE</div>
            <div class="metric-value" style="font-size:1.1rem;">{fm['RMSE']:.4f}</div>
            <div class="metric-sub">Relative: {freq_rel_rmse_str} of mean</div>
        </div>
        <div class="metric-card" style="border-color:rgba(88,166,255,0.4);">
            <div class="metric-icon">🔁</div>
            <div class="metric-label">Freq 5-Fold CV RMSE</div>
            <div class="metric-value" style="font-size:1.1rem;">{cv_freq_str}</div>
            <div class="metric-sub">More stable than single split</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-label">Sev Hold-out RMSE (USD)</div>
            <div class="metric-value" style="font-size:1.1rem;">{sm['RMSE']:,.0f}</div>
            <div class="metric-sub">Mean actual: {sev_mean_str} · Rel: {sev_rel_rmse_str}</div>
        </div>
        <div class="metric-card" style="border-color:rgba(88,166,255,0.4);">
            <div class="metric-icon">🔁</div>
            <div class="metric-label">Sev 5-Fold CV RMSE (USD)</div>
            <div class="metric-value" style="font-size:1.1rem;">{cv_sev_str}</div>
            <div class="metric-sub">More stable than single split</div>
        </div>
        </div>
        <div class="metric-grid" style="grid-template-columns:repeat(2,1fr);margin-top:10px;">
        <div class="metric-card" style="border-color:rgba(63,185,80,0.4);">
            <div class="metric-icon">📐</div>
            <div class="metric-label">Sev Hold-out RMSLE ✅</div>
            <div class="metric-value" style="font-size:1.3rem;color:#3fb950;">{sev_rmsle_str}</div>
            <div class="metric-sub">Log-space error — correct metric for Gamma model (lower = better, &lt;1.0 is good)</div>
        </div>
        <div class="metric-card" style="border-color:rgba(63,185,80,0.4);">
            <div class="metric-icon">🔁</div>
            <div class="metric-label">Sev 5-Fold CV RMSLE ✅</div>
            <div class="metric-value" style="font-size:1.3rem;color:#3fb950;">{sev_cv_rmsle_str}</div>
            <div class="metric-sub">CV RMSLE — most reliable severity metric</div>
        </div>
        </div>
        <div style="background:rgba(255,170,0,0.07);border:1px solid rgba(255,170,0,0.3);
                border-radius:8px;padding:10px 14px;margin-top:10px;font-size:0.78rem;
                color:#e6c84a;">
        ℹ️ <b>Why is Sev RMSE ~1M USD?</b> The severity model targets claim amounts with
        mean ~$1M and heavy right-tail skewness (3.2) and kurtosis (14.6). Dollar RMSE will
        always be large relative to frequency RMSE because the units are ~$1M vs ~0.14 claims.
        <b>RMSLE is the correct metric for this Gamma model</b> — it measures log-space error,
        consistent with the Gamma objective. A RMSLE &lt; 1.0 indicates good model fit.
        Relative RMSE ({sev_rel_rmse_str}) also provides proper context.
        </div>
        """, unsafe_allow_html=True)

        # ── Section: AI Agent Insights ──
        st.markdown("""
        <div class="section-header">
        <span class="section-title">AI Agent Insights</span>
        <div class="section-line"></div>
        </div>""", unsafe_allow_html=True)

        insights = st.session_state.get("insights", {})
        insight_cards = [
        ("🧹", "Agent 1 — Data Analyst", insights.get("data_analyst", "")),
        ("🤖", "Agent 2 — ML Pricing", insights.get("ml_pricing", "")),
        ("💰", "Agent 3 — Actuarial Pricing", insights.get("actuarial_pricing", "")),
        ("📊", "Agent 4 — Report & Visualization", insights.get("report", "")),
        ]
        for icon, title, text in insight_cards:
            st.markdown(f"""
            <div style="background:rgba(0,200,255,0.05);border:1px solid var(--border);
                        border-radius:10px;padding:12px 16px;margin-bottom:10px;">
                <div style="color:var(--cyan);font-size:0.78rem;text-transform:uppercase;
                            letter-spacing:0.06em;margin-bottom:4px;">{icon} {title}</div>
                <div style="color:var(--text);font-size:0.9rem;line-height:1.5;">{text}</div>
            </div>""", unsafe_allow_html=True)

        # ── Section: Downloads ──
        st.markdown("""
        <div class="section-header">
        <span class="section-title">Export</span>
        <div class="section-line"></div>
        </div>""", unsafe_allow_html=True)

        pdf_bytes = build_pdf_bytes(
        {"business_line": cfg["business_line"], "client_name": cfg["client_name"]},
        ps, st.session_state.notes,
        st.session_state.charts, cfg["base_ccy"],
        key_findings=st.session_state.get("insights", {}).get("report"))

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📄  Download PDF Report",
                data=pdf_bytes,
                file_name="ActuarialPricing_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with d2:
            # Drop the row-level prediction columns to remove frequency/severity/loss data from the CSV
            cols_to_remove = ['pred_freq', 'pred_sev', 'expected_loss', 'layer_factor', 'layer_expected_loss']
            export_df = st.session_state.df_priced.drop(columns=cols_to_remove, errors='ignore')
            # Fix 3: rename Loss_Ratio → Historical_Loss_Ratio so it's clear
            # this is from the original data, not a model output.
            if 'Loss_Ratio' in export_df.columns:
                export_df = export_df.rename(columns={'Loss_Ratio': 'Historical_Loss_Ratio'})

            csv_bytes = export_df.to_csv(index=False).encode()
            st.download_button(
                "📋  Download Priced Dataset (CSV)",
                data=csv_bytes,
                file_name="priced_portfolio.csv",
                mime="text/csv",
                use_container_width=True
            )


# ───────────── Ask AI tab ────────────────────────────────────
with tab_ask:
    st.markdown("""
    <div class="section-header">
        <span class="section-title">AI Assistant</span>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.get("priced"):
        st.markdown('''
        <div class="banner banner-info">
            <span class="banner-icon">▶</span>
            <span>Run Pricing first, then come back here to ask questions about the results.</span>
        </div>
        ''', unsafe_allow_html=True)
    elif not gemini_key:
        st.markdown("""
        <div class="banner banner-warn">
            <span class="banner-icon">🔑</span>
            <span>Enter a Gemini API key in the sidebar to enable the AI assistant.</span>
        </div>""", unsafe_allow_html=True)
    else:
        cfg = st.session_state.config
        ps  = st.session_state.ps
        fi  = st.session_state.fi
        system_ctx = f"""You are the MarineReAI Assistant for {cfg['business_line']}.

PRICING SUMMARY:
  Mode          : {ps['pricing_mode']}
  Pure Premium  : {cfg['base_ccy']} {ps['pure_premium']:.4f}
  Gross Premium : {cfg['base_ccy']} {ps['gross_premium']:.4f}
  Rate on Line  : {ps['rol_on_limit']*100:.4f}%
  Loading       : {ps['loading_pct']*100:.0f}%

MODEL DRIVERS:
  Top frequency drivers: {fi['frequency'].head(5).index.tolist()}
  Top severity drivers : {fi['severity'].head(5).index.tolist()}

NOTES:
{chr(10).join(st.session_state.notes)}

Answer concisely and accurately. Redirect off-topic questions back to this pricing analysis.
"""
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(msg)

        user_q = st.chat_input("Ask about pricing, model drivers, or data quality…")
        if user_q:
            st.session_state.chat_history.append(("user", user_q))
            with st.chat_message("user"):
                st.write(user_q)
            # ── AGENT 5: ResponseAgent — AI Q&A assistant ───────────
            # PURPOSE   : Answers natural-language questions about the pricing
            #             results using the Gemini 2.5 Flash LLM.
            # SDK COMPAT: Supports both the new google-genai SDK
            #             (Client / models.generate_content) and the legacy
            #             google-generativeai SDK (GenerativeModel.generate_content)
            #             via the shared shim in py.
            # CONTEXT   : Injects the full pricing summary (mode, pure premium,
            #             gross premium, RoL, top feature drivers, notes) as
            #             system context so answers are grounded in results.
            # HISTORY   : Prepends the conversation history to every prompt so
            #             Gemini can give coherent multi-turn responses — this
            #             is the agent's "memory" across the conversation.
            # FALLBACK  : Any SDK error is caught and shown as a bracketed
            #             error message rather than crashing the dashboard.
            gemini_client = get_gemini_client(gemini_key)
            response_agent = ResponseAgent(gemini_client)
            answer = response_agent.answer(user_q, st.session_state.chat_history, system_ctx)
            st.session_state.chat_history.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.write(answer)

