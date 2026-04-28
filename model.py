"""
contains feature preparation, lightgbm + xgboost training with optuna,
probability calibration, stacking meta-learner, threshold tuning, and shap.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
import shap
import joblib
import json
from datetime import datetime


# tuning constants
hpo_max_rows      = 80_000
hpo_cv_folds      = 3
full_cv_folds     = 5
trial_timeout_sec = 180

# minimum recall to enforce in recall_constrained strategy
min_recall_constraint = 0.75


# logging
def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


# data preparation
def prepare_features(df, label_col="HighCostLabel", drop_cols=None):
    always_drop = {
        "Member_Key", "YEAR", "MONTH",
        "NEXT_YEAR_COST", label_col
    }
    if drop_cols:
        always_drop.update(drop_cols)

    feature_cols = [c for c in df.columns if c not in always_drop]
    y = df[label_col].astype(int) if label_col in df.columns else None
    X = df[feature_cols].copy()

    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X = X.fillna(0)
    log(f"features prepared: {X.shape[1]} columns, {X.shape[0]} rows")
    return X, y, list(X.columns), encoders


def align_columns(X_train, X_test):
    missing = set(X_train.columns) - set(X_test.columns)
    extra   = set(X_test.columns)  - set(X_train.columns)

    if missing:
        log(f"test missing {len(missing)} columns — filling with 0")
        for col in missing:
            X_test[col] = 0
    if extra:
        log(f"test has {len(extra)} extra columns — dropping them")
        X_test = X_test.drop(columns=list(extra))

    return X_train, X_test[X_train.columns]


# checkpoint helpers
def _ckpt_path(output_dir, name):
    return os.path.join(output_dir, f"_ckpt_{name}")


def _save_ckpt(output_dir, name, obj):
    path = _ckpt_path(output_dir, name)
    if isinstance(obj, np.ndarray):
        np.save(path + ".npy", obj)
    else:
        with open(path + ".json", "w") as f:
            json.dump(obj, f)
    log(f"checkpoint saved → {os.path.basename(path)}")


def _load_ckpt_array(output_dir, name):
    path = _ckpt_path(output_dir, name) + ".npy"
    if os.path.exists(path):
        arr = np.load(path)
        log(f"loaded checkpoint: {os.path.basename(path)}")
        return arr
    return None


def _load_ckpt_json(output_dir, name):
    path = _ckpt_path(output_dir, name) + ".json"
    if os.path.exists(path):
        with open(path) as f:
            obj = json.load(f)
        log(f"loaded checkpoint: {os.path.basename(path)}")
        return obj
    return None


def _fold_ckpt_path(output_dir, model_name, fold_idx):
    return os.path.join(output_dir, f"_ckpt_{model_name}_fold{fold_idx}.npy")


def _save_fold_ckpt(output_dir, model_name, fold_idx, val_idx, val_proba):
    path = _fold_ckpt_path(output_dir, model_name, fold_idx)
    np.save(path, np.column_stack([val_idx, val_proba]))
    log(f"fold {fold_idx+1} checkpoint saved")


def _load_fold_ckpt(output_dir, model_name, fold_idx):
    path = _fold_ckpt_path(output_dir, model_name, fold_idx)
    if os.path.exists(path):
        data = np.load(path)
        val_idx = data[:, 0].astype(int)
        val_proba = data[:, 1]
        log(f"fold {fold_idx+1} already done — loading from checkpoint")
        return val_idx, val_proba
    return None, None


# threshold tuning
def tune_threshold(y_true, proba, strategy="f1"):
    thresholds = np.linspace(0.05, 0.95, 181)

    if strategy == "fixed":
        return 0.5

    best_thresh, best_score = 0.5, -1.0

    if strategy == "recall_constrained":
        log(f"threshold strategy: recall_constrained (min recall = {min_recall_constraint})")
        for t in thresholds:
            preds = (proba >= t).astype(int)
            rec = recall_score(y_true, preds, zero_division=0)
            if rec < min_recall_constraint:
                continue
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_score:
                best_score = f1
                best_thresh = t
        if best_score < 0:
            log("no threshold met recall constraint — using lowest threshold")
            best_thresh = thresholds[0]
    else:
        for t in thresholds:
            preds = (proba >= t).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = t

    log(f"best threshold: {best_thresh:.3f} (oof f1 = {best_score:.4f})")
    return best_thresh


# focal loss for lightgbm (optional)
def _focal_loss_lgbm(gamma=2.0, alpha=0.25):
    def focal_loss_obj(y_pred, data):
        y_true = data.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        fw_pos = alpha * (1 - p) ** gamma
        fw_neg = (1 - alpha) * p ** gamma
        grad = np.where(y_true == 1, fw_pos * (p - 1), fw_neg * p)
        hess = np.where(y_true == 1,
                        fw_pos * p * (1 - p) * (gamma * (1 - p) * np.log(p + 1e-7) + 1),
                        fw_neg * p * (1 - p) * (gamma * p * np.log(1 - p + 1e-7) + 1))
        return grad, np.abs(hess)
    return focal_loss_obj


# lightgbm fixed params
lgbm_fixed = dict(
    objective="binary",
    metric="average_precision",
    boosting_type="gbdt",
    n_jobs=-1,
    force_col_wise=True,
    random_state=42,
    verbose=-1,
    device="gpu",
)


def _lgbm_hpo_objective(trial, X_hpo, y_hpo, cv_hpo, scale_pos_weight):
    params = {
        **lgbm_fixed,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }
    scores = []
    for tr_idx, val_idx in cv_hpo.split(X_hpo, y_hpo):
        X_tr, X_val = X_hpo.iloc[tr_idx], X_hpo.iloc[val_idx]
        y_tr, y_val = y_hpo.iloc[tr_idx], y_hpo.iloc[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        proba = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, proba))
    return np.mean(scores)


def _subsample_for_hpo(X, y, max_rows, random_state=42):
    if len(X) <= max_rows:
        return X, y

    rng = np.random.default_rng(random_state)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    ratio = len(idx_1) / len(y)
    n1 = int(max_rows * ratio)
    n0 = max_rows - n1

    sel = np.concatenate([
        rng.choice(idx_0, size=min(n0, len(idx_0)), replace=False),
        rng.choice(idx_1, size=min(n1, len(idx_1)), replace=False),
    ])
    sel.sort()

    log(f"hpo subsample: {len(sel):,} rows ({n0:,} neg + {n1:,} pos)")
    return X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True)


def train_lgbm(X_train, y_train, n_trials=60, cv_folds=5, output_dir="output"):
    log("lightgbm — optuna tuning (optimising pr-auc)")

    pos = int(y_train.sum())
    neg = len(y_train) - pos
    spw = round(neg / pos, 4)
    log(f"class ratio neg:pos = {neg}:{pos} → scale_pos_weight = {spw}")

    X_hpo, y_hpo = _subsample_for_hpo(X_train, y_train, hpo_max_rows)
    cv_hpo = StratifiedKFold(n_splits=hpo_cv_folds, shuffle=True, random_state=42)

    db_path = os.path.join(output_dir, "optuna_studies1.db")
    storage = f"sqlite:///{db_path}"
    study_name = "lgbm_study_v2"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if completed > 0:
        log(f"resuming optuna: {completed} trials done, {remaining} remaining")

    if remaining > 0:
        study.optimize(
            lambda trial: _lgbm_hpo_objective(trial, X_hpo, y_hpo, cv_hpo, spw),
            n_trials=remaining,
            timeout=trial_timeout_sec * remaining,
            show_progress_bar=True,
        )

    best_params = {**lgbm_fixed, "scale_pos_weight": spw, **study.best_params}
    log(f"best pr-auc from hpo: {study.best_value:.5f}")

    _save_ckpt(output_dir, "lgbm_best_params",
               {k: v for k, v in best_params.items() if k not in ("device", "force_col_wise")})

    # full oof
    log(f"computing full oof predictions ({cv_folds}-fold)...")
    cv_full = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y_train))
    fold_models = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv_full.split(X_train, y_train)):
        ck_idx, ck_proba = _load_fold_ckpt(output_dir, "lgbm", fold_idx)
        if ck_idx is not None:
            oof_proba[ck_idx] = ck_proba
            fold_models.append(None)
            continue

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = lgb.LGBMClassifier(**best_params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        fold_proba = m.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = fold_proba
        fold_models.append(m)

        _save_fold_ckpt(output_dir, "lgbm", fold_idx, val_idx, fold_proba)
        log(f"fold {fold_idx+1} done — pr-auc: {average_precision_score(y_val, fold_proba):.5f}")

    # final model on full data
    log("training final lightgbm model on full training set...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, os.path.join(output_dir, "_interim_lgbm_model.pkl"))
    _save_ckpt(output_dir, "lgbm_oof_proba", oof_proba)

    log(f"oof auc: {roc_auc_score(y_train, oof_proba):.5f}")
    log(f"oof pr-auc: {average_precision_score(y_train, oof_proba):.5f}")

    return final_model, fold_models, oof_proba, best_params


# xgboost fixed params
xgb_fixed = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    device="cuda",
    nthread=-1,
    random_state=42,
    verbosity=0,
)


def _xgb_hpo_objective(trial, X_hpo, y_hpo, cv_hpo, scale_pos_weight):
    params = {
        **xgb_fixed,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    scores = []
    for tr_idx, val_idx in cv_hpo.split(X_hpo, y_hpo):
        X_tr, X_val = X_hpo.iloc[tr_idx], X_hpo.iloc[val_idx]
        y_tr, y_val = y_hpo.iloc[tr_idx], y_hpo.iloc[val_idx]
        m = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        proba = m.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, proba))
    return np.mean(scores)


def train_xgb(X_train, y_train, n_trials=40, cv_folds=5, output_dir="output"):
    log("xgboost — optuna tuning (optimising pr-auc)")

    pos = int(y_train.sum())
    neg = len(y_train) - pos
    spw = round(neg / pos, 4)

    X_hpo, y_hpo = _subsample_for_hpo(X_train, y_train, hpo_max_rows)
    cv_hpo = StratifiedKFold(n_splits=hpo_cv_folds, shuffle=True, random_state=42)

    db_path = os.path.join(output_dir, "optuna_studies.db")
    storage = f"sqlite:///{db_path}"
    study_name = "xgb_study_v2"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if completed > 0:
        log(f"resuming optuna: {completed} trials done, {remaining} remaining")

    if remaining > 0:
        study.optimize(
            lambda trial: _xgb_hpo_objective(trial, X_hpo, y_hpo, cv_hpo, spw),
            n_trials=remaining,
            timeout=trial_timeout_sec * remaining,
            show_progress_bar=True,
        )

    best_params = {**xgb_fixed, "scale_pos_weight": spw, **study.best_params}

    # full oof
    cv_full = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y_train))

    for fold_idx, (tr_idx, val_idx) in enumerate(cv_full.split(X_train, y_train)):
        ck_idx, ck_proba = _load_fold_ckpt(output_dir, "xgb", fold_idx)
        if ck_idx is not None:
            oof_proba[ck_idx] = ck_proba
            continue

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        m = xgb.XGBClassifier(**best_params, early_stopping_rounds=50)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        fold_proba = m.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = fold_proba

        _save_fold_ckpt(output_dir, "xgb", fold_idx, val_idx, fold_proba)

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, verbose=False)

    joblib.dump(final_model, os.path.join(output_dir, "_interim_xgb_model.pkl"))
    _save_ckpt(output_dir, "xgb_oof_proba", oof_proba)

    log(f"oof auc: {roc_auc_score(y_train, oof_proba):.5f}")
    log(f"oof pr-auc: {average_precision_score(y_train, oof_proba):.5f}")

    return final_model, oof_proba, best_params


# calibration wrapper
class CalibratedWrapper:
    def __init__(self, base_model, calibrator, method, feature_names=None):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method
        self.feature_names_ = feature_names

    def _align(self, X):
        if self.feature_names_ is None or not isinstance(X, pd.DataFrame):
            return X
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            X = X.copy()
            for col in missing:
                X[col] = 0
        return X[self.feature_names_]

    def predict_proba(self, X):
        X_aligned = self._align(X)
        raw = self.base_model.predict_proba(X_aligned)[:, 1]
        if self.method == "isotonic":
            cal = self.calibrator.transform(raw)
        else:
            cal = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1 - cal, cal])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def calibrate_probabilities(model, X_cal, y_cal,
                            method="isotonic", output_dir="output",
                            feature_names=None):
    log(f"calibrating probabilities using {method}...")

    # align columns to prevent feature mismatch
    if feature_names is not None and isinstance(X_cal, pd.DataFrame):
        missing = set(feature_names) - set(X_cal.columns)
        if missing:
            X_cal = X_cal.copy()
            for col in missing:
                X_cal[col] = 0
        X_cal = X_cal[feature_names]

    # raw probabilities (with checkpoint)
    raw_ckpt_path = os.path.join(output_dir, "_ckpt_calibration_raw.npy")
    if os.path.exists(raw_ckpt_path):
        raw_proba = np.load(raw_ckpt_path)
        log("loaded raw calibration probabilities from checkpoint")
    else:
        raw_proba = model.predict_proba(X_cal)[:, 1]
        np.save(raw_ckpt_path, raw_proba)
        log("raw calibration probabilities saved")

    y_cal_arr = y_cal.values if hasattr(y_cal, "values") else np.asarray(y_cal)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_proba, y_cal_arr)
    else:
        calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        calibrator.fit(raw_proba.reshape(-1, 1), y_cal_arr)

    wrapped = CalibratedWrapper(model, calibrator, method, feature_names=feature_names)

    joblib.dump(wrapped, os.path.join(output_dir, "_calibrated_lgbm.pkl"))
    log("calibrated model saved")

    return wrapped


# stacking meta-learner
def train_stacking_meta(lgbm_oof, xgb_oof, y_train, output_dir="output"):
    log("training stacking meta-learner (logistic regression)...")

    X_meta = np.column_stack([lgbm_oof, xgb_oof])
    y_meta = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(y_meta))

    for tr_idx, val_idx in cv.split(X_meta, y_meta):
        meta_model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        meta_model.fit(X_meta[tr_idx], y_meta[tr_idx])
        meta_oof[val_idx] = meta_model.predict_proba(X_meta[val_idx])[:, 1]

    final_meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    final_meta.fit(X_meta, y_meta)

    joblib.dump(final_meta, os.path.join(output_dir, "_meta_model.pkl"))
    log("stacking meta model saved")

    return final_meta, meta_oof


def predict_stacking_meta(meta_model, lgbm_proba, xgb_proba):
    X_meta = np.column_stack([lgbm_proba, xgb_proba])
    return meta_model.predict_proba(X_meta)[:, 1]


# simple rank average ensemble
def rank_average(proba_list):
    n = len(proba_list[0])
    ranks = np.zeros(n)
    for p in proba_list:
        order = np.argsort(p)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, n + 1) / n
        ranks += r
    return ranks / len(proba_list)


# evaluation
def full_report(y_true, proba, threshold, label=""):
    preds = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    apr = average_precision_score(y_true, proba)
    f1 = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)

    log(f"\n--- {label} ---")
    log(f"auc       : {auc:.5f}")
    log(f"pr-auc    : {apr:.5f}")
    log(f"f1        : {f1:.5f}")
    log(f"precision : {prec:.5f}")
    log(f"recall    : {rec:.5f}")
    log(f"threshold : {threshold:.3f}")

    print(classification_report(y_true, preds, target_names=["low cost", "high cost"]))

    return dict(auc=auc, pr_auc=apr, f1=f1, precision=prec, recall=rec, threshold=threshold)


# shap importance
def shap_importance(model, X, top_n=30, output_dir="output"):
    log("computing shap values...")

    try:
        base_model = model
        if hasattr(model, "base_model"):
            base_model = model.base_model

        explainer = shap.TreeExplainer(base_model)
        sample = X.sample(min(2000, len(X)), random_state=42)
        shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        importance = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        out_path = os.path.join(output_dir, "shap_importance.csv")
        importance.to_csv(out_path, index=False)

        log(f"shap importance saved → {out_path}")
        log(f"\ntop {top_n} features:")
        print(importance.head(top_n).to_string(index=False))

        return importance
    except Exception as e:
        log(f"shap failed: {e}")
        return None