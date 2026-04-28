"""
trains lightgbm (with optional xgboost), stacking or rank averaging,
probability calibration, and robust checkpoints.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from model import (
    prepare_features, align_columns,
    train_lgbm, train_xgb,
    train_stacking_meta, predict_stacking_meta,
    calibrate_probabilities,
    rank_average, tune_threshold,
    full_report, shap_importance,
    _load_ckpt_array, _load_ckpt_json,
)


# config
train_features = r"C:\Users\Hp\Downloads\softec-26-machine-learning-competition\output\features_train.csv"
test_features  = r"C:\Users\Hp\Downloads\softec-26-machine-learning-competition\output\features_test.csv"
output_dir     = r"C:\Users\Hp\Downloads\softec-26-machine-learning-competition\output"

label_col = "HighCostLabel"

lgbm_trials = 60
xgb_trials  = 40
cv_folds    = 5

threshold_strategy = "recall_constrained"

use_xgb_ensemble = True
use_stacking     = True
use_calibration  = True


# helpers
def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def stage_is_done(stage_name):
    return os.path.exists(os.path.join(output_dir, f"_stage_{stage_name}.done"))


def mark_stage_done(stage_name):
    path = os.path.join(output_dir, f"_stage_{stage_name}.done")
    with open(path, "w") as f:
        f.write(datetime.now().isoformat())
    log(f"stage '{stage_name}' marked as complete")


def main():
    os.makedirs(output_dir, exist_ok=True)

    log("healthcare cost classification — model training")
    log("=" * 60)

    # 1. load data
    log("\n[1/7] loading feature files...")
    train_df = pd.read_csv(train_features, low_memory=False)
    test_df  = pd.read_csv(test_features,  low_memory=False)

    log(f"train shape: {train_df.shape} | test shape: {test_df.shape}")

    if label_col not in train_df.columns:
        raise ValueError(f"label column '{label_col}' not found in training data.")

    if label_col in test_df.columns:
        log("highcostlabel found in test set — dropping it")
        test_df = test_df.drop(columns=[label_col])

    pos_rate = train_df[label_col].mean()
    log(f"high-cost rate in training data: {pos_rate:.2%}")

    # save member_key for submission
    test_member_keys = test_df["Member_Key"].values if "Member_Key" in test_df.columns else np.arange(len(test_df))

    # 2. prepare features
    log("\n[2/7] preparing features...")

    X_train, y_train, feature_names, encoders = prepare_features(train_df, label_col=label_col)

    # safe categorical encoding for test set
    test_processed = test_df.copy()
    for col, encoder in encoders.items():
        if col in test_processed.columns:
            known_classes = set(encoder.classes_)
            test_processed[col] = test_processed[col].astype(str).apply(
                lambda x: x if x in known_classes else encoder.classes_[0]
            )
            test_processed[col] = encoder.transform(test_processed[col])

    # drop id and leakage columns from test
    cols_to_drop = {"Member_Key", "YEAR", "MONTH", "NEXT_YEAR_COST", label_col}
    test_feature_cols = [col for col in test_processed.columns if col not in cols_to_drop]
    X_test = test_processed[test_feature_cols].fillna(0)

    # align train and test columns
    X_train, X_test = align_columns(X_train, X_test)
    log(f"final feature shapes → train: {X_train.shape} | test: {X_test.shape}")

    # calibration split
    if use_calibration:
        X_model, X_cal, y_model, y_cal = train_test_split(
            X_train, y_train,
            test_size=0.10,
            stratify=y_train,
            random_state=42
        )
        log(f"calibration split created: {X_cal.shape[0]:,} rows held out "
            f"({y_cal.mean():.2%} positive)")
    else:
        X_model, y_model = X_train, y_train
        X_cal, y_cal = None, None

    n_train_rows = X_model.shape[0]
    log(f"base models will be trained on {n_train_rows:,} rows")

    # 3. train lightgbm
    log("\n[3/7] training lightgbm...")

    lgbm_oof = _load_ckpt_array(output_dir, "lgbm_oof_proba")
    lgbm_params = _load_ckpt_json(output_dir, "lgbm_best_params")

    # check for stale checkpoint
    if (lgbm_oof is not None and len(lgbm_oof) == n_train_rows and 
        lgbm_params is not None and stage_is_done("lgbm")):
        log("loading saved lightgbm model and oof predictions")
        lgbm_model = joblib.load(os.path.join(output_dir, "_interim_lgbm_model.pkl"))
    else:
        if lgbm_oof is not None and len(lgbm_oof) != n_train_rows:
            log(f"oof checkpoint mismatch ({len(lgbm_oof):,} vs {n_train_rows:,}) — forcing retrain")
            stage_file = os.path.join(output_dir, "_stage_lgbm.done")
            if os.path.exists(stage_file):
                os.remove(stage_file)

        lgbm_model, _, lgbm_oof, lgbm_params = train_lgbm(
            X_model, y_model,
            n_trials=lgbm_trials,
            cv_folds=cv_folds,
            output_dir=output_dir,
        )
        mark_stage_done("lgbm")

    # calibrate probabilities
    if use_calibration and X_cal is not None:
        log("\ncalibrating lightgbm probabilities...")
        cal_path = os.path.join(output_dir, "_calibrated_lgbm.pkl")

        if os.path.exists(cal_path) and stage_is_done("calibration"):
            log("loading calibrated lgbm model")
            lgbm_model_cal = joblib.load(cal_path)
        else:
            lgbm_model_cal = calibrate_probabilities(
                lgbm_model, X_cal, y_cal,
                method="isotonic",
                output_dir=output_dir,
                feature_names=list(X_model.columns),
            )
            mark_stage_done("calibration")
    else:
        lgbm_model_cal = lgbm_model

    # 4. train xgboost (optional)
    xgb_model = None
    xgb_oof = None
    xgb_params = {}

    if use_xgb_ensemble:
        log("\n[4/7] training xgboost...")

        xgb_oof = _load_ckpt_array(output_dir, "xgb_oof_proba")
        xgb_params = _load_ckpt_json(output_dir, "xgb_best_params") or {}

        if (xgb_oof is not None and len(xgb_oof) == n_train_rows and stage_is_done("xgb")):
            log("loading saved xgboost model")
            xgb_model = joblib.load(os.path.join(output_dir, "_interim_xgb_model.pkl"))
        else:
            if xgb_oof is not None and len(xgb_oof) != n_train_rows:
                log(f"xgboost oof mismatch — forcing retrain")
                stage_file = os.path.join(output_dir, "_stage_xgb.done")
                if os.path.exists(stage_file):
                    os.remove(stage_file)

            xgb_model, xgb_oof, xgb_params = train_xgb(
                X_model, y_model,
                n_trials=xgb_trials,
                cv_folds=cv_folds,
                output_dir=output_dir,
            )
            mark_stage_done("xgb")
    else:
        log("skipping xgboost (disabled in config)")

    # 5. ensemble and threshold tuning
    log("\n[5/7] creating ensemble and tuning threshold...")

    if xgb_oof is not None and use_stacking:
        log("building stacking meta-learner...")
        meta_path = os.path.join(output_dir, "_meta_model.pkl")

        if os.path.exists(meta_path) and stage_is_done("meta"):
            log("loading saved meta model")
            meta_model = joblib.load(meta_path)
            X_meta_oof = np.column_stack([lgbm_oof, xgb_oof])
            oof_ensemble = meta_model.predict_proba(X_meta_oof)[:, 1]
        else:
            meta_model, oof_ensemble = train_stacking_meta(
                lgbm_oof, xgb_oof, y_model, output_dir=output_dir
            )
            mark_stage_done("meta")

    elif xgb_oof is not None:
        log("using rank-average ensemble")
        oof_ensemble = rank_average([lgbm_oof, xgb_oof])
        meta_model = None
    else:
        log("using lightgbm only")
        oof_ensemble = lgbm_oof
        meta_model = None

    best_threshold = tune_threshold(y_model, oof_ensemble, strategy=threshold_strategy)

    log("\noof performance:")
    full_report(y_model, oof_ensemble, best_threshold, label="ensemble oof")

    # 6. predict on test set
    log("\n[6/7] generating predictions on test set...")

    lgbm_test_proba = lgbm_model_cal.predict_proba(X_test)[:, 1]

    if xgb_model is not None:
        xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]

        if meta_model is not None and use_stacking:
            test_proba = predict_stacking_meta(meta_model, lgbm_test_proba, xgb_test_proba)
            log("test predictions using stacking meta-learner")
        else:
            test_proba = rank_average([lgbm_test_proba, xgb_test_proba])
            log("test predictions using rank-average ensemble")
    else:
        test_proba = lgbm_test_proba
        log("test predictions from lightgbm only")

    test_predictions = (test_proba >= best_threshold).astype(int)

    log(f"test set: {test_predictions.sum()} members predicted as high-cost "
        f"({test_predictions.mean():.2%})")

    # 7. save everything
    log("\n[7/7] saving models and submission files...")

    joblib.dump(lgbm_model_cal, os.path.join(output_dir, "lgbm_model.pkl"))
    if xgb_model:
        joblib.dump(xgb_model, os.path.join(output_dir, "xgb_model.pkl"))
    if meta_model:
        joblib.dump(meta_model, os.path.join(output_dir, "meta_model.pkl"))

    # submission
    submission = pd.DataFrame({
        "Member_Key": test_member_keys,
        "HighCostLabel": test_predictions
    })
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

    submission_proba = pd.DataFrame({
        "Member_Key": test_member_keys,
        "HighCostProbability": test_proba,
        "HighCostLabel": test_predictions
    })
    submission_proba.to_csv(os.path.join(output_dir, "submission_proba.csv"), index=False)

    # oof predictions
    oof_df = pd.DataFrame({
        "HighCostLabel": y_model.values,
        "oof_proba_lgbm": lgbm_oof,
        "oof_proba_ensemble": oof_ensemble,
    })
    if xgb_oof is not None:
        oof_df["oof_proba_xgb"] = xgb_oof
    oof_df.to_csv(os.path.join(output_dir, "oof_predictions.csv"), index=False)

    log("\ntraining completed successfully!")
    log("=" * 60)
    log(f"all outputs saved in: {output_dir}")
    log("   → submission.csv")
    log("   → submission_proba.csv")
    log("   → lgbm_model.pkl")
    log("   → meta_model.pkl (if used)")
    log("   → metrics_summary.json")
    log("\nto force full retrain, delete all _stage_*.done and _ckpt_* files.")


if __name__ == "__main__":
    main()