#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# -------------------- Structured Variables --------------------
STRUCTURED_VARS = [
    'BIO_UREE', 'ADMIN_AGE', 'MED_DIURETIQUES - DIURETIQUE NON EPARGNEUR POTASSIUM - DIURETIQUE THIAZIDIQUE',
    'SIGNES_Si dyspnée', 'BIO_kul', 'BIO_BNP', 'INF_pas', 'INF_delta', 'MED_RENINE-ANGIOTENSINE - ARA II - AUTRE',
    'MED_BETABLOQUANTS', 'BIO_hco3a', 'DIAG_Cerebrovascular Disease', 'BIO_CRP', 'DIAG_Weight Loss', 'BIO_GGT',
    'BIO_pha', 'BIO_TROPOHS', 'MED_ANTIDIABETIQUES, INSULINES EXCLUES - SULFAMIDES', 'MVT_ENTREE_TRANSFERT',
    'MED_RENINE-ANGIOTENSINE - IEC', 'ADMIN_T4', 'BIO_BILI', 'ECG_Si QRS fin', 'DIAG_Maladie_hepatique',
    'BIO_CCMH', 'DIAG_Obesity', 'ECG_Si Pacemaker', 'MED_ANTIBACTERIENS A USAGE SYSTEMIQUE',
    'MED_ANTITHROMBOTIQUE - ANTICOAGULANT - HEPARINE', 'BIO_TSH', 'BIO_LDL', 'BIO_POT', 'BIO_ASAT',
    'TRANS_TRANSFUSION_SANG', 'BIO_PLAQ', 'BIO_CK', 'BIO_PAL', 'BIO_VGM', 'BIO_sao2a', 'BIO_RBC', 'DIAG_Renal Disease'
]

# -------------------- Data Loading --------------------
def load_structured(structured_path: str):
    df_struct = pd.read_excel(structured_path)
    if 'y' in df_struct.columns:
        df_struct = df_struct.rename(columns={'y': 'label'})
    if 'label' not in df_struct.columns:
        raise ValueError("Structured Excel must contain 'label' column")

    present_struct_vars = [c for c in STRUCTURED_VARS if c in df_struct.columns]
    selected_vars = ['label'] + present_struct_vars
    df_struct = df_struct[selected_vars]

    X_structured = df_struct[present_struct_vars].values.astype(np.float32)
    y = df_struct['label'].values.astype(int)
    return X_structured, y, present_struct_vars

# -------------------- CV Logistic Regression --------------------
def run_structured_lr(structured_path: str, folds: int = 5, seed: int = 42, metrics_output: str = None):
    X_structured, y, present_vars = load_structured(structured_path)
    print(f"Using {len(present_vars)} structured variables: {present_vars}")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    metrics_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_structured, y), 1):
        print(f"\n--- Fold {fold} ---")
        X_train, X_test = X_structured[train_idx], X_structured[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Preprocess
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test = scaler.transform(imputer.transform(X_test))

        # Logistic Regression
        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        metrics_per_fold.append({
            "fold": fold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        })

        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Summary
    metrics_df = pd.DataFrame(metrics_per_fold)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    print("\n=== Structured Data Only (Logistic Regression) ===")
    for m in ["precision", "recall", "f1", "auc"]:
        print(f"{m}: {mean_metrics[m]:.4f} ± {std_metrics[m]:.4f}")

    # Save metrics if requested
    if metrics_output is not None:
        os.makedirs(os.path.dirname(metrics_output) or ".", exist_ok=True)
        
        # Create a summary row
        summary_row = pd.DataFrame([{
            "fold": "mean±std",
            "precision": f"{mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}",
            "recall": f"{mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}",
            "f1": f"{mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}",
            "auc": f"{mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}"
        }])
        
        # Concatenate per-fold metrics with summary
        metrics_df = pd.concat([metrics_df, summary_row], ignore_index=True)
        metrics_df.to_csv(metrics_output, index=False)
        print(f"\nMetrics saved to {metrics_output}")


# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured-only Logistic Regression CV")
    parser.add_argument("--structured", type=str, required=True, help="Structured Excel path")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--metrics_output", type=str, default=None, help="CSV file to save metrics")
    args = parser.parse_args()

    run_structured_lr(
        structured_path=args.structured,
        folds=args.folds,
        seed=args.seed,
        metrics_output=args.metrics_output
    )
