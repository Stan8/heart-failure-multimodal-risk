#!/usr/bin/env python3
import ast
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from typing import List
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


# -------------------- Configuration --------------------
CLS_COLUMN = "CLS"
ENTITY_COLUMNS = [
    'AGE', 'PATHOLOGIE', 'SIGNE_SYMPTOME', 'TRAITEMENT', 'ANATOMIE', 'EXAMEN',
    'ENTOURAGE', 'AUTONOMIE', 'CONCENTRATION', 'MODE', 'DOSE', 'FREQUENCE',
    'PARAMETRE_MESURABLE', 'VALEUR', 'NEGATION', 'HYPOTHETIQUE',
    'EVOLUTION_TRAITEMENT_PARAMETRE', 'COMPORTEMENT', 'DUREE', 'EVOLUTION',
    'CHANGEMENT_LIEU', 'LIEU', 'DATE'
]
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

# -------------------- Utilities --------------------
def parse_embedding_column(col: pd.Series) -> pd.Series:
    return col.apply(
        lambda x: np.array(ast.literal_eval(x), dtype=np.float32) if isinstance(x, str)
        else np.array(x, dtype=np.float32)
    )

def load_and_merge(structured_path: str, cls_path: str):
    df_struct = pd.read_excel(structured_path)
    df_cls = pd.read_csv(cls_path)

    if CLS_COLUMN not in df_cls.columns:
        raise ValueError(f"Expected column '{CLS_COLUMN}' in CLS CSV")
    df_cls[CLS_COLUMN] = parse_embedding_column(df_cls[CLS_COLUMN])

    present_entities = []
    for col in ENTITY_COLUMNS:
        if col in df_cls.columns:
            df_cls[col] = parse_embedding_column(df_cls[col])
            present_entities.append(col)

    if 'id' in df_cls.columns:
        df_cls = df_cls.rename(columns={'id': 'NO_SEJOUR'})
    if 'NO_SEJOUR' not in df_cls.columns:
        raise ValueError("CLS CSV must contain 'NO_SEJOUR' or 'id'")

    if 'y' in df_struct.columns:
        df_struct = df_struct.rename(columns={'y': 'label'})
    if 'label' not in df_struct.columns:
        raise ValueError("Structured Excel must contain 'label' column")

    present_struct_vars = [c for c in STRUCTURED_VARS if c in df_struct.columns]
    selected_vars = ['NO_SEJOUR', 'label'] + present_struct_vars
    df_struct = df_struct[selected_vars]

    df_merged = pd.merge(df_struct, df_cls, on='NO_SEJOUR', how='inner')
    if 'label_x' in df_merged.columns:
        df_merged = df_merged.rename(columns={'label_x': 'label'})
    if 'label_y' in df_merged.columns:
        df_merged = df_merged.drop(columns=['label_y'])
    df_merged['label'] = df_merged['label'].astype(int)

    return df_merged, present_struct_vars, present_entities

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if logits.ndim > 1:
        preds = logits.argmax(axis=1)
        probs = logits[:, 1]
    else:
        preds = (logits > 0.5).astype(int)
        probs = logits
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float('nan')
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'auc': float(auc)}

# -------------------- CLS Dataset + Model --------------------
class CLSDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {"input_embeddings": self.embeddings[idx], "labels": self.labels[idx]}

class CLSClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, input_embeddings, labels=None):
        logits = self.fc(input_embeddings)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

# -------------------- Late Fusion with CV --------------------
def run_late_fusion(structured_path: str,
                    cls_path: str,
                    metrics_output: str,
                    output_dir: str,
                    epochs: int = 5,
                    lr: float = 1e-3,
                    batch_size: int = 16,
                    seed: int = 42,
                    balanced_struct: bool = True,
                    folds: int = 5):

    df, struct_cols, _ = load_and_merge(structured_path, cls_path)
    X_cls = np.stack(df[CLS_COLUMN].values)
    y = df['label'].values.astype(int)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_metrics = []
    all_predictions = []  #  to collect per-fold predictions

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_cls, y), 1):
        print(f"\n===== Fold {fold}/{folds} =====")
        X_cls_train, X_cls_test = X_cls[train_idx], X_cls[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        # Structured branch
        if len(struct_cols) > 0:
            imputer = SimpleImputer(strategy="mean")
            scaler = StandardScaler()
            Xs_train = scaler.fit_transform(imputer.fit_transform(train_df[struct_cols].values.astype(np.float32)))
            Xs_test = scaler.transform(imputer.transform(test_df[struct_cols].values.astype(np.float32)))
            lr_struct = LogisticRegression(max_iter=500,
                                           class_weight="balanced" if balanced_struct else None,
                                           random_state=seed)
            lr_struct.fit(Xs_train, y_train)
            preds_struct_train = lr_struct.predict_proba(Xs_train)[:, 1]
            preds_struct_test = lr_struct.predict_proba(Xs_test)[:, 1]
        else:
            preds_struct_train = np.full(len(y_train), 0.5)
            preds_struct_test = np.full(len(y_test), 0.5)

        # CLS branch
        train_dataset = CLSDataset(X_cls_train, y_train)
        test_dataset = CLSDataset(X_cls_test, y_test)
        cls_model = CLSClassifier(input_dim=X_cls_train.shape[1], num_classes=2)

        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            logging_dir=os.path.join(fold_output_dir, "logs"),
            learning_rate=lr,
            weight_decay=0.01,
            logging_steps=10,
            seed=seed,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=[]
        )

        trainer = Trainer(
            model=cls_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            tokenizer=None
        )
        trainer.train()

        preds_cls_train = trainer.predict(train_dataset).predictions
        preds_cls_train_proba = torch.softmax(torch.tensor(preds_cls_train), dim=1)[:, 1].numpy()
        preds_cls_test = trainer.predict(test_dataset).predictions
        preds_cls_test_proba = torch.softmax(torch.tensor(preds_cls_test), dim=1)[:, 1].numpy()

        # Fusion meta-model
        stacked_train = np.vstack([preds_struct_train, preds_cls_train_proba]).T
        stacked_test = np.vstack([preds_struct_test, preds_cls_test_proba]).T
        meta_lr = LogisticRegression(max_iter=500, random_state=seed)
        meta_lr.fit(stacked_train, y_train)
        meta_proba = meta_lr.predict_proba(stacked_test)[:, 1]
        meta_logits = np.stack([1 - meta_proba, meta_proba], axis=1)

        metrics = compute_metrics((meta_logits, y_test))

        print(f"\n=== Fold {fold} Results ===")
        print(f"Test set size: {len(y_test)}")
        print(f"Positives: {(y_test == 1).sum()} | Negatives: {(y_test == 0).sum()}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        all_metrics.append(metrics)

        #  Collect predictions
        fold_preds = pd.DataFrame({
            "NO_SEJOUR": test_df["NO_SEJOUR"].values,
            "y_true": y_test,
            "y_proba": meta_proba,
            "fold": fold
        })
        all_predictions.append(fold_preds)

    # === Metrics summary ===
    df_metrics = pd.DataFrame(all_metrics)
    mean_metrics = df_metrics.mean().to_dict()
    std_metrics = df_metrics.std().to_dict()

    print("\n=== Cross-validation Results ===")
    for k in mean_metrics.keys():
        print(f"{k}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")

    os.makedirs(os.path.dirname(metrics_output) or ".", exist_ok=True)
    df_metrics.loc["mean"] = df_metrics.mean()
    df_metrics.to_csv(metrics_output, index=False)

    # === Save all predictions ===
    all_predictions = pd.concat(all_predictions, ignore_index=True)
    preds_output = metrics_output.replace(".csv", "_predictions.csv")
    all_predictions.to_csv(preds_output, index=False)
    print(f"Saved fold predictions to {preds_output}")



# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Late Fusion (Stacking) with LR on structured + CLS linear head")
    parser.add_argument("--cls", type=str, required=True, help="CLS CSV with CLS embeddings")
    parser.add_argument("--structured", type=str, required=True, help="Structured Excel path")
    parser.add_argument("--metrics_output", type=str, required=True, help="CSV path to save metrics")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory (per-fold CLS logs)")
    parser.add_argument("--folds", type=int, default=5, help="Number of StratifiedKFold splits")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs for CLS classifier")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for CLS classifier")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for CLS classifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_balanced_struct", action="store_true",
                        help="Disable class_weight='balanced' for structured LR")
    args = parser.parse_args()

    run_late_fusion(
        structured_path=args.structured,
        cls_path=args.cls,
        metrics_output=args.metrics_output,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        balanced_struct=not args.no_balanced_struct,
        folds=args.folds
    )
