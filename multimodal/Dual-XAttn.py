#!/usr/bin/env python3
"""
DualCrossAttentionModel with K-Fold Cross-Validation
- Replaces single train/test split with stratified K-fold CV
- Averages metrics across folds (precision, recall, F1, AUC)
"""

import ast
import math
import random
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from transformers import Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# -------------------- Configuration --------------------
CLS_COLUMN = "CLS"

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

SEED = 42


# -------------------- Utilities --------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_embedding_column(col: pd.Series) -> pd.Series:
    return col.apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))


def load_and_prepare_data(structured_path: str, cls_path: str):
    df_struct = pd.read_excel(structured_path)
    df_cls = pd.read_csv(cls_path)

    if CLS_COLUMN not in df_cls.columns:
        raise ValueError(f"Expected column '{CLS_COLUMN}' in CLS CSV")
    df_cls[CLS_COLUMN] = parse_embedding_column(df_cls[CLS_COLUMN])

    if 'id' in df_cls.columns:
        df_cls = df_cls.rename(columns={'id': 'NO_SEJOUR'})
    if 'NO_SEJOUR' not in df_cls.columns:
        raise ValueError("CLS CSV must contain 'NO_SEJOUR' or 'id' column to merge with structured data")

    if 'y' in df_struct.columns:
        df_struct = df_struct.rename(columns={'y': 'label'})
    if 'label' not in df_struct.columns:
        raise ValueError("Structured Excel must contain a 'label' column (or 'y' to be renamed)")

    present_struct_vars = [c for c in STRUCTURED_VARS if c in df_struct.columns]
    selected_vars = ['NO_SEJOUR', 'label'] + present_struct_vars
    df_struct = df_struct[selected_vars]

    df_merged = pd.merge(df_struct, df_cls, on='NO_SEJOUR', how='inner')

    if 'label_x' in df_merged.columns:
        df_merged = df_merged.rename(columns={'label_x': 'label'})
    if 'label_y' in df_merged.columns:
        df_merged = df_merged.drop(columns=['label_y'])

    df_merged['label'] = df_merged['label'].astype(int)

    return df_merged, present_struct_vars



# -------------------- Dataset --------------------
class MultimodalDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, struct_cols):
        self.cls = np.stack(dataframe[CLS_COLUMN].values)
        self.structured = torch.tensor(dataframe[struct_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(dataframe['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'cls': torch.tensor(self.cls[idx], dtype=torch.float32),
                'structured': self.structured[idx],
                'labels': self.labels[idx]}


# -------------------- Model --------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if context.dim() == 2:
            context = context.unsqueeze(1)

        Q, K, V = self.query_proj(query), self.key_proj(context), self.value_proj(context)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        out = self.dropout(attended) + query
        return self.norm(out).squeeze(1)


class DualCrossAttentionModel(nn.Module):
    def __init__(self, cls_dim, struct_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.struct_proj = nn.Linear(struct_dim, cls_dim)
        self.cross1 = CrossAttentionBlock(cls_dim, dropout)
        self.cross2 = CrossAttentionBlock(cls_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cls_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, cls, structured, labels=None):
        struct_proj = self.struct_proj(structured)
        cls_updated = self.cross1(cls, struct_proj)
        struct_updated = self.cross2(struct_proj, cls_updated)
        fused = torch.cat([cls_updated, struct_updated], dim=-1)
        logits = self.classifier(fused)
        loss = None
        if labels is not None:
            labels_np = labels.cpu().numpy()
            class_counts = np.array([np.sum(labels_np == t) for t in [0, 1]])
            weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(logits.device)
            loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return {"loss": loss, "logits": logits}


# -------------------- Metrics --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = logits[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan')
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'auc': float(auc)}


# -------------------- Cross-Validation --------------------
def cross_validate(df, used_struct_vars, args, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    metrics_all = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n--- Fold {fold + 1}/{k} ---")

        train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        train_struct = imputer.fit_transform(train_df[used_struct_vars])
        test_struct = imputer.transform(test_df[used_struct_vars])
        train_df.loc[:, used_struct_vars] = scaler.fit_transform(train_struct)
        test_df.loc[:, used_struct_vars] = scaler.transform(test_struct)

        train_dataset = MultimodalDataset(train_df, used_struct_vars)
        test_dataset = MultimodalDataset(test_df, used_struct_vars)

        model = DualCrossAttentionModel(
            cls_dim=train_dataset.cls.shape[1],
            struct_dim=len(used_struct_vars),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )

        fold_output = os.path.join(args.output_dir, f"fold_{fold+1}")
        os.makedirs(fold_output, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_output,
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            seed=SEED,
            logging_dir=f"{fold_output}/logs",
            logging_steps=20,
            report_to=[],
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        fold_metrics = trainer.evaluate(test_dataset)
        print(f"Fold {fold + 1} metrics:", fold_metrics)
        metrics_all.append(fold_metrics)

    # Aggregate results
    df_metrics = pd.DataFrame(metrics_all)
    mean_metrics = df_metrics.mean().to_dict()
    std_metrics = df_metrics.std().to_dict()
    summary = {f"{k}_mean": v for k, v in mean_metrics.items()}
    summary.update({f"{k}_std": v for k, v in std_metrics.items()})
    pd.DataFrame([summary]).to_csv(args.metrics_path, index=False)
    print("\n==== Cross-Validation Results ====")
    print(pd.DataFrame([summary]).T)


# -------------------- Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualCrossAttentionModel with k-fold CV")
    parser.add_argument("--structured_path", type=str, required=True)
    parser.add_argument("--cls_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(SEED)
    df, used_struct_vars = load_and_prepare_data(args.structured_path, args.cls_path)
    cross_validate(df, used_struct_vars, args, k=args.folds)
