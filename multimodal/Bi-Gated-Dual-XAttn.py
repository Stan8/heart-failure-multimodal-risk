#!/usr/bin/env python3
"""
Gated Cross-Attention pipeline with Stratified K-Fold CV
"""
import ast
import math
import random
import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from transformers import Trainer, TrainingArguments
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
SEED = 42
N_FOLDS = 5

# -------------------- Utilities --------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_embedding_column(col: pd.Series) -> pd.Series:
    return col.apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32))

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
        raise ValueError("Structured Excel must contain a 'label' column")

    present_struct_vars = [c for c in STRUCTURED_VARS if c in df_struct.columns]
    df_struct = df_struct[['NO_SEJOUR', 'label'] + present_struct_vars]

    df_merged = pd.merge(df_struct, df_cls, on='NO_SEJOUR', how='inner')
    if 'label_x' in df_merged.columns:
        df_merged = df_merged.rename(columns={'label_x': 'label'})
    if 'label_y' in df_merged.columns:
        df_merged = df_merged.drop(columns=['label_y'])
    df_merged['label'] = df_merged['label'].astype(int)

    return df_merged, present_struct_vars, present_entities

# -------------------- Dataset --------------------
class GatedCrossDataset(Dataset):
    def __init__(self, df: pd.DataFrame, struct_cols: List[str], entity_cols: List[str]):
        self.cls = np.stack(df[CLS_COLUMN].values)
        if len(entity_cols) > 0:
            entity_list = [np.stack(df[c].values) for c in entity_cols]
            self.entities = np.stack(entity_list, axis=1)
        else:
            emb_dim = self.cls.shape[1]
            self.entities = np.empty((len(df), 0, emb_dim), dtype=np.float32)
        if len(struct_cols) > 0:
            self.structured = df[struct_cols].values.astype(np.float32)
        else:
            self.structured = np.empty((len(df), 0), dtype=np.float32)
        self.labels = df['label'].values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "cls": torch.tensor(self.cls[idx], dtype=torch.float32),
            "entities": torch.tensor(self.entities[idx], dtype=torch.float32),
            "structured": torch.tensor(self.structured[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# -------------------- Cross-Attention Model --------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if context.dim() == 2:
            context = context.unsqueeze(1)
        Q = self.query_proj(query)
        K = self.key_proj(context)
        V = self.value_proj(context)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        out = self.dropout(attended) + query
        out = self.norm(out)
        return out.squeeze(1) if out.size(1) == 1 else out

class GatedCrossAttentionModel(nn.Module):
    def __init__(self, emb_dim: int, struct_dim: int, hidden_dim: int = 128, dropout: float = 0.1, class_weights: Optional[torch.Tensor]=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.struct_dim = struct_dim
        self.gate_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )
        self.struct_proj = nn.Linear(struct_dim, emb_dim)
        self.cross1 = CrossAttentionBlock(emb_dim, dropout)
        self.cross2 = CrossAttentionBlock(emb_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.clone().detach())
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def fuse_embeddings(self, cls: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        if entities.numel() == 0:
            return cls
        entity_avg = entities.mean(dim=1)
        combined = torch.cat([cls, entity_avg], dim=1)
        gate = self.gate_layer(combined).expand(-1, cls.size(1))
        return gate * cls + (1.0 - gate) * entity_avg

    def forward(self, cls: torch.Tensor, entities: torch.Tensor, structured: torch.Tensor, labels: Optional[torch.Tensor] = None):
        fused_cls = self.fuse_embeddings(cls, entities)
        struct_proj = self.struct_proj(structured)
        cls_u = self.cross1(fused_cls, struct_proj)
        struct_u = self.cross2(struct_proj, cls_u)
        fused = torch.cat([cls_u, struct_u], dim=1)
        logits = self.classifier(fused)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# -------------------- Metrics --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1) if logits.ndim > 1 else (logits > 0.5).astype(int)
    probs = logits[:, 1] if logits.ndim > 1 else logits
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = float('nan')
    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'roc_auc': float(auc)}

# -------------------- Training with Stratified K-Fold --------------------
def train_cv(structured_path: str, cls_path: str, metrics_path: str, output_dir: str,
             batch_size: int = 8, epochs: int = 100, hidden_dim: int = 128, dropout: float = 0.1):

    set_seed(SEED)
    df, struct_cols, entity_cols = load_and_merge(structured_path, cls_path)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    X_cls = df[CLS_COLUMN].values
    y = df['label'].values

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_cls, y), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        # Impute & scale structured features
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        if struct_cols:
            train_struct = imputer.fit_transform(train_df[struct_cols])
            test_struct = imputer.transform(test_df[struct_cols])
            scaler.fit(train_struct)
            train_df.loc[:, struct_cols] = scaler.transform(train_struct)
            test_df.loc[:, struct_cols] = scaler.transform(test_struct)

        train_dataset = GatedCrossDataset(train_df, struct_cols, entity_cols)
        test_dataset = GatedCrossDataset(test_df, struct_cols, entity_cols)

        y_train = train_df['label'].astype(int).values
        class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train), dtype=torch.float32)

        emb_dim = train_dataset.cls.shape[1]
        struct_dim = train_dataset.structured.shape[1] if train_dataset.structured.size else 0

        model = GatedCrossAttentionModel(emb_dim=emb_dim, struct_dim=struct_dim,
                                         hidden_dim=hidden_dim, dropout=dropout,
                                         class_weights=class_weights)

        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{fold_output_dir}/logs",
            seed=SEED,
            save_total_limit=1,
            report_to=[],
        )

        trainer = Trainer(model=model, args=training_args,
                          train_dataset=train_dataset, eval_dataset=test_dataset,
                          compute_metrics=compute_metrics)

        trainer.train()
        metrics = trainer.evaluate()
        print(f"Fold {fold} metrics:", metrics)
        metrics['fold'] = fold
        fold_metrics.append(metrics)

    # Aggregate CV results
    df_metrics = pd.DataFrame(fold_metrics)
    summary = df_metrics.drop(columns=['fold']).agg(['mean', 'std'])
    print("\nCross-validation summary:")
    print(summary)

    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    df_metrics.to_csv(metrics_path, index=False)

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, required=True)
    parser.add_argument("--structured", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    train_cv(args.structured, args.cls, args.metrics_output, args.output_dir,
             batch_size=args.batch_size, epochs=args.epochs,
             hidden_dim=args.hidden_dim, dropout=args.dropout)
