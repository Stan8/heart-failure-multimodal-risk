#!/usr/bin/env python3


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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from transformers import Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

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

# -------------------- Utilities --------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_embedding_column(col: pd.Series) -> pd.Series:
    """Safely parse string embeddings like "[0.1, 0.2, ...]" into numpy arrays."""
    return col.apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32))

def load_and_merge(structured_path: str, cls_path: str):
    """
    Loads structured Excel and CLS CSV. Parses CLS and entity embedding columns (if present).
    Returns merged DataFrame, present_struct_vars, present_entity_cols.
    """
    df_struct = pd.read_excel(structured_path)
    df_cls = pd.read_csv(cls_path)

    # parse CLS embeddings
    if CLS_COLUMN not in df_cls.columns:
        raise ValueError(f"Expected column '{CLS_COLUMN}' in CLS CSV")
    df_cls[CLS_COLUMN] = parse_embedding_column(df_cls[CLS_COLUMN])

    # parse entity embedding columns if present
    present_entities = []
    for col in ENTITY_COLUMNS:
        if col in df_cls.columns:
            df_cls[col] = parse_embedding_column(df_cls[col])
            present_entities.append(col)

    # normalize keys
    if 'id' in df_cls.columns:
        df_cls = df_cls.rename(columns={'id': 'NO_SEJOUR'})
    if 'NO_SEJOUR' not in df_cls.columns:
        raise ValueError("CLS CSV must contain 'NO_SEJOUR' or 'id' to merge with structured data")

    if 'y' in df_struct.columns:
        df_struct = df_struct.rename(columns={'y': 'label'})
    if 'label' not in df_struct.columns:
        raise ValueError("Structured Excel must contain a 'label' column (or 'y' to be renamed)")

    # select structured vars present
    present_struct_vars = [c for c in STRUCTURED_VARS if c in df_struct.columns]
    selected_vars = ['NO_SEJOUR', 'label'] + present_struct_vars
    df_struct = df_struct[selected_vars]

    # merge
    df_merged = pd.merge(df_struct, df_cls, on='NO_SEJOUR', how='inner')

    # handle label_x/label_y
    if 'label_x' in df_merged.columns:
        df_merged = df_merged.rename(columns={'label_x': 'label'})
    if 'label_y' in df_merged.columns:
        df_merged = df_merged.drop(columns=['label_y'])

    df_merged['label'] = df_merged['label'].astype(int)

    return df_merged, present_struct_vars, present_entities

# -------------------- Dataset --------------------

class GatedCrossDataset(Dataset):
    """
    Returns items with:
      - 'cls': main CLS embedding (vector)
      - 'entities': stacked entity embeddings (num_entities, emb_dim)  (may be zero-sized if none)
      - 'structured': structured features vector
      - 'labels'
    """
    def __init__(self, df: pd.DataFrame, struct_cols: List[str], entity_cols: List[str]):
        # main CLS vector
        self.cls = np.stack(df[CLS_COLUMN].values)  # (N, emb_dim)
        # prepare entity tensors: if no entity cols -> empty (N, 0, emb_dim)
        if len(entity_cols) > 0:
            entity_list = []
            for c in entity_cols:
                entity_list.append(np.stack(df[c].values))  # (N, emb_dim)
            self.entities = np.stack(entity_list, axis=1)  # (N, num_entities, emb_dim)
        else:
            emb_dim = self.cls.shape[1]
            self.entities = np.empty((len(df), 0, emb_dim), dtype=np.float32)

        # structured data (assumes already imputed & scaled)
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

# -------------------- Model: Bi-directional Gated Cross-Attention --------------------

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query: (B, D) or (B, q_len, D)
        # context: (B, c_len, D)
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if context.dim() == 2:
            context = context.unsqueeze(1)

        Q = self.query_proj(query)    # (B, q_len, D)
        K = self.key_proj(context)    # (B, c_len, D)
        V = self.value_proj(context)  # (B, c_len, D)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))  # (B, q_len, c_len)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)  # (B, q_len, D)

        out = self.dropout(attended) + query
        out = self.norm(out)

        if out.size(1) == 1:
            out = out.squeeze(1)
        return out

class BiGatedCrossAttentionModel(nn.Module):
    """
    Bi-directional gated cross-attention:
      - fuse cls + entities with learnable gate -> fused_cls (B, emb_dim)
      - project structured -> struct_proj (B, emb_dim)
      - cls_u = cross_attn_cls_to_struct(fused_cls, struct_proj)
      - struct_u = cross_attn_struct_to_cls(struct_proj, fused_cls)
      - concat(cls_u, struct_u) -> classifier
    """
    def __init__(self, emb_dim: int, struct_dim: int, hidden_dim: int = 128, dropout: float = 0.1, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.struct_dim = struct_dim

        # gating (same architecture as before)
        self.gate_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

        # project structured features to embedding dim
        self.struct_proj = nn.Linear(struct_dim, emb_dim) if struct_dim > 0 else nn.Identity()

        # bi-directional cross-attention
        self.cross_cls_to_struct = CrossAttentionBlock(emb_dim, dropout=dropout)
        self.cross_struct_to_cls = CrossAttentionBlock(emb_dim, dropout=dropout)

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

        # handle class_weights safely: register buffer + pass cloned tensor to loss
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.clone().detach())
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def fuse_embeddings(self, cls: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        """Fuse CLS and entity embeddings with learnable gate."""
        if entities.numel() == 0:
            return cls
        entity_avg = entities.mean(dim=1)  # (B, emb_dim)
        combined = torch.cat([cls, entity_avg], dim=1)  # (B, 2*emb_dim)
        gate = self.gate_layer(combined)  # (B, 1)
        gate = gate.expand(-1, cls.size(1))  # (B, emb_dim)
        fused = gate * cls + (1.0 - gate) * entity_avg
        return fused

    def forward(self, cls: torch.Tensor, entities: torch.Tensor, structured: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        cls: (B, emb_dim)
        entities: (B, num_entities, emb_dim)
        structured: (B, struct_dim)
        labels: (B,)
        """
        fused_cls = self.fuse_embeddings(cls, entities)            # (B, emb_dim)
        struct_proj = self.struct_proj(structured) if self.struct_dim > 0 else torch.zeros_like(fused_cls)

        # bi-directional cross-attention updates
        cls_u = self.cross_cls_to_struct(fused_cls, struct_proj)   # (B, emb_dim)
        struct_u = self.cross_struct_to_cls(struct_proj, fused_cls) # (B, emb_dim)

        fused = torch.cat([cls_u, struct_u], dim=1)                # (B, 2*emb_dim)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

# -------------------- Metrics (same as gated model) --------------------

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
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc)
    }

# -------------------- Training pipeline --------------------

class NumpySafeTrainer(Trainer):
    """
    Small helper: ensure numpy arrays in metrics get converted if needed.
    Not strictly necessary but helpful for consistent saving.
    """
    pass



def train_cv(structured_path: str, cls_path: str, metrics_path: str, output_dir: str,
             batch_size: int = 8, epochs: int = 100, lr: float = 2e-4,
             hidden_dim: int = 128, dropout: float = 0.1, n_splits: int = 5):
    """
    Perform k-fold cross-validation training with BiGatedCrossAttentionModel.
    Saves the average metrics across folds to metrics_path.
    """
    set_seed(SEED)
    df, struct_cols, entity_cols = load_and_merge(structured_path, cls_path)

    if len(struct_cols) == 0:
        print("Warning: No structured vars found. Model will run with empty structured vector.")

    # Impute & scale structured features
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    if len(struct_cols) > 0:
        df_struct_scaled = scaler.fit_transform(imputer.fit_transform(df[struct_cols]))
        df.loc[:, struct_cols] = df_struct_scaled

    X_cls = df[CLS_COLUMN]
    y = df['label'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Build datasets
        train_dataset = GatedCrossDataset(train_df, struct_cols, entity_cols)
        val_dataset = GatedCrossDataset(val_df, struct_cols, entity_cols)

        # Compute class weights from training fold
        y_train = train_df['label'].astype(int).values
        class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

        emb_dim = train_dataset.cls.shape[1]
        struct_dim = train_dataset.structured.shape[1] if train_dataset.structured.size else 0

        model = BiGatedCrossAttentionModel(
            emb_dim=emb_dim,
            struct_dim=struct_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            class_weights=class_weights
        )

        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        fold_metrics.append(trainer.evaluate())

    # Average metrics across folds
    avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
    print("\n--- Cross-Validation Average Metrics ---")
    print(avg_metrics)
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    pd.DataFrame([avg_metrics]).to_csv(metrics_path, index=False)



# -------------------- CLI --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, required=True, help="CLS CSV with CLS + entity embedding columns")
    parser.add_argument("--structured", type=str, required=True, help="Structured Excel path")
    parser.add_argument("--metrics_output", type=str, required=True, help="CSV to save metrics")
    parser.add_argument("--output_dir", type=str, required=True, help="outputs dir for Trainer")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    train_cv(args.structured, args.cls, args.metrics_output, args.output_dir,
          batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
          hidden_dim=args.hidden_dim, dropout=args.dropout, n_splits=5)
