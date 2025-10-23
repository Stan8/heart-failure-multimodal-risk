import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PretrainedConfig, Trainer, TrainingArguments
import ast
import os

# ==== ENTITY & STRUCTURED VARIABLES ====
ENTITY_COLUMNS = [
    'AGE', 'PATHOLOGIE', 'SIGNE_SYMPTOME', 'TRAITEMENT', 'ANATOMIE', 'EXAMEN',
    'ENTOURAGE', 'AUTONOMIE', 'CONCENTRATION', 'MODE', 'DOSE', 'FREQUENCE',
    'PARAMETRE_MESURABLE', 'VALEUR', 'NEGATION', 'HYPOTHETIQUE',
    'EVOLUTION_TRAITEMENT_PARAMETRE', 'COMPORTEMENT', 'DUREE', 'EVOLUTION',
    'CHANGEMENT_LIEU', 'LIEU', 'DATE'
]

SELECTED_STRUCTURED_VARS = [
    'BIO_UREE', 'ADMIN_AGE', 'MED_DIURETIQUES - DIURETIQUE NON EPARGNEUR POTASSIUM - DIURETIQUE THIAZIDIQUE',
    'SIGNES_Si dyspnée', 'BIO_kul', 'BIO_BNP', 'INF_pas', 'INF_delta', 'MED_RENINE-ANGIOTENSINE - ARA II - AUTRE',
    'MED_BETABLOQUANTS', 'BIO_hco3a', 'DIAG_Cerebrovascular Disease', 'BIO_CRP', 'DIAG_Weight Loss', 'BIO_GGT',
    'BIO_pha', 'BIO_TROPOHS', 'MED_ANTIDIABETIQUES, INSULINES EXCLUES - SULFAMIDES', 'MVT_ENTREE_TRANSFERT',
    'MED_RENINE-ANGIOTENSINE - IEC', 'ADMIN_T4', 'BIO_BILI', 'ECG_Si QRS fin', 'DIAG_Maladie_hepatique',
    'BIO_CCMH', 'DIAG_Obesity', 'ECG_Si Pacemaker', 'MED_ANTIBACTERIENS A USAGE SYSTEMIQUE',
    'MED_ANTITHROMBOTIQUE - ANTICOAGULANT - HEPARINE', 'BIO_TSH', 'BIO_LDL', 'BIO_POT', 'BIO_ASAT',
    'TRANS_TRANSFUSION_SANG', 'BIO_PLAQ', 'BIO_CK', 'BIO_PAL', 'BIO_VGM', 'BIO_sao2a', 'BIO_RBC', 'DIAG_Renal Disease'
]


# ==== DATA HANDLING ====
def parse_embedding_column(col):
    return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


def load_and_merge(structured_path, cls_path):
    df_struct = pd.read_excel(structured_path)
    df_cls = pd.read_csv(cls_path)

    df_cls['CLS'] = parse_embedding_column(df_cls['CLS'])
    for col in ENTITY_COLUMNS:
        if col in df_cls.columns:
            df_cls[col] = parse_embedding_column(df_cls[col])

    if 'id' in df_cls.columns:
        df_cls = df_cls.rename(columns={'id': 'NO_SEJOUR'})
    if 'y' in df_struct.columns:
        df_struct = df_struct.rename(columns={'y': 'label'})

    selected_vars = ['NO_SEJOUR', 'label'] + [col for col in SELECTED_STRUCTURED_VARS if col in df_struct.columns]
    df_struct = df_struct[selected_vars]

    df_merged = pd.merge(df_struct, df_cls, on='NO_SEJOUR', how='inner')
    if 'label_x' in df_merged.columns:
        df_merged = df_merged.rename(columns={'label_x': 'label'}).drop(columns=['label_y'], errors='ignore')

    return df_merged


def prepare_data(cls_path, structured_path):
    df = load_and_merge(structured_path, cls_path)

    cls = np.stack(df['CLS'].values)
    cls_dim = cls.shape[1]

    # Entity embeddings
    entity_embeds = []
    for col in ENTITY_COLUMNS:
        if col in df.columns:
            vecs = np.stack(df[col].values)
            entity_embeds.append(vecs)
    entity = np.stack(entity_embeds, axis=1) if entity_embeds else np.empty((len(df), 0, cls_dim))

    # Structured data
    struct_cols = [col for col in SELECTED_STRUCTURED_VARS if col in df.columns]
    struct_cols = [col for col in struct_cols if pd.api.types.is_numeric_dtype(df[col])]
    struct_df = df[struct_cols].fillna(df[struct_cols].mean())
    struct_data = StandardScaler().fit_transform(struct_df)

    labels = df['label'].astype(int).values
    return cls, entity, struct_data, labels


# ==== MODEL DEFINITION ====
class GatedFusionConfig(PretrainedConfig):
    def __init__(self, hidden_size=768, num_labels=2, structured_size=0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.structured_size = structured_size


class GatedFusionWithStructured(PreTrainedModel):
    config_class = GatedFusionConfig

    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size + config.structured_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.gate_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, cls_embeddings, entity_embeddings, structured_features, labels=None):
        entity_avg = entity_embeddings.mean(dim=1)
        combined = torch.cat([cls_embeddings, entity_avg], dim=1)
        gate = self.gate_layer(combined).expand_as(cls_embeddings)

        fused = gate * cls_embeddings + (1 - gate) * entity_avg
        fused_with_struct = torch.cat([fused, structured_features], dim=1)
        logits = self.classifier(fused_with_struct)

        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# ==== DATASET ====
class EmbeddingDataset(Dataset):
    def __init__(self, cls, entity, struct, labels):
        self.cls = cls
        self.entity = entity
        self.struct = struct
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "cls_embeddings": torch.tensor(self.cls[idx], dtype=torch.float),
            "entity_embeddings": torch.tensor(self.entity[idx], dtype=torch.float),
            "structured_features": torch.tensor(self.struct[idx], dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ==== METRICS ====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    probs = logits[:, 1] if logits.shape[1] > 1 else logits
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan')
    return {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}


# ==== MAIN WITH CROSS-VALIDATION ====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, required=True)
    parser.add_argument("--structured", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, default="cv_results.csv")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cls, entity, struct, labels = prepare_data(args.cls, args.structured)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(cls, labels), 1):
        print(f"\n===== Fold {fold}/{args.folds} =====")

        X_train_cls, X_test_cls = cls[train_idx], cls[test_idx]
        X_train_ent, X_test_ent = entity[train_idx], entity[test_idx]
        X_train_struct, X_test_struct = struct[train_idx], struct[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        class_weights = torch.tensor(
            compute_class_weight('balanced', classes=np.unique(y_train), y=y_train),
            dtype=torch.float32
        )

        train_dataset = EmbeddingDataset(X_train_cls, X_train_ent, X_train_struct, y_train)
        test_dataset = EmbeddingDataset(X_test_cls, X_test_ent, X_test_struct, y_test)

        config = GatedFusionConfig(hidden_size=cls.shape[1], structured_size=struct.shape[1], num_labels=2)
        model = GatedFusionWithStructured(config, class_weights=class_weights)

        training_args = TrainingArguments(
            output_dir=f"{args.output_dir}/fold_{fold}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=f"{args.output_dir}/logs_fold_{fold}",
            logging_steps=10,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        fold_metrics = trainer.evaluate()
        fold_metrics['fold'] = fold
        print(f"Fold {fold} metrics:", fold_metrics)
        all_metrics.append(fold_metrics)

    df = pd.DataFrame(all_metrics)
    mean_metrics = df.mean(numeric_only=True).to_dict()
    mean_metrics['fold'] = 'mean'
    df = pd.concat([df, pd.DataFrame([mean_metrics])], ignore_index=True)

    df.to_csv(args.metrics_output, index=False)
    print("\n===== Cross-validation results =====")
    print(df)


if __name__ == "__main__":
    main()
