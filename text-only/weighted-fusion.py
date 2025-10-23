import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PretrainedConfig, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random, ast, math


# -------------------- Config --------------------
CLS_COLUMN = "CLS"
ENTITY_COLUMNS = [
    'AGE', 'PATHOLOGIE', 'SIGNE_SYMPTOME', 'TRAITEMENT', 'ANATOMIE', 'EXAMEN',
    'ENTOURAGE', 'AUTONOMIE', 'CONCENTRATION', 'MODE', 'DOSE', 'FREQUENCE',
    'PARAMETRE_MESURABLE', 'VALEUR', 'NEGATION', 'HYPOTHETIQUE',
    'EVOLUTION_TRAITEMENT_PARAMETRE', 'COMPORTEMENT', 'DUREE', 'EVOLUTION',
    'CHANGEMENT_LIEU', 'LIEU', 'DATE'
]
SEED = 42
N_FOLDS = 5

# -------------------- Reproducibility --------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------- Parsing --------------------
def parse_embedding_column(col: pd.Series) -> pd.Series:
    """Convert stringified embeddings into np arrays."""
    return col.apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32) if isinstance(x, str) else np.array(x, dtype=np.float32))

# -------------------- Load CLS + Entities --------------------
def load_embeddings(cls_path: str):
    df = pd.read_csv(cls_path)

    if CLS_COLUMN not in df.columns:
        raise ValueError(f"Missing '{CLS_COLUMN}' column in {cls_path}")

    df[CLS_COLUMN] = parse_embedding_column(df[CLS_COLUMN])

    present_entities = []
    for col in ENTITY_COLUMNS:
        if col in df.columns:
            df[col] = parse_embedding_column(df[col])
            present_entities.append(col)

    if 'label' not in df.columns:
        raise ValueError("Expected a 'label' column in the CLS CSV")

    if df['label'].isnull().any():
        print(f" Found {df['label'].isnull().sum()} missing labels — dropping those rows.")
        df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(len(df))

    return df, present_entities

# -------------------- Dataset --------------------
class WeightedFusionDataset(Dataset):
    def __init__(self, cls_embeddings, entity_embeddings, labels):
        self.cls_embeddings = cls_embeddings
        self.entity_embeddings = entity_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "cls_embedding": torch.tensor(self.cls_embeddings[idx], dtype=torch.float),
            "entity_embeddings": torch.tensor(self.entity_embeddings[idx], dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class WeightedFusionDataCollator:
    def __call__(self, features):
        cls_embeds = torch.stack([f["cls_embedding"] for f in features])
        entity_embeds = torch.stack([f["entity_embeddings"] for f in features])
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {"cls_embedding": cls_embeds, "entity_embeddings": entity_embeds, "labels": labels}

# -------------------- Model --------------------
class WeightedFusionConfig(PretrainedConfig):
    def __init__(self, hidden_size=768, num_entities=23, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_entities = num_entities
        self.num_labels = num_labels

class WeightedFusionClassifier(PreTrainedModel):
    config_class = WeightedFusionConfig

    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.cls_weight = nn.Parameter(torch.tensor(1.0))
        self.entity_weights = nn.Parameter(torch.ones(config.num_entities))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

    def forward(self, cls_embedding=None, entity_embeddings=None, labels=None):
        norm_entity_weights = torch.softmax(self.entity_weights, dim=0)
        weighted_entity_emb = (entity_embeddings * norm_entity_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        combined = self.cls_weight * cls_embedding + weighted_entity_emb
        logits = self.classifier(combined)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# -------------------- Metric Computation --------------------
def compute_metrics(p):
    probs = p.predictions if not isinstance(p.predictions, tuple) else p.predictions[0]
    preds = np.argmax(probs, axis=1)
    labels = p.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1, zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = np.nan

    return {'precision': precision, 'recall': recall, 'f1_score': f1, 'roc_auc': auc}

# -------------------- Cross-Validation --------------------
def run_cv_weighted(df, entity_cols, metrics_path, folds=N_FOLDS, projection="pca"):
    cls_embeddings = np.stack(df[CLS_COLUMN])
    cls_dim = cls_embeddings.shape[1]

    entity_vectors = []
    for col in entity_cols:
        arr = np.stack(df[col])
        if arr.shape[1] != cls_dim:
            if projection == "pca":
                arr = PCA(n_components=cls_dim).fit_transform(arr)
            elif projection == "scale":
                arr = StandardScaler().fit_transform(arr)
        entity_vectors.append(arr)
    entity_embeddings = np.stack(entity_vectors, axis=1)
    y = df["label"].values

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(cls_embeddings, y)):
        print(f"\n Fold {fold+1}/{folds}")

        X_cls_train, X_cls_test = cls_embeddings[train_idx], cls_embeddings[test_idx]
        X_ent_train, X_ent_test = entity_embeddings[train_idx], entity_embeddings[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        train_dataset = WeightedFusionDataset(X_cls_train, X_ent_train, y_train)
        test_dataset = WeightedFusionDataset(X_cls_test, X_ent_test, y_test)

        config = WeightedFusionConfig(hidden_size=X_cls_train.shape[1], num_entities=X_ent_train.shape[1], num_labels=2)
        model = WeightedFusionClassifier(config, class_weights=class_weights)

        args = TrainingArguments(
            output_dir=f"./results_weighted/fold_{fold}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=500,
            logging_dir="./logs",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=WeightedFusionDataCollator(),
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate()

        # keep only precision, recall, f1, auc (removing "eval_" prefix)
        metrics = {k.replace("eval_", ""): v for k, v in metrics.items() if any(m in k for m in ["precision", "recall", "f1_score", "roc_auc"])}
        metrics["fold"] = fold + 1
        all_metrics.append(metrics)

        print(f"Fold {fold+1} results:")
        for k, v in metrics.items():
            if k != "fold":
                print(f"  {k}: {v:.4f}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.loc["mean"] = metrics_df.mean(numeric_only=True)
    metrics_df.loc["std"] = metrics_df.std(numeric_only=True)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n Saved metrics to {metrics_path}")
    print(metrics_df)
    return metrics_df

# -------------------- Main --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str, required=True, help="Path to CSV containing CLS + entity embeddings + label")
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to save metrics CSV")
    parser.add_argument("--projection", type=str, default="pca", choices=["pca", "scale"])
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    set_seed(SEED)
    df, entity_cols = load_embeddings(args.cls_path)
    print(f" Loaded {len(df)} samples with {len(entity_cols)} entity types")

    results = run_cv_weighted(df, entity_cols, metrics_path=args.metrics_path, folds=args.folds, projection=args.projection)
