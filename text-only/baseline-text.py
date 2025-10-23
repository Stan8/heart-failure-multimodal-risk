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
import ast, random

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
        print(f"⚠️ Found {df['label'].isnull().sum()} missing labels — dropping those rows.")
        df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    print(f"✅ Loaded {len(df)} samples with {len(present_entities)} entity types.")
    return df, present_entities

# -------------------- Dataset Class --------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "inputs_embeds": torch.tensor(self.embeddings[idx], dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -------------------- Model --------------------
class EmbeddingClassifierConfig(PretrainedConfig):
    def __init__(self, hidden_size=768, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_labels = num_labels

class EmbeddingClassifier(PreTrainedModel):
    config_class = EmbeddingClassifierConfig

    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

    def forward(self, inputs_embeds=None, labels=None):
        logits = self.classifier(inputs_embeds)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# -------------------- Metrics --------------------
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

    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc}

# -------------------- Data Preparation --------------------
def prepare_dataset(df, entity_cols, combination_strategy="concat", projection="pca"):
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

    if len(entity_vectors) == 0:
        raise ValueError("No entity embeddings found.")
    entity_embeddings = np.stack(entity_vectors, axis=1)

    if combination_strategy == "CLS_only":
        combined_embeddings = cls_embeddings
    elif combination_strategy == "average":
        ent_avg = entity_embeddings.mean(axis=1)
        combined_embeddings = (cls_embeddings + ent_avg) / 2
    elif combination_strategy == "sum":
        ent_sum = entity_embeddings.sum(axis=1)
        combined_embeddings = cls_embeddings + ent_sum
    elif combination_strategy == "concat":
        ent_avg = entity_embeddings.mean(axis=1)
        combined_embeddings = np.concatenate([cls_embeddings, ent_avg], axis=1)
    else:
        raise ValueError(f"Unknown strategy: {combination_strategy}")

    labels = df["label"].values
    return combined_embeddings, labels

# -------------------- Cross-Validation --------------------
def run_cv_training(df, entity_cols, strategy="concat", projection="pca", n_splits=5, epochs=100, seed=42):
    combined_embeddings, y = prepare_dataset(df, entity_cols, strategy, projection)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(combined_embeddings, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ({strategy}) ---")
        X_train, X_val = combined_embeddings[train_idx], combined_embeddings[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        train_dataset = EmbeddingDataset(X_train, y_train)
        val_dataset = EmbeddingDataset(X_val, y_val)

        input_dim = X_train.shape[1]
        config = EmbeddingClassifierConfig(hidden_size=input_dim, num_labels=len(np.unique(y)))
        model = EmbeddingClassifier(config, class_weights=class_weights)

        training_args = TrainingArguments(
            output_dir=f"./results/{strategy}_fold{fold}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=epochs,
            logging_dir=f"./logs/{strategy}_fold{fold}",
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        fold_result = trainer.evaluate()
        fold_result = {k.replace("eval_", ""): v for k, v in fold_result.items() if any(m in k for m in ["precision", "recall", "f1", "auc", "loss"])}
        fold_result["fold"] = fold
        fold_metrics.append(fold_result)

    # Mean metrics per strategy
    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean(numeric_only=True).to_dict()
    mean_metrics["strategy"] = strategy
    return mean_metrics

# -------------------- Run All Strategies --------------------
def run_all_strategies(df, entity_cols, metrics_path, projection="pca", folds=5, epochs=1000):
    strategies = ["CLS_only", "average", "sum", "concat"]
    all_results = []

    for strat in strategies:
        print(f"\n Running strategy: {strat}")
        result = run_cv_training(df, entity_cols, strategy=strat, projection=projection, n_splits=folds, epochs=epochs)
        all_results.append(result)

    final_df = pd.DataFrame(all_results)
    final_df = final_df[["loss", "precision", "recall", "f1", "auc", "strategy"]]
    final_df.columns = [f"eval_{c}" if c != "strategy" else "strategy" for c in final_df.columns]
    final_df.to_csv(metrics_path, index=False)
    print(f"\n Final summary saved to {metrics_path}")
    print(final_df)
    return final_df

# -------------------- CLI Entry --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str, required=True, help="Path to CSV with CLS + entities + label")
    parser.add_argument("--metrics_path", type=str, required=True, help="Where to save metrics CSV")
    parser.add_argument("--projection", type=str, default="pca", choices=["pca", "scale"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    set_seed(SEED)
    df, entity_cols = load_embeddings(args.cls_path)
    results = run_all_strategies(df, entity_cols, args.metrics_path, projection=args.projection, folds=args.folds, epochs=args.epochs)
