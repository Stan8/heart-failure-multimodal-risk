import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import argparse
import os

def parse_cls_column(cls_series):
    """Convert CLS column string to numpy array (handles comma-separated format)."""
    return cls_series.apply(lambda x: np.array([float(v) for v in x.strip('[]').split(',')]))

def load_and_merge(structured_path, cls_path):
    """Load structured and CLS embeddings, merge on NO_SEJOUR, and expand CLS into columns."""
    df_struct = pd.read_excel(structured_path)
    df_cls = pd.read_csv(cls_path)
    df_cls['CLS'] = parse_cls_column(df_cls['CLS'])
    df_cls = df_cls.rename(columns={"id": "NO_SEJOUR"})

    df = pd.merge(df_struct, df_cls[['NO_SEJOUR', 'CLS']], on='NO_SEJOUR')

    # Expand CLS embeddings into separate columns
    cls_expanded = pd.DataFrame(df['CLS'].tolist(), index=df.index)
    cls_expanded.columns = [f'CLS_{i}' for i in range(cls_expanded.shape[1])]
    df = pd.concat([df.drop(columns=['CLS']), cls_expanded], axis=1)
    return df

def compute_metrics(y_true, y_pred, y_proba):
    """Compute classification metrics."""
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }

def run_cross_validation(df, output_dir, n_splits=5, random_state=42):
    """Run Stratified K-Fold cross-validation for multimodal logistic regression."""
    os.makedirs(output_dir, exist_ok=True)

    y = df['y'].values
    X = df.drop(columns=['y'])
    cls_cols = [col for col in X.columns if col.startswith('CLS_')]
    struct_cols = [col for col in X.columns if col not in cls_cols + ['NO_SEJOUR']]

    imputer = SimpleImputer(strategy='constant', fill_value=0)
    scaler = StandardScaler()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_metrics = []
    all_preds = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Cross-validation")):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Impute and scale structured features
        X_struct_train = pd.DataFrame(imputer.fit_transform(X_train[struct_cols]), columns=struct_cols)
        X_struct_test = pd.DataFrame(imputer.transform(X_test[struct_cols]), columns=struct_cols)
        X_struct_train = pd.DataFrame(scaler.fit_transform(X_struct_train), columns=struct_cols)
        X_struct_test = pd.DataFrame(scaler.transform(X_struct_test), columns=struct_cols)

        # CLS embeddings (not imputed/scaled)
        X_cls_train = X_train[cls_cols].reset_index(drop=True)
        X_cls_test = X_test[cls_cols].reset_index(drop=True)

        # Combine both
        X_train_final = pd.concat([X_struct_train.reset_index(drop=True), X_cls_train], axis=1)
        X_test_final = pd.concat([X_struct_test.reset_index(drop=True), X_cls_test], axis=1)

        # Model
        model = LogisticRegression(
            class_weight='balanced',
            solver='liblinear',
            max_iter=1000,
            random_state=random_state
        )
        model.fit(X_train_final, y_train)

        # Predict
        y_proba = model.predict_proba(X_test_final)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics['fold'] = fold
        all_metrics.append(metrics)

        # Save predictions
        fold_preds = pd.DataFrame({
            'NO_SEJOUR': X_test['NO_SEJOUR'],
            'fold': fold,
            'y_true': y_test,
            'y_proba': y_proba
        })
        all_preds.append(fold_preds)

    # Aggregate results
    metrics_df = pd.DataFrame(all_metrics)
    mean_metrics = metrics_df.mean(numeric_only=True)
    std_metrics = metrics_df.std(numeric_only=True)
    summary = pd.DataFrame({'mean': mean_metrics, 'std': std_metrics})

    preds_df = pd.concat(all_preds, ignore_index=True)

    metrics_df.to_csv(os.path.join(output_dir, "metrics_per_fold.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"))
    preds_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    print("\nCross-validation results:")
    print(summary)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structured', required=True, help='Path to structured Excel file')
    parser.add_argument('--cls', required=True, help='Path to CLS CSV file')
    parser.add_argument('--output_dir', required=True, help='Directory to save cross-validation results')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation folds')
    args = parser.parse_args()

    df = load_and_merge(args.structured, args.cls)
    run_cross_validation(df, output_dir=args.output_dir, n_splits=args.n_splits)

if __name__ == '__main__':
    main()
