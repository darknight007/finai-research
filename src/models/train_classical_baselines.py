import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from pathlib import Path

def train_and_evaluate(model_name: str, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"--- Training {model_name} ---")
    
    if model_name == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, tree_method='hist')
    elif model_name == 'LightGBM':
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    elif model_name == 'CatBoost':
        model = CatBoostClassifier(iterations=200, learning_rate=0.05, random_seed=42, verbose=0)
    else:
        raise ValueError(f"Unknown model {model_name}")

    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
    test_preds = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    
    val_auc = roc_auc_score(y_val, val_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    print(f"{model_name} - Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    return {'model': model_name, 'val_auc': val_auc, 'test_auc': test_auc}

def main():
    parser = argparse.ArgumentParser(description="Train classical ML baselines")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Path to processed data')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("Loading data...")
    try:
        train_df = pd.read_parquet(data_dir / 'train.parquet')
        val_df = pd.read_parquet(data_dir / 'val.parquet')
        test_df = pd.read_parquet(data_dir / 'test.parquet')
    except Exception as e:
        print(f"Error loading data: {e}. Please run data_pipeline.ipynb first.")
        return

    # Assuming 'isFraud' is the target column
    target_col = 'isFraud'
    
    # Simple feature processing: fill missing, encode categoricals 
    # For a robust benchmark, we select a subset of dense numerical features and target encoded categoricals.
    # In a real run, this would be highly tuned. Here we drop non-numeric for simplicity of the baselines.
    features = [c for c in train_df.columns if c not in [target_col, 'TransactionID', 'TransactionDT', 'ClientID']]
    
    # Keep only numeric features for standard baselines (Tree models handle this fine, Logistic needs scaling/imputation)
    numeric_features = train_df[features].select_dtypes(include=[np.number]).columns.tolist()
    
    X_train = train_df[numeric_features].fillna(-1)
    y_train = train_df[target_col]
    
    X_val = val_df[numeric_features].fillna(-1)
    y_val = val_df[target_col]
    
    X_test = test_df[numeric_features].fillna(-1)
    y_test = test_df[target_col]

    models = ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost']
    results = []

    for model_name in models:
        res = train_and_evaluate(model_name, X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(res)
        
    print("\n--- Final Results ---")
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # Save results
    results_df.to_csv(data_dir / 'classical_baselines_results.csv', index=False)

if __name__ == '__main__':
    main()
