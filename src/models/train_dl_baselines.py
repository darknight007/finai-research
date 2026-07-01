import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Check if torch_frame is installed
try:
    import torch_frame
    from torch_frame.config.text_embedder import TextEmbedderConfig
    from torch_frame.data import Dataset
    from torch_frame.data import DataLoader as TFDataLoader
    from torch_frame.nn.models import TabTransformer, FTTransformer, ResNet
except ImportError:
    print("torch_frame is not installed. Run `pip install pytorch-frame`.")
    torch_frame = None

def get_tf_dataset(df: pd.DataFrame, target_col: str):
    # PyTorch Frame requires column typing
    # Here we perform basic auto-typing for tabular data
    col_to_stype = {}
    for col in df.columns:
        if col == target_col:
            col_to_stype[col] = torch_frame.categorical
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_to_stype[col] = torch_frame.numerical
        else:
            col_to_stype[col] = torch_frame.categorical
            df[col] = df[col].astype(str)
            
    dataset = Dataset(df=df, col_to_stype=col_to_stype, target_col=target_col)
    dataset.materialize()
    return dataset

def train_dl_model(model_name: str, train_dataset, val_dataset, test_dataset):
    print(f"--- Training {model_name} ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = TFDataLoader(train_dataset.tensor_frame, batch_size=256, shuffle=True)
    val_loader = TFDataLoader(val_dataset.tensor_frame, batch_size=256)
    test_loader = TFDataLoader(test_dataset.tensor_frame, batch_size=256)
    
    # Initialize Model
    if model_name == 'TabTransformer':
        model = TabTransformer(
            channels=32,
            out_channels=2, # Binary classification
            num_layers=4,
            num_heads=4,
            encoder_pad_size=2,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            col_stats=train_dataset.col_stats,
            col_names_dict=train_dataset.tensor_frame.col_names_dict
        ).to(device)
    elif model_name == 'FTTransformer':
        model = FTTransformer(
            channels=32,
            out_channels=2,
            num_layers=3,
            col_stats=train_dataset.col_stats,
            col_names_dict=train_dataset.tensor_frame.col_names_dict
        ).to(device)
    elif model_name == 'ResNet': # Used as an MLP baseline representation in PyTorch Frame
        model = ResNet(
            channels=32,
            out_channels=2,
            num_layers=3,
            col_stats=train_dataset.col_stats,
            col_names_dict=train_dataset.tensor_frame.col_names_dict
        ).to(device)
    else:
        raise ValueError(f"Unknown deep learning model: {model_name}")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Simple training loop
    for epoch in range(1, 11): # 10 epochs for baseline
        model.train()
        total_loss = 0
        for tf in train_loader:
            tf = tf.to(device)
            optimizer.zero_grad()
            out = model(tf)
            loss = criterion(out, tf.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch:02d}, Loss: {total_loss / len(train_loader):.4f}")
        
    # Evaluation
    model.eval()
    val_preds, val_y = [], []
    with torch.no_grad():
        for tf in val_loader:
            tf = tf.to(device)
            out = model(tf)
            val_preds.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            val_y.append(tf.y.cpu().numpy())
            
    val_auc = roc_auc_score(np.concatenate(val_y), np.concatenate(val_preds))
    print(f"{model_name} - Val AUC: {val_auc:.4f}")
    
    return {'model': model_name, 'val_auc': val_auc}


def main():
    if torch_frame is None:
        return
        
    parser = argparse.ArgumentParser(description="Train DL baselines")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Path to processed data')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("Loading data...")
    try:
        train_df = pd.read_parquet(data_dir / 'train.parquet').head(10000) # Subset for baseline testing
        val_df = pd.read_parquet(data_dir / 'val.parquet').head(2000)
        test_df = pd.read_parquet(data_dir / 'test.parquet').head(2000)
    except Exception as e:
        print(f"Error loading data: {e}. Please run data_pipeline.ipynb first.")
        return

    target_col = 'isFraud'
    
    # We only take a small subset of features for the baseline model to run quickly
    features = [c for c in train_df.columns if c not in [target_col, 'TransactionID', 'TransactionDT', 'ClientID']]
    # Limit to top 20 numeric and 5 categorical for demonstration
    num_cols = train_df[features].select_dtypes(include=[np.number]).columns.tolist()[:20]
    cat_cols = train_df[features].select_dtypes(exclude=[np.number]).columns.tolist()[:5]
    keep_cols = num_cols + cat_cols + [target_col]
    
    train_df = train_df[keep_cols]
    val_df = val_df[keep_cols]
    test_df = test_df[keep_cols]

    print("Materializing PyTorch Frame datasets...")
    train_dataset = get_tf_dataset(train_df, target_col)
    val_dataset = get_tf_dataset(val_df, target_col)
    test_dataset = get_tf_dataset(test_df, target_col)
    
    models = ['TabTransformer', 'FTTransformer', 'ResNet']
    results = []

    for model_name in models:
        res = train_dl_model(model_name, train_dataset, val_dataset, test_dataset)
        results.append(res)
        
    print("\n--- Final Results ---")
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == '__main__':
    main()
