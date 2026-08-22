import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Try importing config
try:
    import config
    paths = config.setup_environment()
    DEFAULT_RAW_DIR = paths['data_raw']
    DEFAULT_PROCESSED_DIR = paths['data_processed']
except ImportError:
    DEFAULT_RAW_DIR = Path('data/raw')
    DEFAULT_PROCESSED_DIR = Path('data/processed')

TYPE_MAP = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}

def generate_mock_paysim(num_rows=50000):
    """Generate synthetic PaySim data if raw Kaggle CSV is not present for local execution/testing."""
    print(f"Generating synthetic PaySim data with {num_rows} rows...")
    np.random.seed(42)
    users = [f"C{np.random.randint(10000, 99999)}" for _ in range(2000)]
    merchants = [f"M{np.random.randint(10000, 99999)}" for _ in range(500)]
    types = list(TYPE_MAP.keys())
    
    steps = np.random.randint(1, 744, size=num_rows)
    steps.sort()
    
    data = []
    for step in steps:
        orig = np.random.choice(users)
        dest = np.random.choice(users if np.random.rand() > 0.5 else merchants)
        t_type = np.random.choice(types, p=[0.35, 0.25, 0.25, 0.10, 0.05])
        amount = round(float(np.random.exponential(scale=500.0) + 1.0), 2)
        old_orig = round(float(np.random.uniform(0, 50000)), 2)
        new_orig = max(0.0, old_orig - amount) if t_type in ['TRANSFER', 'CASH_OUT', 'PAYMENT'] else old_orig + amount
        old_dest = round(float(np.random.uniform(0, 50000)), 2)
        new_dest = old_dest + amount
        is_fraud = 1 if (t_type in ['TRANSFER', 'CASH_OUT'] and amount > 20000 and np.random.rand() < 0.3) else 0
        
        data.append({
            'step': step,
            'type': t_type,
            'amount': amount,
            'nameOrig': orig,
            'oldbalanceOrg': old_orig,
            'newbalanceOrig': new_orig,
            'nameDest': dest,
            'oldbalanceDest': old_dest,
            'newbalanceDest': new_dest,
            'isFraud': is_fraud,
            'isFlaggedFraud': 0
        })
    return pd.DataFrame(data)

def load_or_create_paysim(raw_dir: Path) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_file = raw_dir / 'PS_20174392719_1491208443941_log.csv'
    alt_file = raw_dir / 'paysim.csv'
    
    if csv_file.exists():
        print(f"Loading PaySim data from {csv_file}...")
        df = pd.read_csv(csv_file)
    elif alt_file.exists():
        print(f"Loading PaySim data from {alt_file}...")
        df = pd.read_csv(alt_file)
    else:
        print("PaySim raw dataset not found in data/raw. Generating synthetic sample dataset...")
        df = generate_mock_paysim(num_rows=100000)
        df.to_csv(alt_file, index=False)
    return df

def process_paysim_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Processing PaySim features into profile and event sequences...")
    df['type_id'] = df['type'].map(lambda x: TYPE_MAP.get(str(x).upper(), 0))
    df['balance_delta_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_delta_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Calculate user-level summary profile attributes (simulated profile state)
    profile_df = df.groupby('nameOrig').agg(
        avg_amount=('amount', 'mean'),
        txn_count=('amount', 'count'),
        avg_balance=('oldbalanceOrg', 'mean'),
        balance_volatility=('oldbalanceOrg', 'std'),
        primary_type=('type_id', lambda x: x.mode()[0] if not x.empty else 0)
    ).reset_index()
    
    profile_df['balance_volatility'] = profile_df['balance_volatility'].fillna(0.0)
    
    # Merge profile attributes back
    merged = df.merge(profile_df, on='nameOrig', how='left')
    merged = merged.sort_values(by=['nameOrig', 'step'])
    return merged

def create_paysim_splits(df: pd.DataFrame, processed_dir: Path):
    processed_dir.mkdir(parents=True, exist_ok=True)
    max_step = df['step'].max()
    min_step = df['step'].min()
    total_steps = max_step - min_step
    
    train_cutoff = min_step + total_steps * 0.70
    val_cutoff = min_step + total_steps * 0.85
    
    train_df = df[df['step'] <= train_cutoff]
    val_df = df[(df['step'] > train_cutoff) & (df['step'] <= val_cutoff)]
    test_df = df[df['step'] > val_cutoff]
    
    print(f"PaySim splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    train_df.to_parquet(processed_dir / 'paysim_train.parquet')
    val_df.to_parquet(processed_dir / 'paysim_val.parquet')
    test_df.to_parquet(processed_dir / 'paysim_test.parquet')
    print(f"PaySim dataset splits saved to {processed_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default=str(DEFAULT_RAW_DIR))
    parser.add_argument('--processed_dir', type=str, default=str(DEFAULT_PROCESSED_DIR))
    args = parser.parse_args()
    
    df = load_or_create_paysim(Path(args.raw_dir))
    df = process_paysim_features(df)
    create_paysim_splits(df, Path(args.processed_dir))

if __name__ == '__main__':
    main()
