import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

def load_data(raw_dir: Path) -> pd.DataFrame:
    """
    Loads and merges the IEEE-CIS Fraud Detection transaction and identity datasets.
    """
    print(f"Loading data from {raw_dir}...")
    try:
        train_transaction = pd.read_csv(raw_dir / 'train_transaction.csv')
        train_identity = pd.read_csv(raw_dir / 'train_identity.csv')
        
        # Merge on TransactionID
        df = train_transaction.merge(train_identity, on='TransactionID', how='left')
        print(f"Successfully loaded and merged data. Shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please download the IEEE-CIS Fraud Detection dataset from Kaggle and place it in the 'data/raw' directory.")
        print("Required files: 'train_transaction.csv', 'train_identity.csv'")
        raise

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps the dataset into 'profile' features and 'event' stream structures 
    as defined by the PRAGMA paper.
    """
    print("Processing features...")
    
    # In PRAGMA, we need:
    # 1. Profile features (static/slow-changing user attributes)
    # 2. Event streams (temporal sequence of transactions/interactions)
    
    # For IEEE-CIS, we don't have explicit user IDs, but we can approximate it using card details and address.
    # We will use card1-card6, addr1, addr2, and email domains to form a pseudo-clientID.
    
    # Fill NAs for grouping
    grouping_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain']
    for col in grouping_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = df[col].fillna(-1)
            
    # Create pseudo ClientID
    df['ClientID'] = df[grouping_cols].astype(str).agg('-'.join, axis=1)
    
    # Sort by time (TransactionDT is a timedelta from a given reference datetime)
    df = df.sort_values(by=['ClientID', 'TransactionDT'])
    
    print("Feature processing complete.")
    return df

def create_splits(df: pd.DataFrame, processed_dir: Path):
    """
    Generates deterministic train/validation/test splits by time.
    """
    print("Creating time-based train/val/test splits...")
    
    # Time-based split to prevent data leakage in sequential models
    total_time = df['TransactionDT'].max() - df['TransactionDT'].min()
    val_split_time = df['TransactionDT'].min() + total_time * 0.7  # First 70% for train
    test_split_time = df['TransactionDT'].min() + total_time * 0.85 # Next 15% for val, last 15% for test
    
    train_df = df[df['TransactionDT'] <= val_split_time]
    val_df = df[(df['TransactionDT'] > val_split_time) & (df['TransactionDT'] <= test_split_time)]
    test_df = df[df['TransactionDT'] > test_split_time]
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(processed_dir / 'train.parquet')
    val_df.to_parquet(processed_dir / 'val.parquet')
    test_df.to_parquet(processed_dir / 'test.parquet')
    
    print(f"Data splits saved to {processed_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process IEEE-CIS Fraud Data for PRAGMA.")
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='Path to raw data directory.')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Path to save processed data.')
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    
    df = load_data(raw_dir)
    df = process_features(df)
    create_splits(df, processed_dir)

if __name__ == '__main__':
    main()
