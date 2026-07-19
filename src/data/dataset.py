import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

class IEEEDataset(Dataset):
    def __init__(self, data_path, max_seq_len=100, is_train=True):
        """
        PyTorch Dataset for IEEE-CIS Fraud Detection.
        Groups transactions by ClientID and formats them for the PRAGMA model.
        """
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        
        print(f"Loading data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        
        # We assume make_dataset.py has created the 'ClientID' column
        if 'ClientID' not in self.df.columns:
            raise ValueError("ClientID column missing. Ensure make_dataset.py was run.")
            
        print("Grouping by ClientID...")
        # Group by ClientID and keep indices
        self.client_groups = self.df.groupby('ClientID').indices
        self.client_ids = list(self.client_groups.keys())
        
        # Select features for demonstration
        # Profile Categorical: card4, card6, P_emaildomain (we'll hash them to integers for embeddings)
        self.profile_cat_cols = ['card4', 'card6', 'P_emaildomain']
        
        # Profile Numerical: C1, C2, C3, C4, C5 (taking the first transaction's value)
        self.profile_num_cols = ['C1', 'C2', 'C3', 'C4', 'C5']
        
        # Event Features (Continuous/Numerical): TransactionAmt + V1 to V19 (Total 20 dims)
        self.event_cols = ['TransactionAmt'] + [f'V{i}' for i in range(1, 20)]
        
        # Simple normalization statistics based on the loaded DataFrame (for demo purposes)
        # In a real setup, these should be computed ONLY on the train set and saved.
        self.event_mean = self.df[self.event_cols].mean().fillna(0).values
        self.event_std = self.df[self.event_cols].std().fillna(1).values
        # Avoid division by zero
        self.event_std[self.event_std == 0] = 1.0

        self.profile_num_mean = self.df[self.profile_num_cols].mean().fillna(0).values
        self.profile_num_std = self.df[self.profile_num_cols].std().fillna(1).values
        self.profile_num_std[self.profile_num_std == 0] = 1.0

    def __len__(self):
        return len(self.client_ids)

    def __getitem__(self, idx):
        client_id = self.client_ids[idx]
        row_indices = self.client_groups[client_id]
        
        # Extract the client's transactions
        client_df = self.df.iloc[row_indices]
        
        # Ensure chronological order (should already be sorted, but let's be safe)
        client_df = client_df.sort_values('TransactionDT')
        
        # 1. PROFILE CATEGORICAL (Taking from the first transaction)
        # Hashing strings to simple integers for the embedding layer [0-99 range for demo]
        cat_vals = []
        for col in self.profile_cat_cols:
            val = str(client_df[col].iloc[0])
            cat_vals.append(hash(val) % 100) # Simple hash modulo for demo embeddings
        x_cat = torch.tensor(cat_vals, dtype=torch.long)
        
        # 2. PROFILE NUMERICAL (Taking from the first transaction)
        num_vals = client_df[self.profile_num_cols].iloc[0].fillna(0).values
        num_vals = (num_vals - self.profile_num_mean) / self.profile_num_std
        x_num = torch.tensor(num_vals, dtype=torch.float32)
        
        # 3. EVENTS
        # Extract event sequences and normalize
        events = client_df[self.event_cols].fillna(0).values
        events = (events - self.event_mean) / self.event_std
        
        seq_len = min(len(events), self.max_seq_len)
        events_tensor = torch.zeros((self.max_seq_len, len(self.event_cols)), dtype=torch.float32)
        events_tensor[:seq_len, :] = torch.tensor(events[:seq_len], dtype=torch.float32)

        # Extract the fraud label! (Take the label of the most recent transaction)
        if 'isFraud' in client_df.columns:
            label = int(client_df['isFraud'].iloc[-1])
        else:
            label = 0 
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return x_num, x_cat, events_tensor, seq_len, label_tensor
