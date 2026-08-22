import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

class PaySimDataset(Dataset):
    """
    PyTorch Dataset for PaySim Mobile Money Data.
    Groups transactions by nameOrig and maps them to PRAGMA format.
    """
    def __init__(self, data_path, max_seq_len=200):
        self.max_seq_len = max_seq_len
        print(f"Loading PaySim parquet data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        
        if 'nameOrig' not in self.df.columns:
            raise ValueError("nameOrig column missing in PaySim data.")
            
        self.user_groups = self.df.groupby('nameOrig').indices
        self.user_ids = list(self.user_groups.keys())
        
        self.profile_num_cols = ['avg_amount', 'txn_count', 'avg_balance', 'balance_volatility']
        self.profile_cat_cols = ['primary_type']
        
        # Event feature set: [type_id, amount, oldbalanceOrg, newbalanceOrig, balance_delta_orig, balance_delta_dest]
        self.event_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_delta_orig', 'balance_delta_dest']
        
        # Normalization stats
        self.profile_num_mean = self.df[self.profile_num_cols].mean().fillna(0).values
        self.profile_num_std = self.df[self.profile_num_cols].std().fillna(1.0).values
        self.profile_num_std[self.profile_num_std == 0] = 1.0
        
        self.event_mean = self.df[self.event_cols].mean().fillna(0).values
        self.event_std = self.df[self.event_cols].std().fillna(1.0).values
        self.event_std[self.event_std == 0] = 1.0

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        row_indices = self.user_groups[user_id]
        user_df = self.df.iloc[row_indices].sort_values('step')
        
        # 1. Profile Numerical
        p_num = user_df[self.profile_num_cols].iloc[0].fillna(0).values
        p_num_norm = (p_num - self.profile_num_mean) / self.profile_num_std
        x_num = torch.tensor(p_num_norm, dtype=torch.float32)
        
        # 2. Profile Categorical
        p_cat = int(user_df['primary_type'].iloc[0]) % 10
        x_cat = torch.tensor([p_cat], dtype=torch.long)
        
        # 3. Events
        type_ids = user_df['type_id'].values
        event_nums = user_df[self.event_cols].fillna(0).values
        event_nums_norm = (event_nums - self.event_mean) / self.event_std
        
        seq_len = min(len(user_df), self.max_seq_len)
        events_tensor = torch.zeros((self.max_seq_len, 6), dtype=torch.float32)
        
        for t in range(seq_len):
            events_tensor[t, 0] = float(type_ids[t])
            events_tensor[t, 1:] = torch.tensor(event_nums_norm[t], dtype=torch.float32)
            
        label = int(user_df['isFraud'].iloc[-1]) if 'isFraud' in user_df.columns else 0
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return x_num, x_cat, events_tensor, seq_len, label_tensor
