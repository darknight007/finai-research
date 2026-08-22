import pandas as pd
import numpy as np

def generate_churn_labels(df: pd.DataFrame, client_col: str, time_col: str, inactivity_threshold: float = 30.0) -> pd.DataFrame:
    """
    Constructs churn labels based on temporal gap between consecutive transactions.
    If a client's maximum gap or tail gap exceeds inactivity_threshold, label = 1.
    """
    print("Generating synthetic churn labels...")
    df_sorted = df.sort_values(by=[client_col, time_col])
    
    # Calculate time diff between consecutive transactions per client
    df_sorted['prev_time'] = df_sorted.groupby(client_col)[time_col].shift(1)
    df_sorted['time_gap'] = df_sorted[time_col] - df_sorted['prev_time']
    
    churn_df = df_sorted.groupby(client_col).agg(
        max_gap=('time_gap', 'max'),
        txn_count=(time_col, 'count')
    ).reset_index()
    
    # A user churns if max gap between transactions > inactivity threshold or has only 1 transaction
    churn_df['is_churned'] = ((churn_df['max_gap'] > inactivity_threshold) | (churn_df['txn_count'] == 1)).astype(int)
    return churn_df[[client_col, 'is_churned']]

def generate_high_value_labels(df: pd.DataFrame, client_col: str, amount_col: str, percentile: float = 80.0) -> pd.DataFrame:
    """
    Constructs high-value customer labels based on total cumulative spend percentile.
    """
    print("Generating synthetic high-value customer labels...")
    spend_df = df.groupby(client_col)[amount_col].sum().reset_index()
    spend_threshold = np.percentile(spend_df[amount_col].values, percentile)
    spend_df['is_high_value'] = (spend_df[amount_col] >= spend_threshold).astype(int)
    return spend_df[[client_col, 'is_high_value']]

def generate_category_labels(df: pd.DataFrame, client_col: str, category_col: str) -> pd.DataFrame:
    """
    Constructs dominant category label per client.
    """
    print(f"Generating category labels from {category_col}...")
    cat_df = df.groupby(client_col)[category_col].agg(lambda x: x.mode()[0] if not x.empty else 0).reset_index()
    cat_df.columns = [client_col, 'primary_category']
    return cat_df

def generate_spend_forecast_labels(df: pd.DataFrame, client_col: str, amount_col: str) -> pd.DataFrame:
    """
    Constructs continuous spend forecasting target (total future spend).
    """
    print("Generating spend forecast targets...")
    forecast_df = df.groupby(client_col)[amount_col].agg(['sum', 'mean', 'std']).reset_index()
    forecast_df['future_spend_target'] = forecast_df['sum']
    return forecast_df[[client_col, 'future_spend_target']]

def attach_all_synthetic_labels(df: pd.DataFrame, client_col: str = 'ClientID', amount_col: str = 'TransactionAmt', time_col: str = 'TransactionDT', category_col: str = 'card4') -> pd.DataFrame:
    """
    Combines synthetic label generators and merges into user-level metadata dataframe.
    """
    churn_df = generate_churn_labels(df, client_col, time_col)
    hv_df = generate_high_value_labels(df, client_col, amount_col)
    cat_df = generate_category_labels(df, client_col, category_col)
    spend_df = generate_spend_forecast_labels(df, client_col, amount_col)
    
    merged = churn_df.merge(hv_df, on=client_col, how='left')
    merged = merged.merge(cat_df, on=client_col, how='left')
    merged = merged.merge(spend_df, on=client_col, how='left')
    return merged
