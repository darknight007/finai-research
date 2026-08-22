import os
import sys
from pathlib import Path

def is_colab():
    """Detect if running inside Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_environment():
    """
    Auto-detect environment, mount Drive if in Colab, and resolve key project paths.
    """
    if is_colab():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as e:
            print(f"Warning mounting drive: {e}")
        PROJECT_ROOT = Path('/content/drive/MyDrive/Finai-research')
    else:
        # Default to standard project directory
        PROJECT_ROOT = Path(__file__).resolve().parent.parent

    paths = {
        'project_root': PROJECT_ROOT,
        'src': PROJECT_ROOT / 'src',
        'data_raw': PROJECT_ROOT / 'data' / 'raw',
        'data_processed': PROJECT_ROOT / 'data' / 'processed',
        'models': PROJECT_ROOT / 'models',
        'results': PROJECT_ROOT / 'results',
        'notebooks': PROJECT_ROOT / 'notebooks',
        'research_notes': PROJECT_ROOT / 'research_notes'
    }

    # Ensure output directories exist
    for k, p in paths.items():
        if k != 'project_root' and not p.suffix:
            p.mkdir(parents=True, exist_ok=True)

    if str(paths['src']) not in sys.path:
        sys.path.insert(0, str(paths['src']))

    return paths

# Centralized Model Configurations
PRAGMA_IEEE_PROFILE_CONFIG = {
    'num_numerical_features': 5,    # C1-C5
    'num_categorical_features': 3,  # card4, card6, P_emaildomain
    'cat_embedding_dims': [(100, 8), (100, 8), (100, 8)]
}

PRAGMA_IEEE_EVENT_CONFIG = {
    'event_dim': 20,      # TransactionAmt + V1-V19
    'embed_dim': 64,
    'num_heads': 4,
    'num_layers': 3,
    'max_seq_len': 100
}

PRAGMA_PAYSIM_PROFILE_CONFIG = {
    'num_numerical_features': 4,    # avg_amount, txn_count, avg_balance, balance_volatility
    'num_categorical_features': 1,  # primary_type
    'cat_embedding_dims': [(10, 4)]
}

PRAGMA_PAYSIM_EVENT_CONFIG = {
    'event_dim': 6,       # type_id(embed 4) + amount + balance_delta
    'embed_dim': 64,
    'num_heads': 4,
    'num_layers': 3,
    'max_seq_len': 200
}

COLAB_SETUP_CELL_CODE = '''# Colab Setup Cell
import os, sys
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = '/content/drive/MyDrive/Finai-research'
    if not os.path.exists(PROJECT_ROOT):
        !git clone https://github.com/outlieralpha/Finai-research.git {PROJECT_ROOT}
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    !pip install -q torch xgboost lightgbm catboost scikit-learn pandas numpy pyarrow matplotlib seaborn tqdm
    !mkdir -p ~/.kaggle
    !cp {PROJECT_ROOT}/kaggle.json ~/.kaggle/ 2>/dev/null || true
    !chmod 600 ~/.kaggle/kaggle.json 2>/dev/null || true
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import config
paths = config.setup_environment()
print("Environment ready. GPU available:", __import__('torch').cuda.is_available())
'''
