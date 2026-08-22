import torch
import torch.nn as nn
from pragma_model import PRAGMA

class MultiTaskPRAGMA(nn.Module):
    """
    Multi-Task Foundation Model Wrapper.
    Replaces single classification head with task-specific adapter heads.
    Supported tasks: 'fraud', 'churn', 'high_value', 'category', 'spend'
    """
    def __init__(self, base_pragma: PRAGMA, embed_dim: int = 64, num_categories: int = 5):
        super(MultiTaskPRAGMA, self).__init__()
        self.profile_encoder = base_pragma.profile_encoder
        self.event_encoder = base_pragma.event_encoder
        self.fusion_mlp = base_pragma.fusion_mlp
        self.embed_dim = embed_dim
        
        # Task Heads
        self.heads = nn.ModuleDict({
            'fraud': nn.Linear(embed_dim, 2),
            'churn': nn.Linear(embed_dim, 2),
            'high_value': nn.Linear(embed_dim, 2),
            'category': nn.Linear(embed_dim, num_categories),
            'spend': nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        })

    def forward(self, x_num, x_cat, events, seq_lengths, task_name='fraud'):
        if task_name not in self.heads:
            raise ValueError(f"Unknown task {task_name}. Supported tasks: {list(self.heads.keys())}")
            
        profile_embed = self.profile_encoder(x_num, x_cat)
        event_embed, _ = self.event_encoder(events, seq_lengths)
        fused = self.fusion_mlp(torch.cat([profile_embed, event_embed], dim=-1))
        
        logits = self.heads[task_name](fused)
        if task_name == 'spend':
            return logits.squeeze(-1)
        return logits
