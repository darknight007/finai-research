import torch
import torch.nn as nn
import torch.nn.functional as F

class ProfileEncoder(nn.Module):
    """
    Encodes static or slow-changing user profile features.
    """
    def __init__(self, num_numerical_features, num_categorical_features, cat_embedding_dims, embed_dim=64):
        super(ProfileEncoder, self).__init__()
        # Embed categorical features
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, dim) for num_classes, dim in cat_embedding_dims
        ])
        
        # Calculate total input dimension
        total_cat_dim = sum(dim for _, dim in cat_embedding_dims)
        input_dim = num_numerical_features + total_cat_dim
        
        # MLP for profile representation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim)
        )
        
    def forward(self, x_num, x_cat):
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        if cat_embeds:
            x_cat_out = torch.cat(cat_embeds, dim=-1)
            x = torch.cat([x_num, x_cat_out], dim=-1)
        else:
            x = x_num
            
        return self.mlp(x)

class EventEncoder(nn.Module):
    """
    Encodes sequential transactional events using a Transformer.
    """
    def __init__(self, event_dim, embed_dim=64, num_heads=4, num_layers=3, max_seq_len=200):
        super(EventEncoder, self).__init__()
        
        self.event_embedding = nn.Linear(event_dim, embed_dim)
        self.positional_encoding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=256, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, events, seq_lengths):
        # events shape: (batch_size, seq_len, event_dim)
        batch_size, seq_len, _ = events.shape
        
        # Embed events
        x = self.event_embedding(events)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=events.device).unsqueeze(0).expand(batch_size, seq_len)
        x = x + self.positional_encoding(positions)
        
        # Create padding mask for variable length sequences
        mask = torch.arange(seq_len, device=events.device)[None, :] >= seq_lengths[:, None]
        
        # Transformer forward
        out = self.transformer(x, src_key_padding_mask=mask)
        
        # Aggregate (e.g., mean pooling over non-padded tokens)
        mask_float = (~mask).float().unsqueeze(-1)
        sum_embeddings = torch.sum(out * mask_float, dim=1)
        valid_counts = torch.clamp(mask_float.sum(dim=1), min=1.0)
        pooled_out = sum_embeddings / valid_counts
        
        return pooled_out, out

class PRAGMA(nn.Module):
    """
    PRAGMA Architecture combining Profile and Event representations.
    """
    def __init__(self, profile_config, event_config, embed_dim=64, num_classes=2):
        super(PRAGMA, self).__init__()
        
        self.profile_encoder = ProfileEncoder(**profile_config, embed_dim=embed_dim)
        self.event_encoder = EventEncoder(**event_config, embed_dim=embed_dim)
        
        # Cross-Attention / Fusion layer
        # For simplicity in this baseline replication, we concatenate and pass through MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim)
        )
        
        # Task specific head (downstream classification)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Self-supervised Pre-training Heads
        # Masked Event Prediction Head
        self.mlm_head = nn.Linear(embed_dim, event_config['event_dim'])
        
    def forward(self, x_num, x_cat, events, seq_lengths, pretrain=False):
        # Encode profile
        profile_embed = self.profile_encoder(x_num, x_cat)
        
        # Encode events
        event_embed, sequence_output = self.event_encoder(events, seq_lengths)
        
        # Fusion
        fused = self.fusion_mlp(torch.cat([profile_embed, event_embed], dim=-1))
        
        if pretrain:
            # During pretraining, we predict the masked events from the sequence output
            mlm_preds = self.mlm_head(sequence_output)
            return fused, mlm_preds
            
        # Downstream task
        logits = self.head(fused)
        return logits
