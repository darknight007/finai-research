import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper around an existing nn.Linear layer.
    W = W_0 + (alpha / r) * B * A
    """
    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super(LoRALinear, self).__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Freeze base layer
        self.linear = original_linear
        for p in self.linear.parameters():
            p.requires_grad = False
            
        # Trainable low-rank decomposition matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, self.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))
            # Gaussian init for A, zero init for B so initial delta W is 0
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.r > 0:
            lora_out = F.linear(x, self.lora_B @ self.lora_A) * self.scaling
            result = result + lora_out
        return result

class AdapterBottleneck(nn.Module):
    """
    Bottleneck Adapter layer inserted into transformer layers.
    d_model -> bottleneck_dim -> d_model with residual connection.
    """
    def __init__(self, d_model: int = 64, bottleneck_dim: int = 16):
        super(AdapterBottleneck, self).__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        
        # Init up_proj to zero so initial forward pass is identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.down_proj(x)
        out = self.activation(out)
        out = self.up_proj(out)
        return residual + out

def apply_lora_to_pragma(pragma_model: nn.Module, r: int = 8, alpha: float = 16.0):
    """
    Applies LoRA adapters to all self-attention linear projections in EventEncoder transformer layers,
    and freezes all base backbone weights.
    """
    # First freeze entire model
    for p in pragma_model.parameters():
        p.requires_grad = False
        
    lora_layers_added = 0
    # Inject LoRA into TransformerEncoderLayers in event_encoder.layers
    for i, layer in enumerate(pragma_model.event_encoder.layers):
        # Target in_proj_weight or QKV projections of MultiheadAttention
        attn = layer.self_attn
        if hasattr(attn, 'in_proj_weight') and attn.in_proj_weight is not None:
            # MultiheadAttention uses combined linear for QKV
            d_in = attn.embed_dim
            d_out = 3 * attn.embed_dim
            
            # Wrap linear projection
            dummy_linear = nn.Linear(d_in, d_out, bias=attn.in_proj_bias is not None)
            dummy_linear.weight = attn.in_proj_weight
            if attn.in_proj_bias is not None:
                dummy_linear.bias = attn.in_proj_bias
                
            lora_linear = LoRALinear(dummy_linear, r=r, alpha=alpha)
            # Register LoRA parameters on the layer
            setattr(layer, f'lora_qkv_{i}', lora_linear)
            lora_layers_added += 1
            
    print(f"Applied LoRA (r={r}, alpha={alpha}) to {lora_layers_added} transformer layers.")
    return pragma_model

def freeze_backbone_for_head_only(pragma_model: nn.Module):
    """
    Freezes profile_encoder, event_encoder, and fusion_mlp, leaving only task heads trainable.
    """
    for param in pragma_model.profile_encoder.parameters():
        param.requires_grad = False
    for param in pragma_model.event_encoder.parameters():
        param.requires_grad = False
    for param in pragma_model.fusion_mlp.parameters():
        param.requires_grad = False
    
    print("Backbone frozen. Only downstream task heads remain trainable.")
    return pragma_model

def count_trainable_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = (trainable / total * 100.0) if total > 0 else 0.0
    return {'trainable': trainable, 'total': total, 'percentage': percentage}
