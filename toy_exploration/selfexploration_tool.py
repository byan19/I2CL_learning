import torch
import torch.nn as nn
import toycode_for_convergence
import pdb
class RescaledLayerNormPEFT(nn.Module):
    def __init__(self, original_ln: nn.LayerNorm, alpha=1.0, trainable_alpha = False,mode="add"):
        super().__init__()
        self.ln = original_ln
        if trainable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.alpha = alpha
        self.mode = mode  # "add" or "mul"
        
        # Freeze base gamma (ln.weight)
        self.ln.weight.requires_grad = False
        if self.ln.bias is not None:
            self.ln.bias.requires_grad = False
        
        # Learnable η, initialized to zeros
        self.eta = nn.Parameter(torch.zeros_like(self.ln.weight))
    
    def forward(self, x):
        if self.mode == "add":
            gamma = self.ln.weight + self.alpha * self.eta
        elif self.mode == "mul":
            gamma = self.ln.weight * (1.0 + self.eta)
        else:
            raise ValueError("Mode must be 'add' or 'mul'.")
        print('input the layer norm')
        return nn.functional.layer_norm(
            x,
            normalized_shape=self.ln.normalized_shape,
            weight=gamma,
            bias=self.ln.bias,
            eps=self.ln.eps
        )

def patch_layernorm_with_rescaled(model, alpha=1.0, mode="add"):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]]
            ln_name = name.split('.')[-1]
            setattr(parent_module, ln_name, RescaledLayerNormPEFT(module, alpha=alpha, mode=mode))

if __name__ == '__main__':
    seq_len = 10
    embed_dim = 32
    n_heads = 4
    n_classes = 5
    batch_size = 16
    checking_l2_grad = 1
    
    # Random input: (batch, seq_len)
    x = torch.randint(0, 100, (batch_size, seq_len))
    
    # Simple embedding layer (fixed, not learnable)
    embedding = nn.Embedding(100, embed_dim)
    with torch.no_grad():
        embed_x = embedding(x)  # (batch, seq_len, embed_dim)
    print('model definition')
    model = toycode_for_convergence.ToyTransformerModel(embed_dim, n_heads, n_classes)
    
    print('Gradient checking')
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    patch_layernorm_with_rescaled(model, alpha=0.5, mode="add")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    
    targets = torch.randint(0, n_classes, (batch_size,))
    
    # Forward pass
    print('produce feature and predictions')
    logits, feat1, feat2 = model(embed_x)