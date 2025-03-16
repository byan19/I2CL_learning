import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

# Hyperparameters

class TransformerLayerWithFrozenWeights(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        # Only learnable parameters: LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=True)

        # Freeze everything except LayerNorm
        for param in self.attn.parameters():
            param.requires_grad = False
        for param in self.ffn.parameters():
            param.requires_grad = False

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# Define model
class ToyTransformerModel(nn.Module):
    def __init__(self, embed_dim, n_heads, n_classes):
        super().__init__()
        self.layer1 = TransformerLayerWithFrozenWeights(embed_dim, n_heads)
        self.layer2 = TransformerLayerWithFrozenWeights(embed_dim, n_heads)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        # Mean pooling over sequence
        pooled = feat2.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, feat1, feat2

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
	model = ToyTransformerModel(embed_dim, n_heads, n_classes)

	print('optimiser definition')
	# Only optimize LayerNorm parameters
	optimizer = torch.optim.Adam([
		{'params': model.layer1.norm1.parameters()},
		{'params': model.layer1.norm2.parameters()},
		{'params': model.layer2.norm1.parameters()},
		{'params': model.layer2.norm2.parameters()},
	], lr=1e-3)

	# Fake targets
	targets = torch.randint(0, n_classes, (batch_size,))

	# Forward pass
	print('produce feature and predictions')
	logits, feat1, feat2 = model(embed_x)

	if not checking_l2_grad:
		# Cross-entropy loss
		ce_loss = F.cross_entropy(logits, targets)
		dist_loss = F.mse_loss(feat1.detach(), feat2)

		loss =  ce_loss + 0.1 * dist_loss
		ce_plot = ce_loss.item()
	else:
		dist_loss = F.mse_loss(feat1.detach(), feat2)

		loss =  0.1 * dist_loss
		ce_plot = 0.0

	pdb.set_trace()

	# Backward and optimize
	optimizer.zero_grad()
	loss.backward()
	if checking_l2_grad:
		with torch.no_grad():
			for name, ele in model.layer1.norm1.named_parameters():
				print(f'{name}: {ele.grad.norm(2): .4f}')
	optimizer.step()

	print(f"Loss: {loss.item():.4f} | CE: {ce_plot:.4f} | Dist: {dist_loss.item():.4f}")
