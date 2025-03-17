import torch
import torch.nn as nn
import pdb
class LayerNorm_Self(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm_Self, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # learnable scale
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # learnable shift

    def forward(self, x):
        # compute mean and variance along the last dimension (features)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # scale and shift
        return self.gamma * x_norm + self.beta



batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
layer_norm_self = LayerNorm_Self(embedding_dim)
# Activate module

#layer_norm(embedding)
#layer_norm_self(embedding)
# Image Example
N, C, H, W = 20, 5, 10, 10
for i in range(10):
	with torch.no_grad():
		input = torch.randn(N, C, H, W)
		# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
		# as shown in the image below
		output = layer_norm(input)
		output_self = layer_norm_self(input)
		print(f'index {i}: {torch.nn.functional.mse_loss(output, output_self).item(): .4f}')

