import torch
import torch.nn as nn
import pdb
from types import MethodType
class RescaledLayerNormPEFT(nn.Module):
	def __init__(self, original_ln, alpha=1.0, trainable_alpha=False, mode="add"):
		super().__init__()
		self.ln = original_ln
		if trainable_alpha:
			self.alpha = nn.Parameter(torch.tensor(float(alpha)))
		else:
			self.alpha = alpha
		self.mode = mode  # "add" or "mul"
		
		# Freeze base gamma (ln.weight)
		self.ln.weight.requires_grad = False
		self.has_bias = hasattr(self.ln, 'bias') and self.ln.bias is not None
		if self.has_bias:
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
			bias=self.ln.bias if self.has_bias else None,
			eps=self.ln.eps
		)


def patch_layernorm_with_rescaled(model, alpha=1.0, trainable_alpha=False, mode="add"):
	for name, module in model.named_modules():
		if isinstance(module, nn.LayerNorm):
			parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]]
			ln_name = name.split('.')[-1]
			setattr(parent_module, ln_name, RescaledLayerNormPEFT(module, alpha=alpha, trainable_alpha=trainable_alpha, mode=mode))


def patch_layernorm_with_rescaled_by_name_old(model, alpha=1.0, mode="add", match_key="inputnorm", trainable_alpha=False):
	for name, module in model.named_modules():
		#if isinstance(module, nn.LayerNorm) and any(k in name.lower() for k in match_keywords):
		if match_key in name :
		# Identify the parent module
			parent_name = name.rsplit(".", 1)[0]
			ln_name = name.split(".")[-1]
			
			# Get reference to parent module
			parent_module = model
			for attr in parent_name.split("."):
				parent_module = getattr(parent_module, attr)
			
			# Replace the LayerNorm
			setattr(parent_module, ln_name, RescaledLayerNormPEFT(module, alpha=alpha,  trainable_alpha=trainable_alpha, mode=mode))
			print(f"Replaced LayerNorm: {name} -> RescaledLayerNormPEFT")
			


def override_llama_rmsnorm_forward(module, mode="add", alpha_init=1.0):
    hidden_size = module.weight.shape[0]

    # Inject new trainable parameters
    module.register_parameter("eta", nn.Parameter(torch.zeros_like(module.weight)))
    module.register_parameter("alpha", nn.Parameter(torch.tensor(alpha_init)))

    # Store original gamma
    module.register_buffer("original_weight", module.weight.clone())
    module.weight.requires_grad = False  # Freeze original gamma

    # Define new forward
    def custom_forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states / torch.sqrt(variance + self.variance_epsilon)

        if mode == "add":
            gamma = self.original_weight + self.alpha * self.eta
        elif mode == "mul":
            gamma = self.original_weight * (1.0 + self.alpha * self.eta)
        else:
            raise ValueError("Mode must be 'add' or 'mul'")

        return gamma * hidden_states

    # Replace forward
    module.forward = MethodType(custom_forward, module)

def patch_layernorm_with_rescaled_by_name(model, alpha=1.0, mode="add", match_key="inputnorm", trainable_alpha=False):
	for name, module in model.named_modules():
		# if isinstance(module, nn.LayerNorm) and any(k in name.lower() for k in match_keywords):
		if match_key in name:
			# Identify the parent module
			override_llama_rmsnorm_forward(module, mode=mode, alpha = alpha)
	