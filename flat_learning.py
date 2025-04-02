from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch
import pdb
class ToggleableNoisyLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config,layer_idx)
        self.add_noise = False  # Toggle noise
        self.noise_scale = config.noise_scale if hasattr(config, "noise_scale") else 0.01  # Default scale
        self.noise_storage = None  # Store noise

    def forward(self, hidden_states, *args, **kwargs):
        if self.add_noise:
            noise = torch.randn_like(hidden_states) * self.noise_scale
            self.noise_storage = noise
            hidden_states = hidden_states + noise
        else:
            self.noise_storage = None  # Reset when noise is off
        return super().forward(hidden_states, *args, **kwargs)
class NoisyLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.noise_outputs = []  # Store per-layer noise
        self.noise_scale = config.noise_scale if hasattr(config, "noise_scale") else 0.01

        # Replace all layers with the custom noisy layer
        for i in range(len(self.model.layers)):
            self.model.layers[i] = ToggleableNoisyLlamaDecoderLayer(config, layer_idx=i)

    def set_noise(self, add_noise=True):
        """Toggle noise injection on/off for all layers."""
        for layer in self.model.layers:
            layer.add_noise = add_noise

    def set_noise_scale(self, scale):
        """Adjust noise scale dynamically."""
        self.noise_scale = scale
        for layer in self.model.layers:
            layer.noise_scale = scale  # Update noise scale for all layers

    def forward(self, *args, **kwargs):
        self.noise_outputs = []  # Reset stored noise
        outputs = super().forward(*args, **kwargs)

        # Collect noise from all layers
        for layer in self.model.layers:
            if layer.noise_storage is not None:
                self.noise_outputs.append(layer.noise_storage)

        return outputs, self.noise_outputs  # Return noise along with model output


class NoiseInjector:
    def __init__(self, noise_scale=0.01):
        self.add_noise = False  # Toggle noise injection
        self.noise_scale = noise_scale  # Default noise scale
        self.noise_outputs = []  # Store noise tensors per layer

    def hook_fn(self, module, input):
        """Function to add noise and store it."""
        if self.add_noise:
            print('add noise inside')
            noise = torch.randn_like(input[0]) * self.noise_scale
            self.noise_outputs.append(noise)  # Store noise per layer
            #input = input[0].detach().data +  noise
            #print('add noise zeros')
            input = input[0] + noise
            return (input,)
        else:
            print('no noise inside')
            return (input,)

    def set_noise(self, status: bool):
        """Enable/disable noise injection and reset stored noise."""
        self.add_noise = status
        self.noise_outputs = []  # Reset stored noise when toggling

    def set_noise_scale(self, scale: float):
        """Dynamically adjust noise scale."""
        self.noise_scale = scale
    
    def remove_noise(self):
        self.noise_outputs = []
        
def hook_fn_test(module, input ):
    """Function to add noise and store it."""
    print('add noise inside')
    noise = torch.randn_like(input[0])
    input = torch.zeros_like(input[0])
    return (input,)
