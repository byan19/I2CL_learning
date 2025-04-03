import sys
sys.path.append('../')
import my_datasets as md


config = {}
# general
#config['exp_name'] = 'exps/Llama3_32layers_debug'
#config['exp_name'] = 'exps_restart/Llama2_32layers_layernorm_conv_ratio_normaliser_debug'
#config['exp_name'] = 'exps_restart2/Llama2_32layers_layernorm_1e-5AdamW'
#config['exp_name'] = 'exps_restart3/Llama2_32layers_allLN_convregular1e-2_trajcontroltmp10'
config['exp_name'] = 'exps_apr3/Llama2_32layers_sharpnessencoding_softplushdeskalone'
config['exp_name'] = 'exps_apr3/Llama2_32layers_sharpnessencoding_softplusN1e-2'
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-2-7b-hf'] # 'meta-llama/Meta-Llama-3-8B', 'gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'
config['datasets'] = list(md.target_datasets.keys())
config['seed'] = 42
config['run_num'] = 5
config['run_baseline'] = False
config['metric'] = 'acc'  # 'acc', 'macro_f1'
config['bs'] = 1
config['load_in_8bit'] = False
config['use_cache'] = True
config['example_separator'] = '\n'

# data
config['shot_per_class'] = 5
config['test_data_num'] = 500
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['add_extra_query'] = False

# prompt_tuning
pt_config = {}
pt_config['task_type'] = 'CAUSAL_LM'
pt_config['num_virtual_tokens'] = 1
pt_config['num_layers'] = 32
config['pt_config'] = pt_config


# optimization
config['epochs'] = 30
config['optim'] = 'adamW'  # 'adam', 'adamW', 'sgd'
config['grad_bs'] = 1
config['lr'] = 0.001
config['wd'] = 0.001
config['rho'] = 1e-4

#config['sharpness_aware'] = False
#config['sharpness_aware_approx'] = True
#config['layernorm_DyT'] = False

#config['learning_type'] = 'sharpness_aware_approx' #'sharpness_encoding' # 'sharpness_aware' 'sharpness_aware_approx' 'layernorm_DyT'
config['learning_type'] = 'sharpness_encoding' #'sharpness_encoding' # 'sharpness_aware' 'sharpness_aware_approx' 'layernorm_DyT'

config['post_attention'] = False
config['input_attention'] = False
config['entropy_loss'] = False
config['ce_loss_lambda'] = 1.0
config['conver_bound'] = False
config['pushing_loss'] = False
config['pushing_loss_lambda'] = 1e-2
config['conver_loss_lambda'] = 1e-2
config['flat_loss_lambda'] = 1e-2
config['additional_layernorm_mode'] = 'add'
config['noise_scale_hess'] = 1e-2
config['conver_loss'] = False
config['conver_loss_regular'] = False
config['conver_loss_regular_expo'] = False
config['conver_loss_regular_temp'] = 100

