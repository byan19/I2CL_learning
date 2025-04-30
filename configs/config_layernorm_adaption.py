import sys
sys.path.append('../')
import my_datasets as md

config = {}
# general
#config['exp_name'] = 'exps/Llama3_32layers_debug'
config['exp_name'] = 'exps_apr26/gpt2_xl_again'
config['exp_name'] = 'exps_apr28/llama3_version4'
config['exp_name'] = 'exps_apr28/probe_raw_debugging'
config['exp_name'] = 'exps_apr28/debugging_debugging'
config['exp_name'] = 'exps_apr29/probe_raw_analysis'
config['exp_name'] = 'exps_apr29/llama3_8B_version4'
config['exp_name'] = 'exps_apr29_downloading/llama3_instruct_Lora_1'
config['exp_name'] = 'exps_apr30_ibme/gpt-j-6b'
config['exp_name'] = 'exps_apr30_ibme/gpt-j-6b'

config['gpus'] = ['0']

config['models'] = ['openai-community/gpt2-xl'] # 'meta-llama/Meta-Llama-3-8B', 'openai-community/gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'
config['models'] = ['meta-llama/Llama-2-7b-hf'] # 'meta-llama/Meta-Llama-3-8B', 'openai-community/gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'
config['models'] = ['meta-llama/Meta-Llama-3-8B'] # 'meta-llama/Meta-Llama-3-8B', 'openai-community/gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'
config['models'] = ['meta-llama/Meta-Llama-3-8B-Instruct'] # 'meta-llama/Meta-Llama-3-8B', 'openai-community/gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'
config['models'] = ['EleutherAI/gpt-j-6b'] # 'meta-llama/Meta-Llama-3-8B', 'openai-community/gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'

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
config['epochs'] = 60
config['optim'] = 'adamW'  # 'adam', 'adamW', 'sgd'
config['lr'] = 0.001
config['wd'] = 0.001
config['rho'] = 1e-4

#config['learning_type'] = 'sharpness_aware_approx' #'sharpness_encoding' # 'sharpness_aware' 'sharpness_aware_approx' 'layernorm_DyT'
config['learning_type'] = 'version4' # 'rola_analysis' #'sharpness_encoding' # 'sharpness_aware' 'sharpness_aware_approx' 'layernorm_DyT'
config['learning_type'] = 'rola_analysis' # 'rola_analysis' #'sharpness_encoding' # 'sharpness_aware' 'sharpness_aware_approx' 'layernorm_DyT'
config['eval_type'] = 'eval_with_demonstration' # 'eval_with_demonstration', 'eval_with_empty'
config['eval_type'] = 'eval_with_empty' # 'eval_with_demonstration', 'eval_with_empty'

config['grad_bs'] = 1
config['demon_bs'] = 0

#config['post_attention'] = False
#config['input_attention'] = False

# lora
config['lora_rank'] = 1 # 8, 16, 64, 128
config['lora_target_modules'] = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
config['lora_target_modules'] = ['q_proj', 'v_proj']

config['layernorm_type'] = 'all' # 'post_attention', 'input_attention', 'all'
config['entropy_loss'] = False
config['ce_loss_lambda'] = 1.0
config['pushing_loss'] = False
config['pushing_loss_lambda'] = 1e-2


config['skip_training']  = False
config['learning_property_analysis'] = False
config['probe_analysis'] = False


config['flatness_loss'] = True
config['flat_loss_lambda'] = 0.0
config['flat_loss_lambda'] = 1e-3
config['flat_loss_lambda'] = 1e-4

config['conver_loss_lambda'] = 0.0
config['conver_loss_lambda'] = 1e-2
config['conver_loss_lambda'] = 1e-3

config['noise_scale_hess'] = 1e-3


config['conver_loss'] = False
config['conver_loss_regular'] = True

config['conver_loss_regular_expo'] = False
config['conver_loss_regular_temp'] = 100
config['additional_layernorm_mode'] = 'add'

