import sys
sys.path.append('../')
import my_datasets as md


config = {}
# general
#config['exp_name'] = 'exps/Llama3_32layers_debug'
config['exp_name'] = 'exps_restart/Llama2_32layers_conver_bound00001layer32'
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
config['optim'] = 'sgd'  # 'adam', 'adamW', 'sgd'
config['grad_bs'] = 4
config['lr'] = 0.001
config['wd'] = 0.001
config['rho'] = 1e-4
config['sharpness_aware'] = False
config['sharpness_aware_approx'] = True
config['post_attention'] = False
config['input_attention'] = True
config['conver_bound'] = True
config['conver_loss_lambda'] = 0.0001
