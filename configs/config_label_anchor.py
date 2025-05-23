import sys
sys.path.append('../')
import my_datasets as md


config = {}
# general
config['exp_name'] = 'exps_May4_yibo/label_anchor_llama3'

config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-2-7b-hf'] # 'gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'
config['models'] = ['meta-llama/Meta-Llama-3-8B-Instruct'] # 'meta-llama/Meta-Llama-3-8B', 'openai-community/gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'
config['models'] = ['meta-llama/Meta-Llama-3-8B'] # 'meta-llama/Meta-Llama-3-8B', 'openai-community/gpt2-xl', 'meta-llama/Llama-2-7b-hf', 'EleutherAI/gpt-j-6B'

#config['datasets'] = list(md.dataset_classification.keys())
config['datasets'] = list(md.target_datasets.keys())
config['seed'] = 42
config['run_num'] = 5
config['run_baseline'] = False
config['metric'] = 'acc'  # 'acc', 'macro_f1'
config['bs'] = 2
config['load_in_8bit'] = False
config['use_cache'] = True

# data
config['shot_per_class'] = 5
config['test_data_num'] = 500
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = False
config['example_separator'] = '\n'