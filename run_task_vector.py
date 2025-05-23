import gc
import json
import copy
import random
import time
import argparse
import itertools
import torch
import pdb
from multiprocessing import Process, Queue

from huggingface_hub import login
import utils
import my_datasets as md
import evaluator as ev


def run_task(gpu_id, config, model_name, dataset_name):
    #while not task_queue.empty():
        #model_name, dataset_name = task_queue.get()
    print(f"Running {model_name} on {dataset_name} with GPU {gpu_id}")
    input_args = argparse.Namespace()
    cur_config = copy.deepcopy(config)
    input_args.model_name = model_name
    input_args.dataset_name = dataset_name
    input_args.gpu = gpu_id
    input_args.config = cur_config
    try:
        main(input_args)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"CUDA memory cleared for GPU {gpu_id}")
        time.sleep(5)
    finally:
        # Clean up CUDA memory after each task
        gc.collect()
        torch.cuda.empty_cache()
        print(f"CUDA memory cleared for GPU {gpu_id}")
        time.sleep(5)
        
def target_layer_selection(args, model_wrapper, tokenizer, evaluator, context_vector_dict):
    num_layers = model_wrapper.num_layers
    with torch.no_grad():
        best_layer = 0
        best_result = 0
        for layer in range(num_layers):
            with model_wrapper.replace_latent(context_vector_dict, [layer], args.config):
                val_result = evaluator.evaluate(model_wrapper, tokenizer, demonstration='', 
                                                use_cache=args.config['use_cache'])
                print(f'Layer {layer} result: {val_result}\n')
                if val_result[args.metric] > best_result:
                    best_result = val_result[args.metric]
                    best_layer = layer
    print(f'Best layer: {best_layer}')
    return best_layer


def main(args):
    # set global seed
    utils.set_seed(args.config['seed'])
    # set device
    args.device = utils.set_device(args.gpu)
    # set metric used
    args.metric = args.config['metric']
    # get save dir
    utils.init_exp_path(args, args.config['exp_name'])

    # load tokenizer and model
    model, tokenizer, model_config = \
    utils.load_model_tokenizer(args.model_name, args.device)
    
    # get model_wrapper
    model_wrapper = utils.get_model_wrapper(args.model_name, model, 
                                            tokenizer, model_config, 
                                            args.device)
    # load datasets
    train_dataset = md.get_dataset(args.dataset_name, split='train', max_data_num=None)
    val_dataset = md.get_dataset(args.dataset_name, split='validation', 
                                 max_data_num=args.config['val_data_num'],
                                 sample_mode=args.config['sample_method'])
    test_dataset = md.get_dataset(args.dataset_name, split='test', 
                                  max_data_num=args.config['test_data_num'],
                                  sample_mode=args.config['sample_method'])

    # get max demonstration token length for each dataset
    args.val_max_token = val_dataset.get_max_demonstration_token_length(tokenizer)
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
    
    # get shot_num
    if args.dataset_name  == 'dbpedia':  # always use 1-shot for dbpedia
        args.config['shot_per_class'] = 1
        args.config['bs'] = 1
    args.shot_num = utils.get_shot_num(train_dataset, args.config['shot_per_class'])
    # build evaluate
    val_evaluator = ev.Evaluator(val_dataset, batch_size=args.config['bs'])
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])
    # init result_dict
    result_dict = {'demon': {},
                   'split_demon': {},
                   'best_replace_layer': {},
                   'test_result': {'zero_shot': [], 'few_shot': [], 'ours': []}, 
                   'val_result': {'zero_shot': [], 'few_shot': [], 'ours': []}, 
                   'time': {'calibrate': [], 'evaluate': []},
                   }
    
    for run_id in range(args.config['run_num']):
        run_name = f'run_{run_id}'
        args.run_name = run_name
        print(f'Run time {run_name}')
        run_seed = args.config['seed'] + run_id
        utils.set_seed(run_seed)

        # zero-shot baseline
        if run_id == 0 and args.config['run_baseline']:  
            val_zeroshot_result = val_evaluator.evaluate(model_wrapper, tokenizer, demonstration='',
                                                         use_cache=args.config['use_cache'])
            test_zeroshot_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='',
                                                           use_cache=args.config['use_cache'])
            result_dict['val_result']['zero_shot'].append(val_zeroshot_result)
            result_dict['test_result']['zero_shot'].append(test_zeroshot_result)
            print(f'Validation zero-shot result: {val_zeroshot_result}\n')
            print(f'Test zero-shot result: {test_zeroshot_result}\n')

        # sample demonstration
        demon, _, _ = \
        train_dataset.gen_few_shot_demonstration(tokenizer=tokenizer, shot_num=args.shot_num, 
                                                 max_demonstration_tok_len=min(args.val_max_token, 
                                                                               args.test_max_token),
                                                 add_extra_query=args.config['add_extra_query'],
                                                 example_separator=args.config['example_separator'],
                                                 return_data_index=True, seed=random.randint(0, 1e6)
                                                 )

        if args.config['add_extra_query']:
            first_format_anchor = train_dataset.get_dmonstration_template()['format'][0]
            # remove all contents after the last first_format_anchor including the anchor
            if first_format_anchor in demon:
                baseline_demon = demon[:demon.rfind(first_format_anchor)]
                query_demon = demon[demon.rfind(first_format_anchor):]
        else:
            baseline_demon = demon
            query_demon = None
        print(f'Demonstration:\n{demon}\n')
        print(f'Baseline demonstration:\n{baseline_demon}\n')
        print(f'Query demonstration:\n{query_demon}\n')
        
        # few-shot baseline
        if args.config['run_baseline']:
            val_fewshot_result = val_evaluator.evaluate(model_wrapper, tokenizer, 
                                                        demonstration=baseline_demon, 
                                                        use_cache=args.config['use_cache'])
            test_fewshot_result = test_evaluator.evaluate(model_wrapper, tokenizer,
                                                         demonstration=baseline_demon,
                                                         use_cache=args.config['use_cache'])
            result_dict['val_result']['few_shot'].append(val_fewshot_result)
            result_dict['test_result']['few_shot'].append(test_fewshot_result)
            print(f'Validation few-shot result: {val_fewshot_result}\n')
            print(f'Test few-shot result: {test_fewshot_result}\n')

        # extract latents ======================================================================
        all_latent_dicts = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                demon_token = tokenizer(demon, return_tensors='pt').to(args.device)
                _ = model(**demon_token)
            all_latent_dicts.append(model_wrapper.latent_dict)

        # generate context vector ==============================================================
        context_vector_dict = model_wrapper.get_context_vector(all_latent_dicts, args.config)
        del all_latent_dicts

        # injection layer selection ============================================================
        best_replace_layer = target_layer_selection(args, model_wrapper, tokenizer, 
                                                    val_evaluator, context_vector_dict)
        result_dict['best_replace_layer'][run_name] = best_replace_layer

        # evaluate task_vector ========================================================================
        s_t = time.time()
        with model_wrapper.replace_latent(context_vector_dict, [best_replace_layer], args.config):
            val_ours_result = val_evaluator.evaluate(model_wrapper, tokenizer, demonstration='', 
                                                     use_cache=args.config['use_cache'])
            print(f'Validation task_vector result: {val_ours_result}\n')
            result_dict['val_result']['ours'].append(val_ours_result)

            test_ours_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='', 
                                                       use_cache=args.config['use_cache'])
            print(f'Test task_vector result: {test_ours_result}\n')
            result_dict['test_result']['ours'].append(test_ours_result)
        e_t = time.time()

        print(f'Evaluate time: {e_t - s_t}')
        result_dict['time']['evaluate'].append(e_t - s_t)

        # save result_dict after each run
        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(result_dict, f, indent=4)
    
    if config['run_baseline']:
        utils.result_mean_calculator(result_dict, 'few_shot')
    utils.result_mean_calculator(result_dict, 'ours')
    # save result_dict after each run
    with open(args.save_dir + '/result_dict.json', 'w') as f:
        json.dump(result_dict, f, indent=4)
    # delete all variables
    del model_wrapper, model, tokenizer, train_dataset, val_dataset, test_dataset
    del val_evaluator, test_evaluator, result_dict, context_vector_dict, 
            

# get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_task_vector.py', help='path to config file')
    return parser.parse_args()


if __name__ == "__main__":
    
    hf_token = "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"
    
    # Log in using the token
    login(token=hf_token)
    # get args
    args = get_args()
    # load config
    config = utils.load_config(args.config_path)
    # Generate all combinations of models and datasets
    combinations = list(itertools.product(config['models'], config['datasets']))
    # Queue to hold tasks
    print(combinations)
    for model_name, dataset_name in combinations:
        run_task('0', config, model_name, dataset_name)  # without parallelisation
    
    
    #run_task('0', config) # without parallelisation
    '''
    # Create a process for each GPU
    processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
    # Start all processes
    for p in processes:
        p.start()
    # Wait for all processes to finish
    for p in processes:
        p.join()
    print("All tasks completed.")
'''
