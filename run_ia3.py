import gc
import json
import copy
import time
import random
import argparse
import itertools
import torch
from multiprocessing import Process, Queue

import utils
import evaluator as ev
import my_datasets as md
from huggingface_hub import login
import pdb

def run_task(gpu_id, config, model_name, dataset_name):
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

def main(args):
    # set global seed
    utils.set_seed(args.config['seed'])
    # set device
    
    args.device = utils.set_device(args.gpu)
    # set comprare metric
    args.metric = args.config['metric']
    # get save dir
    utils.init_exp_path(args, args.config['exp_name'])

    # load datasets
    train_dataset = md.get_dataset(args.dataset_name, split='train', max_data_num=None)
    test_dataset = md.get_dataset(args.dataset_name, split='test', 
                                  max_data_num=args.config['test_data_num'],
                                  sample_mode=args.config['sample_method'])
    # load tokenizer and model
    model, tokenizer, model_config = utils.load_model_tokenizer(args.model_name, args.device)
    # get model_wrapper
    model_wrapper = utils.get_model_wrapper(args.model_name, model,
                                            tokenizer, model_config,
                                            args.device)

    # get max demonstration token length for each dataset
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
    
    # get shot_num
    if args.dataset_name  == 'dbpedia':  # always use 1-shot for dbpedia
        args.config['shot_per_class'] = 1
        args.config['bs'] = 1
    args.shot_num = utils.get_shot_num(train_dataset, args.config['shot_per_class'])
    # build evaluator
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])

    # init result_dict
    result_dict = {'demon': {},
                   'test_result': {'zero_shot': [], 'few_shot': [], 'ours': []}, 
                   'time': {'calibrate': [], 'evaluate': []},
                   }

    for run_id in range(args.config['run_num']):
        run_name = f'run_{run_id}'
        args.run_name = run_name
        print(f'Run time {run_name}')
        run_seed = args.config['seed'] + run_id
        utils.set_seed(run_seed)        

        # zero-shot baseline
        zero_shot_prompt = 'classify the type of the given sentence. '
        zero_shot_prompt = ''
        if run_id == 0 and args.config['run_baseline']:
            test_zeroshot_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration=zero_shot_prompt,
                                                           use_cache=args.config['use_cache'])
            result_dict['test_result']['zero_shot'].append(test_zeroshot_result)
            print(f'Test zero-shot result: {test_zeroshot_result}\n')
        
        #pdb.set_trace()
        # sample demonstration
        demon, _, demon_data_index = train_dataset.gen_few_shot_demonstration(tokenizer=tokenizer, shot_num=args.shot_num,
                                                 max_demonstration_tok_len=args.test_max_token,
                                                 add_extra_query=args.config['add_extra_query'],
                                                 example_separator=args.config['example_separator'],
                                                 return_data_index=True, seed=random.randint(0, 1e6))
        
        
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
        #pdb.set_trace()
        if args.config['run_baseline']:
            test_fewshot_result = test_evaluator.evaluate(model_wrapper, tokenizer,
                                                          demonstration=baseline_demon,
                                                          use_cache=args.config['use_cache'])
            result_dict['test_result']['few_shot'].append(test_fewshot_result)
            print(f'Test few-shot result: {test_fewshot_result}\n')

        # generate demon_list
        demon_list = [demon]
        # save demon_list
        result_dict['demon'][run_name] = demon_list

        # prepare peft_train_dataset
        cali_dataset = copy.deepcopy(train_dataset)
        #pdb.set_trace()
        cali_dataset.all_data = [train_dataset.all_data[i] for i in demon_data_index]

        # train sharpness_aware
        s_t = time.time()
        if args.config['learning_type'] == 'sharpness_aware':
            model_wrapper.layernorm_adaptation_sharpness_aware(args.config, cali_dataset, save_dir=args.save_dir, run_name=run_name)
        elif args.config['learning_type'] == 'sharpness_aware_approx':
            model_wrapper.layernorm_adaptation_sharpness_aware_approx(args.config, cali_dataset, save_dir=args.save_dir,
                                                                   run_name=run_name)
        elif args.config['learning_type'] == 'layernorm_DyT':
            model_wrapper.layernorm_adaptation_additional_learnDyT(args.config, cali_dataset, save_dir=args.save_dir,
                                                                      run_name=run_name)
        elif args.config['learning_type'] == 'sharpness_encoding':
            model_wrapper.layernorm_adaptation_sharpness_encoding_nocache(args.config, cali_dataset, save_dir=args.save_dir,
                                                                  run_name=run_name)
        elif args.config['learning_type'] == 'version4':
            model_wrapper.layernorm_adaptation_verion4(args.config, cali_dataset, save_dir=args.save_dir, run_name=run_name)
        elif args.config['learning_type'] == 'rola_analysis':
            model_wrapper.layernorm_adaptation_verion4_basedrola(args.config, cali_dataset, save_dir=args.save_dir,
                                                       run_name=run_name)
        elif args.config['learning_type'] == 'ia3':
            model_wrapper.layernorm_adaptation_verion4_IA3(args.config, cali_dataset, save_dir=args.save_dir,
                                                                 run_name=run_name)
        else:
            model_wrapper.layernorm_adaptation(args.config, cali_dataset, save_dir=args.save_dir, run_name=run_name)
            
        e_t = time.time()
        print(f'Calibration time: {e_t - s_t}')
        result_dict['time']['calibrate'].append(e_t - s_t)

        # evaluate sharpness_aware
        s_t = time.time()
        #test_ours_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='', use_cache=args.config['use_cache'])
        
        if args.config['learning_property_analysis']:
            model_wrapper.layernorm_adaptation_verion4_analysis(args.config, cali_dataset, test_dataset, baseline_demon, save_dir=args.save_dir, run_name=run_name)
            
        if args.config['probe_analysis']:
            model_wrapper.layernorm_adaptation_verion4_probe(args.config, cali_dataset, test_dataset, baseline_demon, save_dir=args.save_dir, run_name=run_name)

        if args.config['eval_type'] == 'eval_with_demonstration':
            test_ours_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration=baseline_demon, use_cache=args.config['use_cache'])
        elif args.config['eval_type'] == 'eval_with_empty':
            test_ours_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='', use_cache=args.config['use_cache'])
        
        

        print(f'Test Soft Prompt result: {test_ours_result}\n')
        result_dict['test_result']['ours'].append(test_ours_result)
        e_t = time.time()
        print(f'Evaluate time: {e_t - s_t}')
        result_dict['time']['evaluate'].append(e_t - s_t)

        # save result_dict after each run
        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(result_dict, f, indent=4)

        # reset model_wrapper to unadapted model
        del model_wrapper, model, tokenizer, model_config
        model, tokenizer, model_config = \
        utils.load_model_tokenizer(args.model_name, args.device, 
                                load_in_8bit=args.config['load_in_8bit'],
                                output_hidden_states=False)
        # get model_wrapper
        model_wrapper = utils.get_model_wrapper(args.model_name, model, 
                                                tokenizer, model_config, 
                                                args.device)
        
    # delete all variables

    if config['run_baseline']:
        utils.result_mean_calculator(result_dict, 'few_shot')
    utils.result_mean_calculator(result_dict, 'ours')
    # save result_dict after each run
    with open(args.save_dir + '/result_dict.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

    del model_wrapper, model, tokenizer, train_dataset, test_dataset, cali_dataset
    del test_evaluator, result_dict, demon_list

# get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_ia3.py', help='path to config file')
    return parser.parse_args()


if __name__ == "__main__":
    # get args
    hf_token = "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"
    # Log in using the token
    login(token=hf_token)
    start_time = time.time()
    args = get_args()
    # load config
    config = utils.load_config(args.config_path)
    # Generate all combinations of models and datasets
    combinations = list(itertools.product(config['models'], config['datasets']))
    # Queue to hold tasks
    print('all the experiments')
    print(combinations)

    for model_name, dataset_name in combinations:
        run_task('0', config, model_name, dataset_name)  # without parallelisation
        
    print(f' total running time: {(time.time() - start_time)/3600: .4f} hours ')



# Create a process for each GPU
'''
processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
# Start all processes
for p in processes:
    p.start()
# Wait for all processes to finish
for p in processes:
    p.join()
print("All tasks completed.")
'''
