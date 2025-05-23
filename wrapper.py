import math
import string
import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from functools import reduce
import numpy as np
import utils
import global_vars as gv
from peft import get_peft_model, PromptTuningConfig, LNTuningConfig, IA3Config, TaskType, LoraConfig
import pdb
from self_exploration_tool import *
import flat_learning
import inspect
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss


class ModelWrapper(nn.Module):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device
        self.num_layers = self._get_layer_num()
        self.latent_dict = {}
        self.linear_coef = None
        self.inject_layers = None
        print(f"The model has {self.num_layers} layers:")

    def reset_latent_dict(self):
        self.latent_dict = {}
            
    @contextmanager
    def extract_latent(self):
        handles = []
        try:
            # attach hook
            for layer_idx in range(self.num_layers):
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'attn')).register_forward_hook(
                    self.extract_hook_func(layer_idx, 'attn')))
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'mlp')).register_forward_hook(
                    self.extract_hook_func(layer_idx, 'mlp')))
                handles.append(
                    self._get_nested_attr(self._get_arribute_path(layer_idx, 'hidden')).register_forward_hook(
                    self.extract_hook_func(layer_idx, 'hidden')))
            yield
        finally:
            # remove hook
            for handle in handles:
                handle.remove()

    def extract_hook_func(self, layer_idx, target_module):
        if layer_idx not in self.latent_dict:
            self.latent_dict[layer_idx] = {}
        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                outputs = outputs[0]
            self.latent_dict[layer_idx][target_module] = outputs.detach().cpu()
        return hook_func
    
    @contextmanager
    def inject_latent(self, context_vector_dict, config, linear_coef, train_mode=False):
        handles = []
        assert self.inject_layers is not None, "inject_layers is not set!"
        inject_method = config['inject_method']
        inject_pos = config['inject_pos']
        add_noise = config['add_noise']
        noise_scale = config['noise_scale']
        try:
            # attach hook
            for layer_idx, layer in enumerate(self.inject_layers):
                for module_idx, module in enumerate(config['module']):
                    context_vector_container = [context_vector_dict[layer][module].to(self.device)]
                    strength = linear_coef[layer_idx, module_idx, :]
                    inject_func = self.inject_hook_func(context_vector_container, strength,
                                                        inject_method, add_noise, noise_scale, 
                                                        inject_pos, train_mode)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func)
                        )
            yield
        finally:
            # remove hook
            print(f"Removing {len(handles)} hooks...")
            for handle in handles:
                handle.remove()

    def inject_hook_func(self, context_vector_container, strength, inject_method,
                         add_noise, noise_scale, inject_pos, train_mode=False):

        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                output = outputs[0]     
            else:
                output = outputs
            # init context_vector
            context_vector = context_vector_container[0]
            # expand inject_value to match output size (b, seq_len, d)
            context_vector = context_vector.expand(output.size(0), output.size(1), context_vector.size(-1))
            
            if inject_method == 'add':
                output = output + F.relu(strength) * context_vector
            elif inject_method == 'linear':
                if inject_pos == 'all':
                    output = strength[1] * output + strength[0] * context_vector
                else:
                    if inject_pos == 'last':
                        for i in range(output.size(0)):
                            end_idx = gv.ATTN_MASK_END[i] - 1
                            content = strength[1] * output[i, end_idx, :].clone().detach() + strength[0] * context_vector[i, end_idx, :]
                            output[i, end_idx, :] = content
                    elif inject_pos == 'first':
                        content = strength[1] * output[:, 0, :].clone().detach() + strength[0] * context_vector[:, 0, :]
                        output[:, 0, :] = content
                    elif inject_pos == 'random':
                        for i in range(output.size(0)):
                            end_idx = gv.ATTN_MASK_END[i]
                            random_idx = random.randint(0, end_idx)
                            content = strength[1] * output[i, random_idx, :].clone().detach() + strength[0] * context_vector[i, random_idx, :]
                            output[i, random_idx, :] = content
                    else:
                        raise ValueError("only support all, last, first or random!")
                    
            elif inject_method == 'balance':
                a, b = strength[0], strength[1]
                output = ((1.0 - a) * output + a * context_vector) * b
            else:
                raise ValueError("only support add, linear or balance!")

            if add_noise and train_mode:
                # get l2_norm of output and use it as a scalar to scale noise, make sure no gradient
                output_norm = torch.norm(output, p=2, dim=-1).detach().unsqueeze(-1)
                noise = torch.randn_like(output).detach()
                output += noise * output_norm * noise_scale
            
            if type(outputs) is tuple:
                outputs = list(outputs)
                outputs[0] = output
                outputs = tuple(outputs)
            else:
                outputs = output
            return outputs
        return hook_func
    

    @contextmanager
    def replace_latent(self, context_vector_dict, target_layers, config):
        handles = []
        try:
            # attach hook
            for _, layer in enumerate(target_layers):
                for _, module in enumerate(config['module']):
                    context_vector_container = [context_vector_dict[layer][module].to(self.device)]
                    inject_func = self.replace_hook_func(context_vector_container)
                    handles.append(
                        self._get_nested_attr(self._get_arribute_path(layer, module)).
                        register_forward_hook(inject_func))
            yield
        finally:
            # remove hook
            print(f"Removing {len(handles)} hooks...")
            for handle in handles:
                handle.remove()

    def replace_hook_func(self, context_vector_container):
        def hook_func(module, inputs, outputs):
            if type(outputs) is tuple:
                output = outputs[0]     
            else:
                output = outputs
            # init context_vector
            context_vector = context_vector_container[0]
            # replace hidden states of last token position with context_vector
            for i in range(output.size(0)):
                end_idx = gv.ATTN_MASK_END[i]
                output[i, end_idx, :] = context_vector
            
            if type(outputs) is tuple:
                outputs = list(outputs)
                outputs[0] = output
                outputs = tuple(outputs)
            else:
                outputs = output
            return outputs
        return hook_func
    

    def get_context_vector(self, all_latent_dicts, config):
        if len(all_latent_dicts) == 1:
            latent_dict = all_latent_dicts[0]
            output_dict = {}
            for layer, sub_dict in latent_dict.items():
                output_dict[layer] = {}
                for module in config['module']:
                    latent_value = sub_dict[module]
                    if config['tok_pos'] == 'last':
                        latent_value = latent_value[:, -1, :].squeeze()
                    elif config['tok_pos'] == 'first':
                        latent_value = latent_value[:, 0, :].squeeze()
                    elif config['tok_pos'] == 'random':
                        latent_value = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                    else:
                        raise ValueError("only support last, first or random!")
                    output_dict[layer][module] = latent_value.detach().to('cpu')
        else:
            # concatenate context vector for each module
            ensemble_dict = {module:[] for module in config['module']} # {module_name: []}
            for _, latent_dict in enumerate(all_latent_dicts):
                cur_dict = {module:[] for module in config['module']}  # {module_name: []}
                for layer, sub_dict in latent_dict.items():
                    for module in config['module']:
                        latent_value = sub_dict[module]  # (b, seq_len, d)  
                        if config['tok_pos'] == 'last':
                            latent_value = latent_value[:, -1, :].squeeze()
                        elif config['tok_pos'] == 'first':
                            latent_value = latent_value[:, 0, :].squeeze()
                        elif config['tok_pos'] == 'random':
                            latent_value = latent_value[:, random.randint(0, latent_value.size(1)), :].squeeze()
                        else:
                            raise ValueError("only support last, first or random!")
                        cur_dict[module].append(latent_value)

                for module, latent_list in cur_dict.items():
                    cur_latent = torch.stack(latent_list, dim=0) # (layer_num, d)
                    ensemble_dict[module].append(cur_latent)

            for module, latent_list in ensemble_dict.items():
                if config['post_fuse_method'] == 'mean':
                    context_vector = torch.stack(latent_list, dim=0).mean(dim=0)  # (layer_num, d)
                    ensemble_dict[module] = context_vector 
                elif config['post_fuse_method'] == 'pca':
                    latents = torch.stack(latent_list, dim=0)  # (ensemble_num, layer_num, d)
                    ensemble_num, layer_num, d = latents.size()
                    latents = latents.view(ensemble_num, -1)  # (ensemble_num*layer_num, d)
                    # apply pca
                    pca = utils.PCA(n_components=1).to(latents.device).fit(latents.float())
                    context_vector = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0)
                    ensemble_dict[module] = context_vector.view(layer_num, d)  # (layer_num, d)
                else:
                    raise ValueError("Unsupported ensemble method!")
            # reorganize ensemble_dict into layers
            layers = list(all_latent_dicts[0].keys())
            output_dict = {layer: {} for layer in layers} 
            for module, context_vector in ensemble_dict.items():
                for layer_idx, layer in enumerate(layers):
                    output_dict[layer][module] = context_vector[layer_idx, :].detach().to('cpu')  # (d)

        return output_dict
    

    def calibrate_strength(self, context_vector_dict, dataset, config, 
                           save_dir=None, run_name=None):
        # prepare label dict          
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")

        # frozen all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # init optimizer
        optim_paramters = [{'params': self.linear_coef}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'], 
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'], 
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')

        # get all_data
        all_data = dataset.all_data
        
        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05*epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
                    if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # train
        print('Calibrating strength params...')
        with self.inject_latent(context_vector_dict, config,
                                self.linear_coef, train_mode=True):
            loss_list = []
            all_data_index = list(range(len(all_data)))
            epoch_iter = len(all_data) // batch_size
            for _ in range(epochs):
                epoch_loss = []
                for i in range(epoch_iter):
                    np.random.shuffle(all_data_index)
                    batch_index = all_data_index[:batch_size]
                    batch_data = [all_data[idx] for idx in batch_index]
                    batch_input, batch_label = [], []
                    for data in batch_data:
                        input_str, ans_list, label = dataset.apply_template(data)

                        # collect single demonstration example
                        if config['cali_example_method'] == 'normal':
                            pass
                        elif config['cali_example_method'] == 'random_label':
                            label = random.choice(list(range(len(ans_list))))
                        else:
                            raise ValueError("only support normal or random_label!")
                        
                        batch_input.append(input_str)
                        batch_label.append(label)

                    input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                    input_ids = input_tok['input_ids'].to(self.device)
                    attn_mask = input_tok['attention_mask'].to(self.device)
                    pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                    # set global vars
                    gv.ATTN_MASK_END = pred_loc
                    gv.ATTN_MASK_START = torch.zeros_like(pred_loc)
                    # forward
                    logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                    # get prediction logits
                    pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                    # get loss
                    gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                    epoch_loss.append(loss.item())
                    # update strength params
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    cur_lr = optimizer.param_groups[0]['lr']
                    print(f'Epoch {_+1}/{epochs}, batch {i//batch_size+1}/{len(all_data)//batch_size+1}, loss: {loss.item()}, lr: {cur_lr}')
                epoch_loss = np.mean(epoch_loss)
                loss_list.append(epoch_loss)

        # fronzen all learnable strength params
        self.linear_coef.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')


    def layernorm_adaptation_sharpness_aware(self, config, dataset, save_dir=None, run_name=None):
        pt_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.model, pt_config)

        tuning_param_list = []
        tuning_name_list = []

        '''
        name_holder = [ name for name, pamra in peft_model.named_parameters()]
        print(name_holder)
        print('runing layernorm implementation')
        '''
        if config['post_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'post_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)

            for param in peft_model.parameters():
                param.requires_grad = False

            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        elif config['input_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'input_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)

            for param in peft_model.parameters():
                param.requires_grad = False

            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True

        else:
            for name, param in peft_model.named_parameters():
                if param.requires_grad:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)


        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")

        # print trainable parameters
        peft_model.print_trainable_parameters()
        print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model

        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')

        # get all data
        all_data = dataset.all_data

        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # train
        loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                for data in batch_data:
                    input_str, _, label = dataset.apply_template(data)
                    batch_input.append(input_str)
                    batch_label.append(label)

                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                # forward
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())

                # update strength params
                optimizer.zero_grad()
                loss.backward()
                old_state = {}
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            old_state[name]= param.data.clone()
                            scale = config['rho']/(param.grad.norm() + 1e-12)
                            print(name)
                            print(param.grad.norm().item())
                            e_w = torch.pow(param, 2) * param.grad * scale.to(param)
                            param.add_(e_w)

                # second round
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())

                # update strength params
                optimizer.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            param.data = old_state[name]
                optimizer.step()
                scheduler.step()

            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)


        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')

    def layernorm_adaptation_sharpness_aware_approx(self, config, dataset, save_dir=None, run_name=None):
        pt_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.model, pt_config)

        tuning_param_list = []
        tuning_name_list = []

        if config['post_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'post_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)

            for param in peft_model.parameters():
                param.requires_grad = False

            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        elif config['input_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'input_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)

            for param in peft_model.parameters():
                param.requires_grad = False

            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True

        else:
            for name, param in peft_model.named_parameters():
                if param.requires_grad:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)

        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")

        # print trainable parameters
        peft_model.print_trainable_parameters()
        print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model

        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')

        # get all data
        all_data = dataset.all_data

        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # train
        loss_list = []
        conv_loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            epoch_conv_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                for data in batch_data:
                    input_str, _, label = dataset.apply_template(data)
                    batch_input.append(input_str)
                    batch_label.append(label)

                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ######################
                # forward
                ######################
                print('working on the convergence bound and sharp approxy')
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                #loss = torch.tensor(0.0)
                #epoch_loss.append(loss.item())
                print('entropy flat')
                loss -= 0.01 * utils.entropy_from_logits(pred_logits).mean()
                
                conver_loss = 0.0
                weight_scale = [hold for hold in range(1, len(hidden_states))]
                weight_scale = torch.softmax(torch.from_numpy(np.asarray(weight_scale)/config['conver_loss_regular_temp']), dim =0)
                
                
                if config['conver_loss']:
                    for  i in range(1, len(hidden_states)-1):
                        conver_loss += torch.nn.functional.mse_loss(hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                                                                    ,hidden_states[i+1][torch.arange(logits.size(0)), pred_loc] )
                        
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                elif config['conver_loss_regular']:
                    for  i in range(2, len(hidden_states)-1):
                        numerator = torch.nn.functional.mse_loss(hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                                                                    ,hidden_states[i+1][torch.arange(logits.size(0)), pred_loc] )
                        
                        demoninator =  torch.nn.functional.mse_loss(hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                                                                    ,hidden_states[i-1][torch.arange(logits.size(0)), pred_loc] )
                        
                        if config['conver_loss_regular_expo']:
                            conver_loss += weight_scale[i].item() * numerator/demoninator
                        else:
                            #conver_loss += torch.log(numerator/demoninator)
                            conver_loss += numerator/demoninator
                
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                
                if config['pushing_loss']:
                    print('pushing_loss')
                    pushing_loss = 0.0
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i+1][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        pushing_loss += numerator / demoninator
                    
                    loss += config['pushing_loss_lambda'] * pushing_loss
                    
                
                '''
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())
                '''

                # update strength params
                optimizer.zero_grad()
                '''
                loss.backward(retain_graph= True, create_graph= True)
                gradient_holder= {}
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            gradient_holder[name] = param.grad.norm(2)
                loss.backward()
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            print((gradient_holder[name] - param.grad.norm(2)) **2 )
                pdb.set_trace()
                '''
                loss.backward()

                optimizer.step()
                scheduler.step()

            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
            if config['conver_loss'] or config['conver_loss_regular']:
                epoch_conv_loss = np.mean(epoch_conv_loss)
                conv_loss_list.append(epoch_conv_loss)


        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        if config['conver_loss'] or config['conver_loss_regular']:
            utils.plot_loss_curve(conv_loss_list, save_dir + f'/{run_name}_conv_loss_curve.png')
    
    def layernorm_adaptation_sharpness_encoding_backup(self, config, dataset, save_dir=None, run_name=None):
        pt_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.model, pt_config)
        print('in sharpness encoding')
        
        '''
        for layer in peft_model.model.model.layers:
            #hook = layer.register_forward_pre_hook(noise_injector.hook_fn)
            #hook = layer.register_forward_hook(noise_injector.hook_fn)
            hook = layer.register_forward_pre_hook(flat_learning.hook_fn_test)
            hooks.append(hook)
        '''
        
        tuning_param_list = []
        tuning_name_list = []
        
        if config['post_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'post_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        elif config['input_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'input_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        
        else:
            for name, param in peft_model.named_parameters():
                if param.requires_grad:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
        
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[
                0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        # peft_model.print_trainable_parameters()
        # print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model
        
        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data
        
        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # train
        loss_list = []
        conv_loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            epoch_conv_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                # construct the demonstration here.
                
                for data in batch_data:
                    input_str, _, label = dataset.apply_template(data)
                    batch_input.append(input_str)
                    batch_label.append(label)
                
                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ######################
                # forward
                ######################
                print('working on the convergence bound and sharp approxy')
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                
                conver_loss = 0.0
                weight_scale = [hold for hold in range(1, len(hidden_states))]
                weight_scale = torch.softmax(
                    torch.from_numpy(np.asarray(weight_scale) / config['conver_loss_regular_temp']), dim=0)
                if config['conver_loss']:
                    for i in range(1, len(hidden_states) - 1):
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                elif config['conver_loss_regular']:
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i - 1][torch.arange(logits.size(0)), pred_loc])
                        
                        if config['conver_loss_regular_expo']:
                            conver_loss += weight_scale[i].item() * numerator / demoninator
                        else:
                            # conver_loss += torch.log(numerator/demoninator)
                            conver_loss += numerator / demoninator
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                
                if config['pushing_loss']:
                    print('pushing_loss')
                    pushing_loss = 0.0
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        pushing_loss += numerator / demoninator
                    
                    loss += config['pushing_loss_lambda'] * pushing_loss
                
                # flatness approximation
                noise_scale = config['noise_scale_hess']
                noise_holder = []
                post_layer_norm_holder = []
                hooks = []
                
                def hook_fn_local(module, input):
                    """Function to add noise and store it."""
                    noise = torch.randn_like(input[0]) * noise_scale
                    post_layer_norm_holder.append(module.post_attention_layernorm.weight)
                    input = (input[0] + noise * module.post_attention_layernorm.weight,)
                    noise_holder.append(noise)
                    return input
                
                for layer in self.model.model.model.layers:
                    # hook = layer.register_forward_pre_hook(noise_injector.hook_fn)
                    # hook = layer.register_forward_hook(noise_injector.hook_fn)
                    hook = layer.register_forward_pre_hook(hook_fn_local)
                    hooks.append(hook)
                noise_output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                # noise_logits = output_noise.logits
                noise_hidden_states = noise_output.hidden_states
                
                '''
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                '''
                
                flat_loss = 0.0
                for i in range(1, len(hidden_states) - 1):
                    '''
                    conver_loss += torch.nn.functional.mse_loss(
                        hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                        , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    '''
                    grad_noise = noise_hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - \
                                 noise_hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                    grad = hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - hidden_states[i][
                        torch.arange(logits.size(0)), pred_loc]
                    # flat_loss += post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale
                    # flat_loss += torch.nn.functional.softplus(post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale)
                    flat_loss += torch.nn.functional.softplus(noise_holder[i] @ (grad_noise - grad).t() / noise_scale)
                
                loss += config['flat_loss_lambda'] * flat_loss.mean()
                
                '''
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())
                '''
                
                # update strength params
                optimizer.zero_grad()
                '''
                loss.backward(retain_graph= True, create_graph= True)
                gradient_holder= {}
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            gradient_holder[name] = param.grad.norm(2)
                loss.backward()
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            print((gradient_holder[name] - param.grad.norm(2)) **2 )
                pdb.set_trace()
                '''
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                for ele in hooks:
                    ele.remove()
            
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
            if config['conver_loss'] or config['conver_loss_regular']:
                epoch_conv_loss = np.mean(epoch_conv_loss)
                conv_loss_list.append(epoch_conv_loss)
        
        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        if config['conver_loss'] or config['conver_loss_regular']:
            utils.plot_loss_curve(conv_loss_list, save_dir + f'/{run_name}_conv_loss_curve.png')
    
    def layernorm_adaptation_verion4_IA3(self, config, dataset, save_dir=None, run_name=None):
        '''
        # learning for fast convergence
        # add demonstration in the inner loop to trigger the optimization in the inner loop
        # Flatness learning
        # cross entropy tuning
        '''
        print('Version 4 based IA3')
        
        '''
        pt_config = LoraConfig(r=config['lora_rank'], lora_alpha=16, target_modules=config['lora_target_modules'],
                               lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        '''
        if 'gpt' in config['models'][0]:
            #pt_config = IA3Config( task_type=TaskType.SEQ_CLS, target_modules=['c_attn'] )
            pt_config = IA3Config( task_type=TaskType.SEQ_CLS, target_modules=config['target_modules'] )
        else:
            pt_config = IA3Config(
                task_type=TaskType.SEQ_CLS, target_modules=config['target_modules'],
                feedforward_modules=["down_proj"]
            )
        # pt_config = LoraConfig(r =config['lora_rank'], lora_alpha = 16, target_modules= ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=  0.05, bias = "none" , task_type="CAUSAL_LM" )
        
        peft_model = get_peft_model(self.model, pt_config)
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[
                0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        peft_model.print_trainable_parameters()
        # print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model
        
        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data
        example_separator = dataset.example_separator
        
        # set the batch_size for training
        epochs, batch_size = config['epochs'], config['grad_bs']
        batch_size += config['demon_bs']
        sub_batch_size = config['grad_bs']
        
        # init lr_scheduler
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # train
        loss_list = []
        conv_loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            epoch_conv_loss = []
            np.random.shuffle(all_data_index)
            for batch_i in range(0, len(all_data) - (len(all_data) % batch_size), batch_size):
                batch_index = all_data_index[batch_i: batch_i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                # construct the demonstration here.
                # print(f'epoch{_}/{epochs}: iter: {i}/{len(all_data)/batch_size}')
                instruct = ""
                demonstration = ""
                if config['demon_bs'] == 0:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        batch_input.append(input_str)
                        batch_label.append(label)
                
                else:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        if sub_index < batch_size - sub_batch_size:
                            ans = ans_str[label]
                            new_example = input_str + ' ' + ans
                            demonstration = demonstration + new_example + example_separator
                        else:
                            batch_input.append(demonstration + input_str)
                            batch_label.append(label)
                
                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ######################
                # forward
                ######################
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                
                print(f'loss: {loss.item()}')
                epoch_loss.append(loss.item())
               
                # update strength params
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
            
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
        
        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        
        if not os.path.exists(f'{save_dir}/loss_record/'):
            os.makedirs(f'{save_dir}/loss_record/')
        
    
    def layernorm_adaptation_verion4_basedrola(self, config, dataset, save_dir=None, run_name=None):
        '''
        # learning for fast convergence
        # add demonstration in the inner loop to trigger the optimization in the inner loop
        # Flatness learning
        # cross entropy tuning
        '''
        print('Version 4 based LoRA')
        
        pt_config = LoraConfig(r =config['lora_rank'], lora_alpha = 16, target_modules= config['lora_target_modules'], lora_dropout=  0.05, bias = "none" , task_type="CAUSAL_LM" )
        #pt_config = LoraConfig(r =config['lora_rank'], lora_alpha = 16, target_modules= ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=  0.05, bias = "none" , task_type="CAUSAL_LM" )

        peft_model = get_peft_model(self.model, pt_config)
        
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[
                0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        peft_model.print_trainable_parameters()
        print(f"LORA Rank: {config['lora_rank']}")
        #print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model
        
        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data
        example_separator = dataset.example_separator
        
        # set the batch_size for training
        epochs, batch_size = config['epochs'], config['grad_bs']
        batch_size += config['demon_bs']
        sub_batch_size = config['grad_bs']
        
        # init lr_scheduler
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # train
        loss_list = []
        conv_loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            epoch_conv_loss = []
            np.random.shuffle(all_data_index)
            for batch_i in range(0, len(all_data) - (len(all_data) % batch_size), batch_size):
                batch_index = all_data_index[batch_i: batch_i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                # construct the demonstration here.
                # print(f'epoch{_}/{epochs}: iter: {i}/{len(all_data)/batch_size}')
                instruct = ""
                demonstration = ""
                if config['demon_bs'] == 0:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        batch_input.append(input_str)
                        batch_label.append(label)
                
                else:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        if sub_index < batch_size - sub_batch_size:
                            ans = ans_str[label]
                            new_example = input_str + ' ' + ans
                            demonstration = demonstration + new_example + example_separator
                        else:
                            batch_input.append(demonstration + input_str)
                            batch_label.append(label)
                
                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ######################
                # forward
                ######################
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                
                epoch_loss.append(loss.item())
                
                conver_loss = 0.0
                weight_scale = [hold for hold in range(1, len(hidden_states))]
                weight_scale = torch.softmax(
                    torch.from_numpy(np.asarray(weight_scale) / config['conver_loss_regular_temp']), dim=0)
                if config['conver_loss']:
                    print('conver loss')
                    for i in range(1, len(hidden_states) - 1):
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                elif config['conver_loss_regular']:
                    print('conver loss with regualrizer')
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i - 1][torch.arange(logits.size(0)), pred_loc])
                        
                        if config['conver_loss_regular_expo']:
                            conver_loss += weight_scale[i].item() * numerator / demoninator
                        else:
                            # conver_loss += torch.log(numerator/demoninator)
                            conver_loss += numerator / demoninator
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                
                if config['pushing_loss']:
                    print('adding pushing_loss')
                    pushing_loss = 0.0
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        pushing_loss += numerator / demoninator
                    
                    loss += config['pushing_loss_lambda'] * pushing_loss
                
                ####################################
                # flatness approximation
                ####################################
                if config['flatness_loss']:
                    noise_scale = config['noise_scale_hess']
                    noise_holder = []
                    post_layer_norm_holder = []
                    hooks = []
                    
                    if 'gpt' in config['models'][0]:
                        def hook_fn_local(module, input):
                            """Function to add noise and store it."""
                            noise = torch.randn_like(input[0]) * noise_scale
                            post_layer_norm_holder.append(module.ln_2.base_layer.weight)
                            input = (input[0] + noise * module.ln_2.base_layer.weight,)
                            noise_holder.append(noise)
                            return input
                        
                        for layer in self.model.transformer.h:
                            hook = layer.register_forward_pre_hook(hook_fn_local)
                            hooks.append(hook)
                    else:
                        def hook_fn_local(module, input):
                            """Function to add noise and store it."""
                            noise = torch.randn_like(input[0]) * noise_scale
                            post_layer_norm_holder.append(module.post_attention_layernorm.weight)
                            input = (input[0] + noise * module.post_attention_layernorm.weight,)
                            noise_holder.append(noise)
                            return input
                        
                        for layer in self.model.model.model.layers:
                            hook = layer.register_forward_pre_hook(hook_fn_local)
                            hooks.append(hook)
                    
                    noise_output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                    noise_hidden_states = noise_output.hidden_states
                    
                    flat_loss = 0.0
                    for i in range(1, len(hidden_states) - 1):
                        '''
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        '''
                        grad_noise = noise_hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - \
                                     noise_hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                        grad = hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - hidden_states[i][
                            torch.arange(logits.size(0)), pred_loc]
                        # flat_loss += post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale
                        # flat_loss += torch.nn.functional.softplus(post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale)
                        
                        # worked version
                        # flat_loss += torch.nn.functional.softplus( -1 * noise_holder[i] @ (grad_noise - grad).t() / noise_scale)
                        
                        # precised version
                        flat_loss += torch.nn.functional.softplus(
                            -1 * noise_holder[i][torch.arange(logits.size(0)), pred_loc] @ (
                                    grad_noise - grad).t() / noise_scale)
                    
                    loss += config['flat_loss_lambda'] * flat_loss.mean()
                
                # update strength params
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                if config['flatness_loss']:
                    for ele in hooks:
                        ele.remove()
            
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
            if config['conver_loss'] or config['conver_loss_regular']:
                epoch_conv_loss = np.mean(epoch_conv_loss)
                conv_loss_list.append(epoch_conv_loss)
        
        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
       
        if not os.path.exists(f'{save_dir}/loss_record/'):
            os.makedirs(f'{save_dir}/loss_record/')
        
        np.save(f'{save_dir}/loss_record/{run_name}_ce_loss.npy', loss_list)
        
        if config['conver_loss'] or config['conver_loss_regular']:
            utils.plot_loss_curve(conv_loss_list, save_dir + f'/{run_name}_conv_loss_curve.png')
            np.save(f'{save_dir}/loss_record/{run_name}_conv_loss.npy', conv_loss_list)

    def layernorm_adaptation_verion4(self, config, dataset, save_dir=None, run_name=None):
        '''
        # learning for fast convergence
        # add demonstration in the inner loop to trigger the optimization in the inner loop
        # Flatness learning
        # cross entropy tuning
        '''
        print('Version 4')
        pt_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.model, pt_config)
        
        if config['skip_training']:
            for param in peft_model.parameters():
                param.requires_grad = False
            self.model = peft_model
            return

        tuning_param_list = []
        tuning_name_list = []
        
        if config['layernorm_type'] == 'post_attention':
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'post_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        
        elif config['layernorm_type'] == 'input_attention':
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'input_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        else:
            for name, param in peft_model.named_parameters():
                if param.requires_grad:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
        
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[
                0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        peft_model.print_trainable_parameters()
        #print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model
        
        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data
        example_separator = dataset.example_separator
        
        # set the batch_size for training
        epochs, batch_size = config['epochs'], config['grad_bs']
        batch_size += config['demon_bs']
        sub_batch_size = config['grad_bs']
        
        # init lr_scheduler
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # train
        loss_list = []
        conv_loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            epoch_conv_loss = []
            np.random.shuffle(all_data_index)
            for batch_i in range(0, len(all_data) - (len(all_data) % batch_size), batch_size):
                batch_index = all_data_index[batch_i: batch_i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                # construct the demonstration here.
                # print(f'epoch{_}/{epochs}: iter: {i}/{len(all_data)/batch_size}')
                instruct = ""
                demonstration = ""
                if config['demon_bs'] == 0:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        batch_input.append(input_str)
                        batch_label.append(label)
                
                else:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        if sub_index < batch_size - sub_batch_size:
                            ans = ans_str[label]
                            new_example = input_str + ' ' + ans
                            demonstration = demonstration + new_example + example_separator
                        else:
                            batch_input.append(demonstration + input_str)
                            batch_label.append(label)
                
                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ######################
                # forward
                ######################
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                
                epoch_loss.append(loss.item())
                
                conver_loss = 0.0
                weight_scale = [hold for hold in range(1, len(hidden_states))]
                weight_scale = torch.softmax(
                    torch.from_numpy(np.asarray(weight_scale) / config['conver_loss_regular_temp']), dim=0)
                if config['conver_loss']:
                    print('conver loss')
                    for i in range(1, len(hidden_states) - 1):
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                elif config['conver_loss_regular']:
                    print('conver loss with regualrizer')
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i - 1][torch.arange(logits.size(0)), pred_loc])
                        
                        if config['conver_loss_regular_expo']:
                            conver_loss += weight_scale[i].item() * numerator / demoninator
                        else:
                            # conver_loss += torch.log(numerator/demoninator)
                            conver_loss += numerator / demoninator
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                
                if config['pushing_loss']:
                    print('adding pushing_loss')
                    pushing_loss = 0.0
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        pushing_loss += numerator / demoninator
                    
                    loss += config['pushing_loss_lambda'] * pushing_loss
                
                ####################################
                # flatness approximation
                ####################################
                if config['flatness_loss']:
                    noise_scale = config['noise_scale_hess']
                    noise_holder = []
                    post_layer_norm_holder = []
                    hooks = []
                    
                    if 'gpt' in config['models'][0]:
                        def hook_fn_local(module, input):
                            """Function to add noise and store it."""
                            noise = torch.randn_like(input[0]) * noise_scale
                            post_layer_norm_holder.append(module.ln_2.base_layer.weight)
                            input = (input[0] + noise * module.ln_2.base_layer.weight,)
                            noise_holder.append(noise)
                            return input
                        for layer  in self.model.transformer.h:
                            hook = layer.register_forward_pre_hook(hook_fn_local)
                            hooks.append(hook)
                    else:
                        def hook_fn_local(module, input):
                            """Function to add noise and store it."""
                            noise = torch.randn_like(input[0]) * noise_scale
                            post_layer_norm_holder.append(module.post_attention_layernorm.weight)
                            input = (input[0] + noise * module.post_attention_layernorm.weight,)
                            noise_holder.append(noise)
                            return input
                        
                        for layer in self.model.model.model.layers:
                            hook = layer.register_forward_pre_hook(hook_fn_local)
                            hooks.append(hook)
                            
                    noise_output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                    noise_hidden_states = noise_output.hidden_states
                    
                    flat_loss = 0.0
                    for i in range(1, len(hidden_states) - 1):
                        '''
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        '''
                        grad_noise = noise_hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - \
                                     noise_hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                        grad = hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - hidden_states[i][
                            torch.arange(logits.size(0)), pred_loc]
                        # flat_loss += post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale
                        # flat_loss += torch.nn.functional.softplus(post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale)
                        
                        # worked version
                        #flat_loss += torch.nn.functional.softplus( -1 * noise_holder[i] @ (grad_noise - grad).t() / noise_scale)
                        
                        # precised version
                        flat_loss += torch.nn.functional.softplus( -1 * noise_holder[i][torch.arange(logits.size(0)), pred_loc] @ (grad_noise - grad).t() / noise_scale)

                    loss += config['flat_loss_lambda'] * flat_loss.mean()
                
                # update strength params
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                if config['flatness_loss']:
                    for ele in hooks:
                        ele.remove()
            
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
            if config['conver_loss'] or config['conver_loss_regular']:
                epoch_conv_loss = np.mean(epoch_conv_loss)
                conv_loss_list.append(epoch_conv_loss)
        
        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        
        if not os.path.exists(f'{save_dir}/loss_record/'):
            os.makedirs(f'{save_dir}/loss_record/')
        
        np.save(f'{save_dir}/loss_record/{run_name}_ce_loss.npy', loss_list)
        
        if config['conver_loss'] or config['conver_loss_regular']:
            utils.plot_loss_curve(conv_loss_list, save_dir + f'/{run_name}_conv_loss_curve.png')
            np.save(f'{save_dir}/loss_record/{run_name}_conv_loss.npy', conv_loss_list)

    
    def layernorm_adaptation_verion4_analysis(self, config, dataset, test_dataset, demonstration, save_dir=None, run_name=None):
        '''
        # learning for fast convergence
        # add demonstration in the inner loop to trigger the optimization in the inner loop
        # Flatness learning
        # cross entropy tuning
        '''
        print('Version 4 analysis')
        '''
        pt_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.model, pt_config)
        
        tuning_param_list = []
        tuning_name_list = []
        
        if config['layernorm_type'] == 'post_attention' :
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'post_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
                    
        elif config['layernorm_type'] == 'input_attention':
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'input_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        else:
            for name, param in peft_model.named_parameters():
                if param.requires_grad:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
        '''
        
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        # set model to peft model
        
        # init optimizer
        '''
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        '''
        
        # get all data
        all_data = dataset.all_data
        example_separator = dataset.example_separator
        all_data = test_dataset.all_data
        
        # set the batch_size for training
        epochs, batch_size = config['epochs'], config['grad_bs']
        batch_size += config['demon_bs']
        sub_batch_size = config['grad_bs']
        
        '''
        # init lr_scheduler
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        '''
        
        # train
        loss_list = []
        conv_loss_list = []
        all_data_index = list(range(len(all_data)))
        indiv_condis_holder = []
        indiv_flatness_holder = []
        with torch.no_grad():
            epoch_loss = []
            epoch_conv_loss = []
            np.random.shuffle(all_data_index)
            for batch_i in range(0, len(all_data) - (len(all_data) % batch_size), batch_size):
                batch_index = all_data_index[batch_i: batch_i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                # construct the demonstration here.
                #print(f'epoch{_}/{epochs}: iter: {i}/{len(all_data)/batch_size}')
                instruct = ""
                demonstration = ""
                if config['demon_bs'] == 0:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        batch_input.append(input_str)
                        batch_label.append(label)
                
                else:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        if sub_index < batch_size - sub_batch_size:
                            ans = ans_str[label]
                            new_example = input_str + ' ' + ans
                            demonstration = demonstration + new_example + example_separator
                        else:
                            batch_input.append(demonstration + input_str)
                            batch_label.append(label)
                
                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ######################
                # forward
                ######################
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                
                epoch_loss.append(loss.item())
                
                conver_loss = 0.0
                weight_scale = [hold for hold in range(1, len(hidden_states))]
                weight_scale = torch.softmax( torch.from_numpy(np.asarray(weight_scale) / config['conver_loss_regular_temp']), dim=0)
                
                if config['conver_loss']:
                    print('conver loss')
                    for i in range(1, len(hidden_states) - 1):
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                    
                elif config['conver_loss_regular']:
                    print('conver loss with regualrizer')
                    sub_indiv_condis_holder  = []
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i - 1][torch.arange(logits.size(0)), pred_loc])
                        
                        if config['conver_loss_regular_expo']:
                            conver_loss += weight_scale[i].item() * numerator / demoninator
                        else:
                            # conver_loss += torch.log(numerator/demoninator)
                            conver_loss += numerator / demoninator
                        sub_indiv_condis_holder.append(numerator.item())
                    
                    indiv_condis_holder.append(sub_indiv_condis_holder)
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                
                ####################################
                # flatness approximation
                ####################################
                if config['flatness_loss']:
                    noise_scale = config['noise_scale_hess']
                    noise_holder = []
                    post_layer_norm_holder = []
                    hooks = []
                    
                    if 'gpt' in config['models'][0]:
                        def hook_fn_local(module, input):
                            """Function to add noise and store it."""
                            noise = torch.randn_like(input[0]) * noise_scale
                            post_layer_norm_holder.append(module.ln_2.base_layer.weight)
                            input = (input[0] + noise * module.ln_2.base_layer.weight,)
                            noise_holder.append(noise)
                            return input
                        
                        for layer in self.model.transformer.h:
                            hook = layer.register_forward_pre_hook(hook_fn_local)
                            hooks.append(hook)
                    else:
                        def hook_fn_local(module, input):
                            """Function to add noise and store it."""
                            noise = torch.randn_like(input[0]) * noise_scale
                            post_layer_norm_holder.append(module.post_attention_layernorm.weight)
                            input = (input[0] + noise * module.post_attention_layernorm.weight,)
                            noise_holder.append(noise)
                            return input
                        
                        for layer in self.model.model.model.layers:
                            hook = layer.register_forward_pre_hook(hook_fn_local)
                            hooks.append(hook)
                    
                    noise_output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                    noise_hidden_states = noise_output.hidden_states
                    sub_indiv_flatness_holder = []
                    flat_loss = 0.0
                    for i in range(1, len(hidden_states) - 1):
                        '''
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        '''
                        grad_noise = noise_hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - \
                                     noise_hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                        grad = hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - hidden_states[i][
                            torch.arange(logits.size(0)), pred_loc]
                        # flat_loss += post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale
                        # flat_loss += torch.nn.functional.softplus(post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale)
                        tmp = torch.nn.functional.softplus(-1*noise_holder[i][torch.arange(logits.size(0)), pred_loc] @ (grad_noise - grad).t() / noise_scale)
                        flat_loss += tmp
                        sub_indiv_flatness_holder.append(tmp.item())
                    
                    indiv_flatness_holder.append(sub_indiv_flatness_holder)
                    loss += config['flat_loss_lambda'] * flat_loss.mean()
                    
                # update strength params
                if config['flatness_loss']:
                    for ele in hooks:
                        ele.remove()
            
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
            if config['conver_loss'] or config['conver_loss_regular']:
                epoch_conv_loss = np.mean(epoch_conv_loss)
                conv_loss_list.append(epoch_conv_loss)
        
        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        
        indiv_condis_holder = numpy.asarray(indiv_condis_holder)
        indiv_flatness_holder = numpy.asarray(indiv_flatness_holder)
        
        if not os.path.exists(f'{save_dir}/loss_analysis/'):
            os.makedirs(f'{save_dir}/loss_analysis/')
        
        np.save(f'{save_dir}/loss_analysis/indiv_condis_holder.npy', indiv_condis_holder)
        np.save(f'{save_dir}/loss_analysis/indiv_flatness_holder.npy', indiv_flatness_holder)
        
        '''
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        if config['conver_loss'] or config['conver_loss_regular']:
            utils.plot_loss_curve(conv_loss_list, save_dir + f'/{run_name}_conv_loss_curve.png')
        '''
    
    def layernorm_adaptation_verion4_probe(self, config, dataset, test_dataset, demonstration ,save_dir=None, run_name=None):
        '''
        # learning for fast convergence
        # add demonstration in the inner loop to trigger the optimization in the inner loop
        # Flatness learning
        # cross entropy tuning
        '''
        print('Version 4 probing')
        
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[
                0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        # set model to peft model
        
        # get all data
       
        # all_data is from the training set
        all_data = dataset.all_data
        example_separator = dataset.example_separator
        
        
        # set the batch_size for training
        epochs, batch_size = config['epochs'], config['grad_bs']
        #batch_size += config['demon_bs']
        
        # train
        all_data_index = list(range(len(all_data)))
        indiv_condis_holder = []
        indiv_flatness_holder = []
        demonstration =  demonstration
        with torch.no_grad():
            
            if 'Llama-2' in config['models'][0] or 'Llama-3' in config['models'][0] :
                number_layer = 32
            elif 'gpt2-xl' in config['models'][0]:
                number_layer = 48
            train_x = [[] for tmp_holder in range(number_layer)]
            train_y = [[] for tmp_holder in range(number_layer)]
            
            for batch_i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[batch_i: batch_i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                # construct the demonstration here.
                # print(f'epoch{_}/{epochs}: iter: {i}/{len(all_data)/batch_size}')
                instruct = ""
                if config['demon_bs'] == 0:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        batch_input.append(input_str)
                        batch_label.append(label)
                
                else:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        batch_input.append(demonstration + input_str)
                        batch_label.append(label)
                        
                        '''
                        if sub_index < batch_size - sub_batch_size:
                            ans = ans_str[label]
                            new_example = input_str + ' ' + ans
                            demonstration = demonstration + new_example + example_separator
                        else:
                            batch_input.append(demonstration + input_str)
                            batch_label.append(label)
                        '''
                
                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ####################################
                # generate the training samples
                ####################################
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                '''
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                '''
                
                
                # check the number of hidden states
                for i in range(len(train_x)):
                    train_x[i].append(hidden_states[i+1][torch.arange(logits.size(0)), pred_loc].cpu().numpy())
                    train_y[i].append(label)
            
            
            ####################################
            # train models
            ####################################
            prob_models = []
            for i in range(len(train_x)):
                print(f'training probe model {i}' )
                layer_train_x = np.vstack(train_x[i])
                layer_train_y = np.array(train_y[i])
            
                clf = LogisticRegression(max_iter= 1000)
                clf.fit(layer_train_x, layer_train_y)
                prob_models.append(clf)
                
            ####################################
            # generate the test samples
            # with or without demonstration
            ####################################
            test_x = [[] for tmp_holder in range(len(train_x))]
            test_y = [[] for tmp_holder in range(len(train_x))]
            
            all_data = test_dataset.all_data
            all_data_index = list(range(len(all_data)))
            
            for batch_i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[batch_i: batch_i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                # construct the demonstration here.
                # print(f'epoch{_}/{epochs}: iter: {i}/{len(all_data)/batch_size}')
                instruct = ""
                if config['demon_bs'] == 0:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        batch_input.append(input_str)
                        batch_label.append(label)
                
                else:
                    for sub_index in range(batch_size):
                        input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                        batch_input.append(demonstration + input_str)
                        batch_label.append(label)
                        
                        '''
                        if sub_index < batch_size - sub_batch_size:
                            ans = ans_str[label]
                            new_example = input_str + ' ' + ans
                            demonstration = demonstration + new_example + example_separator
                        else:
                            batch_input.append(demonstration + input_str)
                            batch_label.append(label)
                        '''
                
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ####################################
                # generate the test samples
                ####################################
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                
                '''
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                '''
                
                # check the number of hidden states
                for i in range(len(train_x)):
                    test_x[i].append(hidden_states[i+1][torch.arange(logits.size(0)), pred_loc].cpu().numpy())
                    test_y[i].append(label)
            
            acc_mean = []
            loss_mean = []
            for task_i in range(len(train_x)):
                print(f'Evaluating Model {task_i}')
                layer_test_x = np.vstack(test_x[task_i])
                layer_test_y = np.array(test_y[task_i])
                
                y_pred = prob_models[task_i].predict(layer_test_x)
                y_probs = prob_models[task_i].predict_proba(layer_test_x)
                probe_acc = accuracy_score(layer_test_y,y_pred)
                probe_loss = log_loss(test_y[task_i], y_probs)
                acc_mean.append(probe_acc)
                loss_mean.append(probe_loss)
                
        #pdb.set_trace()
        # plot loss curve and save it
        acc_mean = np.asarray(acc_mean)
        loss_mean = np.asarray(loss_mean)
        if not os.path.exists(f'{save_dir}/probe_analysis/'):
            os.makedirs(f'{save_dir}/probe_analysis/')
        np.save(f'{save_dir}/probe_analysis/{run_name}_probe_acc.npy', acc_mean)
        np.save(f'{save_dir}/probe_analysis/{run_name}_probe_loss.npy', loss_mean)

        '''
        np.save(f'{save_dir}/loss_analysis/indiv_condis_holder.npy', indiv_condis_holder)
        np.save(f'{save_dir}/loss_analysis/indiv_flatness_holder.npy', indiv_flatness_holder)
        '''
        
    
    def layernorm_adaptation_sharpness_encoding_nocache(self, config, dataset, save_dir=None, run_name=None):
        pt_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.model, pt_config)
        print('in sharpness encoding')
        
        '''
        for layer in peft_model.model.model.layers:
            #hook = layer.register_forward_pre_hook(noise_injector.hook_fn)
            #hook = layer.register_forward_hook(noise_injector.hook_fn)
            hook = layer.register_forward_pre_hook(flat_learning.hook_fn_test)
            hooks.append(hook)
        '''
        
        tuning_param_list = []
        tuning_name_list = []
        
        if config['layernorm_type'] == 'post_attention':
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'post_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        elif config['layernorm_type'] == 'input_attention':
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'input_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        
        else:
            for name, param in peft_model.named_parameters():
                if param.requires_grad:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
        
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[
                0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        # peft_model.print_trainable_parameters()
        # print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model
        
        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data
        example_separator = dataset.example_separator

        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        batch_size += config['demon_bs']
        sub_batch_size =  config['grad_bs']

        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # train
        loss_list = []
        conv_loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            epoch_conv_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data) - (len(all_data) % batch_size), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                # construct the demonstration here.
                instruct = ""
                demonstration = ""
                for sub_index in range(batch_size):
                    input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                    if sub_index < batch_size - sub_batch_size:
                        ans = ans_str[label]
                        new_example = input_str + ' ' + ans
                        demonstration = demonstration + new_example + example_separator
                    else:
                        batch_input.append(demonstration + input_str)
                        batch_label.append(label)
                
                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ######################
                # forward
                ######################
                print('working on the convergence bound and sharp approxy no cache update version 2.4')
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                
                conver_loss = 0.0
                weight_scale = [hold for hold in range(1, len(hidden_states))]
                weight_scale = torch.softmax(
                    torch.from_numpy(np.asarray(weight_scale) / config['conver_loss_regular_temp']), dim=0)
                if config['conver_loss']:
                    for i in range(1, len(hidden_states) - 1):
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                elif config['conver_loss_regular']:
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i - 1][torch.arange(logits.size(0)), pred_loc])
                        
                        if config['conver_loss_regular_expo']:
                            conver_loss += weight_scale[i].item() * numerator / demoninator
                        else:
                            # conver_loss += torch.log(numerator/demoninator)
                            conver_loss += numerator / demoninator
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                
                if config['pushing_loss']:
                    print('pushing_loss')
                    pushing_loss = 0.0
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        pushing_loss += numerator / demoninator
                    
                    loss += config['pushing_loss_lambda'] * pushing_loss
                
                # flatness approximation
                noise_scale = config['noise_scale_hess']
                noise_holder = []
                post_layer_norm_holder = []
                hooks = []
                
                def hook_fn_local(module, input):
                    """Function to add noise and store it."""
                    noise = torch.randn_like(input[0]) * noise_scale
                    post_layer_norm_holder.append(module.post_attention_layernorm.weight)
                    input = (input[0] + noise * module.post_attention_layernorm.weight,)
                    noise_holder.append(noise)
                    return input
                
                for layer in self.model.model.model.layers:
                    # hook = layer.register_forward_pre_hook(noise_injector.hook_fn)
                    # hook = layer.register_forward_hook(noise_injector.hook_fn)
                    hook = layer.register_forward_pre_hook(hook_fn_local)
                    hooks.append(hook)
                noise_output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                # noise_logits = output_noise.logits
                noise_hidden_states = noise_output.hidden_states
                
                '''
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                '''
                
                flat_loss = 0.0
                for i in range(1, len(hidden_states) - 1):
                    '''
                    conver_loss += torch.nn.functional.mse_loss(
                        hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                        , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    '''
                    grad_noise = noise_hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - \
                                 noise_hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                    grad = hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc] - hidden_states[i][
                        torch.arange(logits.size(0)), pred_loc]
                    # flat_loss += post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale
                    # flat_loss += torch.nn.functional.softplus(post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale)
                    flat_loss += torch.nn.functional.softplus(noise_holder[i] @ (grad_noise - grad).t() / noise_scale)
                
                loss += config['flat_loss_lambda'] * flat_loss.mean()
                
                '''
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())
                '''
                
                # update strength params
                optimizer.zero_grad()
                '''
                loss.backward(retain_graph= True, create_graph= True)
                gradient_holder= {}
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            gradient_holder[name] = param.grad.norm(2)
                loss.backward()
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            print((gradient_holder[name] - param.grad.norm(2)) **2 )
                pdb.set_trace()
                '''
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                for ele in hooks:
                    ele.remove()
            
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
            if config['conver_loss'] or config['conver_loss_regular']:
                epoch_conv_loss = np.mean(epoch_conv_loss)
                conv_loss_list.append(epoch_conv_loss)
        
        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        if config['conver_loss'] or config['conver_loss_regular']:
            utils.plot_loss_curve(conv_loss_list, save_dir + f'/{run_name}_conv_loss_curve.png')
    
    def layernorm_adaptation_sharpness_encoding(self, config, dataset, save_dir=None, run_name=None):
        pt_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.model, pt_config)
        '''
        peft_model = self.model
        print('in sharpness encoding')
        '''
        
        '''
        for layer in peft_model.model.model.layers:
            #hook = layer.register_forward_pre_hook(noise_injector.hook_fn)
            #hook = layer.register_forward_hook(noise_injector.hook_fn)
            hook = layer.register_forward_pre_hook(flat_learning.hook_fn_test)
            hooks.append(hook)
        '''

        tuning_param_list = []
        tuning_name_list = []
        
        if config['post_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'post_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        elif config['input_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'input_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        else:
            '''
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if 'layernorm' in name:
                    param.requires_grad = True
            '''
            
            for name, param in peft_model.named_parameters():
                if param.requires_grad:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
        
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[
                0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        #peft_model.print_trainable_parameters()
        #print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model
        
        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data
        example_separator = dataset.example_separator
        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        batch_size += config['demon_bs']
        sub_batch_size =  config['grad_bs']
        
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # train
        loss_list = []
        conv_loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            epoch_conv_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data) - (len(all_data) % batch_size), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                # construct the demonstration here.
                instruct = ""
                demonstration = ""
                for sub_index in range(batch_size):
                    input_str, ans_str, label = dataset.apply_template(batch_data[sub_index])
                    if sub_index < batch_size - sub_batch_size:
                        ans = ans_str[label]
                        new_example = input_str + ' ' + ans
                        demonstration = demonstration + new_example + example_separator
                    else:
                        batch_input.append(input_str)
                        batch_label.append(label)
                
                '''
                for data in batch_data:
                    input_str, _, label = dataset.apply_template(data)
                    batch_input.append(input_str)
                    batch_label.append(label)
                '''
                # cache the demonstration
                
                demon_token = self.tokenizer(demonstration, return_tensors="pt", padding=True).to(self.model.device)
                with torch.no_grad():
                    demon_outputs = self.model(**demon_token, use_cache=True)
                demon_past_key_values = demon_outputs.past_key_values
                demon_attn_mask = demon_token['attention_mask']
                # pdb.set_trace()
                demon_past_key_values = tuple(tuple(t.repeat(sub_batch_size, 1, 1, 1) for t in tup) for tup in demon_past_key_values)
                
                demon_attn_mask = demon_attn_mask.repeat(sub_batch_size, 1)
                if len(batch_input) % sub_batch_size != 0:  # last batch
                    sp_demon_past_key_values = tuple(tuple(t.repeat(len(batch_input) % sub_batch_size, 1, 1, 1)
                                                           for t in tup) for tup in demon_outputs.past_key_values)
                    sp_demon_attn_mask = demon_attn_mask[-(len(batch_input) % sub_batch_size):]
                use_cache = True
                 
               ###############################
                if len(batch_input) != sub_batch_size:
                    demon_past_key_values = sp_demon_past_key_values
                    demon_attn_mask = sp_demon_attn_mask
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ######################
                # forward
                ######################
                print('working on the convergence bound and sharp approxy')
                attn_mask = torch.cat([demon_attn_mask, attn_mask], dim=1)
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, past_key_values=demon_past_key_values, use_cache=use_cache, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                    
                conver_loss = 0.0
                weight_scale = [hold for hold in range(1, len(hidden_states))]
                weight_scale = torch.softmax(
                    torch.from_numpy(np.asarray(weight_scale) / config['conver_loss_regular_temp']), dim=0)
                if config['conver_loss']:
                    for i in range(1, len(hidden_states) - 1):
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                elif config['conver_loss_regular']:
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i - 1][torch.arange(logits.size(0)), pred_loc])
                        
                        if config['conver_loss_regular_expo']:
                            conver_loss += weight_scale[i].item() * numerator / demoninator
                        else:
                            # conver_loss += torch.log(numerator/demoninator)
                            conver_loss += numerator / demoninator
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                
                if config['pushing_loss']:
                    print('pushing_loss')
                    pushing_loss = 0.0
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        pushing_loss += numerator / demoninator
                    
                    loss += config['pushing_loss_lambda'] * pushing_loss
                
                # flatness approximation
                noise_scale = config['noise_scale_hess']
                noise_holder = []
                post_layer_norm_holder = []
                hooks = []
                
                def hook_fn_local(module, input):
                    """Function to add noise and store it."""
                    noise = torch.randn_like(input[0]) * noise_scale
                    post_layer_norm_holder.append(module.post_attention_layernorm.weight)
                    input = (input[0] + noise * module.post_attention_layernorm.weight,)
                    noise_holder.append(noise)
                    return input
                
                for layer in self.model.model.model.layers:
                    # hook = layer.register_forward_pre_hook(noise_injector.hook_fn)
                    # hook = layer.register_forward_hook(noise_injector.hook_fn)
                    hook = layer.register_forward_pre_hook(hook_fn_local)
                    hooks.append(hook)
                #noise_output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                noise_output = self.model(input_ids=input_ids, attention_mask=attn_mask, past_key_values=demon_past_key_values, use_cache=use_cache, output_hidden_states=True)
                #noise_logits = output_noise.logits
                noise_hidden_states = noise_output.hidden_states
                
                '''
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                '''

                flat_loss = 0.0
                for i in range(1, len(hidden_states) - 1):
                    '''
                    conver_loss += torch.nn.functional.mse_loss(
                        hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                        , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    '''
                    grad_noise =  noise_hidden_states[i+1][torch.arange(logits.size(0)), pred_loc] - noise_hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                    grad = hidden_states[i+1][torch.arange(logits.size(0)), pred_loc] - hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                    #flat_loss += post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale
                    #flat_loss += torch.nn.functional.softplus(post_layer_norm_holder[i] @ (grad_noise - grad).t()/noise_scale)
                    flat_loss += torch.nn.functional.softplus(noise_holder[i] @ (grad_noise - grad).t()/noise_scale)

                loss += config['flat_loss_lambda'] * flat_loss.mean()
                
                '''
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())
                '''
                
                # update strength params
                optimizer.zero_grad()
                '''
                loss.backward(retain_graph= True, create_graph= True)
                gradient_holder= {}
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            gradient_holder[name] = param.grad.norm(2)
                loss.backward()
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            print((gradient_holder[name] - param.grad.norm(2)) **2 )
                pdb.set_trace()
                '''
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                for ele in hooks:
                    ele.remove()

            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
            if config['conver_loss'] or config['conver_loss_regular']:
                epoch_conv_loss = np.mean(epoch_conv_loss)
                conv_loss_list.append(epoch_conv_loss)
        
        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        if config['conver_loss'] or config['conver_loss_regular']:
            utils.plot_loss_curve(conv_loss_list, save_dir + f'/{run_name}_conv_loss_curve.png')
    
    def layernorm_adaptation_additional_learnDyT(self, config, dataset, save_dir=None, run_name=None):
        print(inspect.currentframe().f_code.co_name)
        #pt_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
        #peft_model = get_peft_model(self.model, pt_config)
        peft_model = self.model

        tuning_param_list = []
        tuning_name_list = []
        
        if config['post_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'post_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        elif config['input_attention']:
            for name, param in peft_model.named_parameters():
                if param.requires_grad and 'input_layernorm' in name:
                    tuning_name_list.append(name)
                    tuning_param_list.append(param)
            
            for param in peft_model.parameters():
                param.requires_grad = False
            
            for name, param in peft_model.named_parameters():
                if name in tuning_name_list:
                    param.requires_grad = True
        
        else:
            patch_layernorm_with_dyt_by_name(peft_model, alpha=0.1, trainable_alpha=True, match_key="layernorm", mode=config['additional_layernorm_mode'])
            
            for param in peft_model.parameters():
                param.requires_grad = False
                
            for name, param in peft_model.named_parameters():
                if 'layernorm' in name:
                    param.requires_grad = True
            print('DYT setup')
        
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                print(name)
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[
                0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        #peft_model.print_trainable_parameters()
        #print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model
        
        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data
        
        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # train
        loss_list = []
        conv_loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            epoch_conv_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                for data in batch_data:
                    input_str, _, label = dataset.apply_template(data)
                    batch_input.append(input_str)
                    batch_label.append(label)
                
                # first round
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                
                ######################
                # forward
                ######################
                print('working on the convergence bound and sharp approxy')
                output = self.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                logits = output.logits
                hidden_states = output.hidden_states
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                if not config['entropy_loss']:
                    loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                else:
                    loss = utils.entropy_from_logits(pred_logits).mean()
                # loss = torch.tensor(0.0)
                # epoch_loss.append(loss.item())
                conver_loss = 0.0
                weight_scale = [hold for hold in range(1, len(hidden_states))]
                weight_scale = torch.softmax(
                    torch.from_numpy(np.asarray(weight_scale) / config['conver_loss_regular_temp']), dim=0)
                
                if config['conver_loss']:
                    print('conver loss ')
                    for i in range(1, len(hidden_states) - 1):
                        conver_loss += torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                elif config['conver_loss_regular']:
                    print('conver regular loss')
                    
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[i - 1][torch.arange(logits.size(0)), pred_loc])
                        
                        if config['conver_loss_regular_expo']:
                            conver_loss += weight_scale[i].item() * numerator / demoninator
                        else:
                            # conver_loss += torch.log(numerator/demoninator)
                            conver_loss += numerator / demoninator
                    
                    loss = config['ce_loss_lambda'] * loss + config['conver_loss_lambda'] * conver_loss
                    epoch_conv_loss.append(conver_loss.item())
                    
                if config['pushing_loss']:
                    print('pushing_loss')
                    pushing_loss = 0.0
                    for i in range(2, len(hidden_states) - 1):
                        numerator = torch.nn.functional.mse_loss(
                            hidden_states[i][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        
                        demoninator = torch.nn.functional.mse_loss(
                            hidden_states[i + 1][torch.arange(logits.size(0)), pred_loc]
                            , hidden_states[1][torch.arange(logits.size(0)), pred_loc])
                        pushing_loss += numerator / demoninator
                    
                    loss += config['pushing_loss_lambda'] * pushing_loss
                
                '''
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())
                '''
                
                # update strength params
                optimizer.zero_grad()
                '''
                loss.backward(retain_graph= True, create_graph= True)
                gradient_holder= {}
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            gradient_holder[name] = param.grad.norm(2)
                loss.backward()
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        if param.requires_grad:
                            print((gradient_holder[name] - param.grad.norm(2)) **2 )
                pdb.set_trace()
                '''
                loss.backward()
                
                optimizer.step()
                scheduler.step()
            
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
            if config['conver_loss'] or config['conver_loss_regular']:
                epoch_conv_loss = np.mean(epoch_conv_loss)
                conv_loss_list.append(epoch_conv_loss)
        
        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        if config['conver_loss'] or config['conver_loss_regular']:
            utils.plot_loss_curve(conv_loss_list, save_dir + f'/{run_name}_conv_loss_curve.png')
    
    def layernorm_adaptation(self, config, dataset, save_dir=None, run_name=None):
        pt_config = LNTuningConfig(task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(self.model, pt_config)
        '''
        tuning_param_list = []
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                print(name)
                tuning_param_list.append(name)

        # tunn_off all the parameters
        print('turn off the gradinet require for all the parameters')
        for param in self.model.parameters():
            param.requires_grad = False

        pdb.set_trace()
        off_param_list = []
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                print(name)
                off_param_list.append(name)

        print('turn on the gradinet require for all the parameters')
        for name, param in peft_model.named_parameters():
            if name in tuning_param_list:
                param.requires_grad = True

        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                print(name)
        print('check turn on the gradinet require for all the parameters')
        '''

        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")

        # print trainable parameters
        peft_model.print_trainable_parameters()
        print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model

        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')

        # get all data
        all_data = dataset.all_data

        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # train
        loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                for data in batch_data:
                    input_str, _, label = dataset.apply_template(data)
                    batch_input.append(input_str)
                    batch_label.append(label)

                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                # forward
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())
                # update strength params
                optimizer.zero_grad()
                loss.backward()
                # get gradient
                with torch.no_grad():
                    for name, param in peft_model.named_parameters():
                        try:
                            print(f"{name}: {param.grad.norm(2): .4f}")
                        except:
                            print(f"{name}")
                optimizer.step()
                scheduler.step()
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)

        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
    
    def softprompt(self, config, dataset, save_dir=None, run_name=None):
        pt_config = PromptTuningConfig(**config['pt_config'])
        peft_model = get_peft_model(self.model, pt_config)
        
        # prepare label dict
        label_map = {}
        ans_txt_list = dataset.get_dmonstration_template()['options']
        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in self.tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            ans_tok = self.tokenizer.encode(ans_txt, add_special_tokens=False)[
                0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[label] = ans_tok  # index is the label
        print(f"label_map: {label_map}")
        
        # print trainable parameters
        peft_model.print_trainable_parameters()
        print(f'PEFT model:\n {peft_model}')
        # set model to peft model
        self.model = peft_model
        
        # init optimizer
        optim_paramters = [{'params': self.model.parameters()}]
        if config['optim'] == 'sgd':
            optimizer = torch.optim.SGD(optim_paramters, lr=config['lr'],
                                        weight_decay=config['wd'])
        elif config['optim'] == 'adamW':
            optimizer = torch.optim.AdamW(optim_paramters, config['lr'],
                                          weight_decay=config['wd'])
        elif config['optim'] == 'adam':
            optimizer = torch.optim.Adam(optim_paramters, config['lr'])
        else:
            raise ValueError('optim must be sgd, adamW or adam!')
        
        # get all data
        all_data = dataset.all_data
        
        # init lr_scheduler
        epochs, batch_size = config['epochs'], config['grad_bs']
        total_steps = epochs * len(all_data) // batch_size
        warmup_steps = int((0.05 * epochs) * (len(all_data) // batch_size))
        lr_lambda = lambda step: min(1.0, step / warmup_steps) * (1 + math.cos(math.pi * step / total_steps)) / 2 \
            if step > warmup_steps else step / warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # train
        loss_list = []
        all_data_index = list(range(len(all_data)))
        for _ in range(epochs):
            epoch_loss = []
            np.random.shuffle(all_data_index)
            for i in range(0, len(all_data), batch_size):
                batch_index = all_data_index[i: i + batch_size]
                batch_data = [all_data[idx] for idx in batch_index]
                batch_input, batch_label = [], []
                for data in batch_data:
                    input_str, _, label = dataset.apply_template(data)
                    batch_input.append(input_str)
                    batch_label.append(label)
                
                input_tok = self.tokenizer(batch_input, return_tensors='pt', padding=True)
                input_ids = input_tok['input_ids'].to(self.device)
                attn_mask = input_tok['attention_mask'].to(self.device)
                pred_loc = utils.last_one_indices(attn_mask).to(self.device)
                # forward
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
                # get prediction logits
                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                # get loss
                gt_label = torch.tensor([label_map[label] for label in batch_label]).to(self.device)
                loss = F.cross_entropy(pred_logits, gt_label, reduction='mean')
                epoch_loss.append(loss.item())
                # update strength params
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            epoch_loss = np.mean(epoch_loss)
            loss_list.append(epoch_loss)
        
        # fronzen all learnable strength params
        for param in self.model.parameters():
            param.requires_grad = False
        # set model to eval mode
        self.model.eval()
        # plot loss curve and save it
        utils.plot_loss_curve(loss_list, save_dir + f'/{run_name}_loss_curve.png')
        

    def init_strength(self, config):
        # get linear_coef size
        if type(config['layer']) == str:
            if config['layer'] == 'all':
                layers = list(range(self.num_layers))
                layer_dim = len(layers)
            elif config['layer'] == 'late':
                layers = list(range((self.num_layers*2)//3, self.num_layers))
                layer_dim = len(layers)
            elif config['layer'] == 'early':
                layers = list(range(self.num_layers//3))
                layer_dim = len(layers)
            elif config['layer'] == 'mid':
                layers = list(range(self.num_layers//3, (self.num_layers*2)//3))
                layer_dim = len(layers)
        elif type(config['layer']) == list:
            layers = config['layer']
            layer_dim = len(layers)
        else:
            raise ValueError("layer must be all, late, early, mid or a list of layer index!")

        if config['inject_method'] == 'add':
            param_size = (layer_dim, len(config['module']), 1)  # (layer_num, module_num, 1)
        elif config['inject_method'] in ['linear', 'balance']:
            param_size = (layer_dim, len(config['module']), 2)  # (layer_num, module_num, 2)
        else:
            raise ValueError("only support add, linear or balance!")
        # set inject_layers
        self.inject_layers = layers
        # init linear_coef
        linear_coef = torch.zeros(param_size, device=self.device) 
        linear_coef += torch.tensor(config['init_value'], device=self.device)
        self.linear_coef = nn.Parameter(linear_coef)
        print(f"linear_coef shape: {self.linear_coef.shape}\n")
        if not self.linear_coef.is_leaf:
            raise ValueError("linear_coef is not a leaf tensor, which is required for optimization.")
        

    def init_noise_context_vector(self, context_vector_dict):
        # init learnable context_vector
        for layer, sub_dict in context_vector_dict.items():
            for module, latent in sub_dict.items():
                noise_vector = torch.randn_like(latent).detach().cpu()
                context_vector_dict[layer][module] = noise_vector
        return context_vector_dict
            
                    
    def _get_nested_attr(self, attr_path):
        """
        Accesses nested attributes of an object based on a dot-separated string path.

        :param obj: The object (e.g., a model).
        :param attr_path: A dot-separated string representing the path to the nested attribute.
                        For example, 'transformer.h' or 'model.layers'.
        :return: The attribute at the specified path.
        """
        try:
            return reduce(getattr, attr_path.split('.'), self.model)
        except AttributeError:
            raise AttributeError(f"Attribute path '{attr_path}' not found.")
        
    def _get_layer_num(self):
        raise NotImplementedError("Please implement get_layer_num function for each model!")
    
    def _get_arribute_path(self, layer_idx, target_module):
        raise NotImplementedError("Please implement get_arribute_path function for each model!")

            
class LlamaWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__(model, tokenizer, model_config, device)
        self.embed_matrix = self.model.model.embed_tokens.weight.data
        self.embed_dim = self.model_config.hidden_size
        self.last_norm = self.model.model.norm
        
    def _get_layer_num(self):
        return len(self.model.model.layers)
    
    def _get_arribute_path(self, layer_idx, target_module):
        if target_module == "attn":
            return f"model.layers.{layer_idx}.self_attn"
        elif target_module == "mlp":
            return f"model.layers.{layer_idx}.mlp"
        elif target_module == "hidden":
            return f"model.layers.{layer_idx}"
        else:
            raise ValueError("only support att or mlp!")


class GPTWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, model_config, device):
        super().__init__(model, tokenizer, model_config, device)
        self.embed_matrix = self.model.transformer.wte.weight.data
        self.embed_dim = self.embed_matrix.size(-1)
        self.last_norm = self.model.transformer.ln_f
        
    def _get_layer_num(self):
        return len(self.model.transformer.h)
    
    def _get_arribute_path(self, layer_idx, target_module):
        if target_module == "attn":
            return f"transformer.h.{layer_idx}.attn"
        elif target_module == "mlp":
            return f"transformer.h.{layer_idx}.mlp"
        elif target_module == "hidden":
            return f"transformer.h.{layer_idx}"
        else:
            raise ValueError("only support att or mlp!")