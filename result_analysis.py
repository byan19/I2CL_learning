import json
import numpy as np

def analysis(exp_name, model_name, dataset_list):
	ours_acc_list = []
	ours_acc_std_list = []
	ours_macro_f1_list = []
	ours_macro_f1_std_list = []
	for dataset in dataset_list:
		file_name = f'{exp_name}/meta-llama/{model_name}/{dataset}/result_dict.json'

		with open(file_name) as file:
			data = json.load(file)

		result_holder = {}
		if not data['test_result']['zero_shot'] ==[]:
			result_holder["zero_shot"] = {'acc': data['test_result']['zero_shot']['acc'], 'macro_f1':  data['test_result']['zero_shot']['macro_f1']}
		else:
			result_holder["zero_shot"] = []

		if not data['test_result']['few_shot'] ==[]:
			acc_ = []
			macro_ = []
			for ele in data['test_result']['few_shot']:
				acc_.append(ele["acc"])
				macro_.append(ele['macro_f1'])

			acc_ = np.asarray(acc_)
			macro_ = np.asarray(macro_)

			acc_mean = acc_.mean()
			acc_std = acc_.std()

			macro_mean = macro_.mean()
			macro_std = macro_.std()

		acc_ = []
		macro_ = []
		for ele in data['test_result']['ours']:
			acc_.append(ele["acc"])
			macro_.append(ele['macro_f1'])

		acc_ = np.asarray(acc_)
		macro_ = np.asarray(macro_)

		acc_mean = acc_.mean()
		acc_std = acc_.std()

		macro_mean = macro_.mean()
		macro_std = macro_.std()

		ours_acc_list.append(acc_mean)
		ours_acc_std_list.append(acc_std)
		ours_macro_f1_list.append(macro_mean)
		ours_macro_f1_std_list.append(macro_std)
		print(ours_acc_list)


if __name__ == "__main__":
	exp_name = 'baseline_layernorm_Llama2'
	model_name = 'Llama-2-7b-hf'
	dataset_list = ['sst2', 'sst5', 'trec','agnews', 'subj', 'hate_speech18', 'dbpedia', 'emoc', 'mr' ]
	analysis(exp_name, model_name, dataset_list)