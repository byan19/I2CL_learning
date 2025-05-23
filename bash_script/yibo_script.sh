#!/bin/bash
MultiRun (){
sessionname=$1
File=$2
Param=$3
Tunning=$e
echo "${sessionname}"

screen -dmS $sessionname

screen -S $sessionname -X stuff "source activate /ibex/user/yangy0o/by_g/by_g_env/i2cl_env^M"
#screen -S $sessionname -X stuff "export HF_HOME=/ibex/user/yangy0o/by_g/huggingface_cache^M"
#screen -S $sessionname -X stuff "export HF_HUB_CACHE=/ibex/user/yangy0o/by_g/huggingface_cache^M"

screen -S $sessionname -X stuff "export HF_DATASETS_CACHE=/ibex/user/yangy0o/huggingface_datasets^M"
screen -S $sessionname -X stuff "export HF_HOME=/ibex/user/yangy0o/huggingface_models^M"

screen -S $sessionname -X stuff "cd /ibex/user/yangy0o/by_g/I2CL_learning^M"
sleep 1.0s
echo ${File}
#screen -S $sessionname -X stuff "salloc --job-name by_g --account conf-neurips-2025.05.22-ghanembs --gres=gpu:a100:1 --time=10:00:00 --cpus-per-task=10 srun bash ${File} ${Param} ${sessionname} > Z_${sessionname}_$(date +"T%H:%M:%S_D%d_%m_%Y").log 2>&1^M"
screen -S $sessionname -X stuff "salloc --job-name by_g --account conf-neurips-2025.05.22-ghanembs --gres=gpu:a100:1 --mem=50G --time=7:00:00 --cpus-per-task=10 srun bash ${File} ${Param} ${sessionname} > Z_${sessionname}_$(date +"T%H:%M:%S_D%d_%m_%Y").log 2>&1^M"

}

Tunning=1

Target_Para_List=(q_proj k_proj v_proj o_proj mlp.down_proj mlp.up_proj gate_proj)
Target_Para_List=(v_proj )

Tunning_list=(0 )
for ((idx=0; idx<${#Target_Para_List[@]}; idx++)); do
Param=${Target_Para_List[$idx]}
Tunning=${Tunning_list[$idx]}

sessionname=version4_test
sessionname=lora16_llama2
sessionname=lora1all_llama2_ours
sessionname=ia3_llama3_may9
sessionname=llama3_instruct_lora16
sessionname=lora1_llama3_instruct_ours

File=bash_script/bash_sub_script/sub_layernorm_adaptation.sh
File=bash_script/bash_sub_script/sub_run.sh
File=bash_script/bash_sub_script/sub_lora.sh

CHECK=${sessionname}

MultiRun ${sessionname} ${File} ${Param} ${Tunning}
done