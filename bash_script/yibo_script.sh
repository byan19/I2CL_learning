#!/bin/bash
MultiRun (){
sessionname=$1
File=$2
Param=$3
Tunning=$4
echo "${sessionname}"

screen -dmS $sessionname

screen -S $sessionname -X stuff "source activate /ibex/user/yangy0o/by_g/by_g_env/i2cl_env^M"
screen -S $sessionname -X stuff "export HF_HOME=/ibex/user/yangy0o/by_g/huggingface_cache^M"
screen -S $sessionname -X stuff "export HF_HUB_CACHE=/ibex/user/yangy0o/by_g/huggingface_cache^M"

screen -S $sessionname -X stuff "cd /ibex/user/yangy0o/by_g/I2CL_learning^M"
sleep 1.0s
echo ${File}
screen -S $sessionname -X stuff "salloc --job-name by_g --gres=gpu:a100:1 --time=25:00:00 --cpus-per-task=40 srun bash ${File} ${Param} ${sessionname} > Z_${sessionname}_$(date +"T%H:%M:%S_D%d_%m_%Y").log 2>&1^M"

}

Tunning=1

Target_Para_List=(q_proj k_proj v_proj o_proj mlp.down_proj mlp.up_proj gate_proj)
Target_Para_List=(v_proj )

Tunning_list=(0 )
for ((idx=0; idx<${#Target_Para_List[@]}; idx++)); do
Param=${Target_Para_List[$idx]}
Tunning=${Tunning_list[$idx]}

sessionname=inputlayernorm_bound00001alone
File=bash_script/bash_sub_script/sub_layernorm_adaptation.sh

CHECK=${sessionname}

MultiRun ${sessionname} ${File} ${Param} ${Tunning}
done