#!/bin/bash
MultiRun (){
sessionname=$1
File=$2
Param=$3
Tunning=$4
echo "${sessionname}"

screen -dmS "$sessionname"

screen -S $sessionname -X stuff "cd /home/engs2575/project/I2CL_learning^M"
screen -S $sessionname -X stuff "source activate i2cl_env^M"
sleep 1.0s
echo ${File}
screen -S $sessionname -X stuff "bash ${File} ${Param} ${sessionname} ${Tunning}^M"
}

Tunning=1

Target_Para_List=(o_proj q_proj k_proj v_proj)
Target_Para_List=(o_proj)

Tunning_list=(0 )
for ((idx=0; idx<${#Target_Para_List[@]}; idx++)); do
Param=${Target_Para_List[$idx]}
Tunning=${Tunning_list[$idx]}


sessionname=lora_analysis
sessionname=raw_result_for_comparsion
sessionname=lora_debugging
sessionname=lora16_scripttest
sessionname=softprompt
sessionname=taskvector
sessionname=label_ancho
sessionname=ia3_llama2
sessionname=ia3_gpt2
sessionname=test_trainable_parameters
sessionname=lora64_llama3_instruct
sessionname=llama3_instruct_ours

File=bash_script/bash_sub_script/sub_run.sh
File=bash_script/bash_sub_script/sub_lora.sh
File=bash_script/bash_sub_script/sub_layernorm_adaptation.sh

CHECK=${sessionname}

MultiRun ${sessionname} ${File} ${Param} ${Tunning}
done
