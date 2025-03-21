#!/bin/bash
MultiRun (){
sessionname=$1
File=$2
Param=$3
Tunning=$4
echo "${sessionname}"

screen -dmS "$sessionname"

screen -S $sessionname -X stuff "cd /home/engs2575/project/I2CL_learning^M"
screen -S $sessionname -X stuff "source activate lm_eval^M"
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

sessionname=I2CL_contextlearning_noCE
sessionname=I2CL_contextlearning_dyt
sessionname=I2CL_contextlearning_addtional_learn
sessionname=I2CL_contextlearning_weightexpo_desk
sessionname=I2CL_contextlearning_weightexpo_desk
sessionname=I2CL_contextlearning_layernorm_conv1ONLY
File=bash_script/bash_sub_script/sub_layernorm_adaptation.sh

CHECK=${sessionname}

MultiRun ${sessionname} ${File} ${Param} ${Tunning}
done
