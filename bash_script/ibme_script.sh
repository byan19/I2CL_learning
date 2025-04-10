#!/bin/bash
MultiRun (){
sessionname=$1
File=$2
Param=$3
Tunning=$4
echo "${sessionname}"

screen -dmS "$sessionname"

screen -S $sessionname -X stuff "cd /users/engs2575/project/I2CL_learning^M"
screen -S $sessionname -X stuff "source activate i2cl_env^M"
sleep 1.0s
echo ${File}
#screen -S $sessionname -X stuff "sbatch -w node04 --gres=gpu:1 ${File} ${GPU} ${sessionname} ${Tunning}^M"
#screen -S $sessionname -X stuff "sbatch -w node05 --gres=gpu:1 ${File} ${GPU} ${sessionname} ${Tunning}^M"
screen -S $sessionname -X stuff "sbatch -w node05 --gres=gpu:1 ${File} ${Param} ${sessionname} ${Tunning} ^M"
}

Tunning=1

Target_Para_List=(o_proj)
Target_Para_List=(mlp.down_proj )
Target_Para_List=(q_proj k_proj v_proj)

Target_Para_List=(v_proj )
Tunning_list=(0 )
for ((idx=0; idx<${#Target_Para_List[@]}; idx++)); do
Param=${Target_Para_List[$idx]}
Tunning=${Tunning_list[$idx]}

sessionname=I2CL_contextlearning_dyt_learning
sessionname=I2CL_contextlearning_layernorm_regular_convloss
sessionname=I2CL_contextlearning_alllayernorm_convregular1e-3
sessionname=I2CL_alllayernorm_convregular1e-3
sessionname=I2CL_allLN_convregular1e-2_log
sessionname=I2CL_allLN_convregular1e-2_entropy
sessionname=I2CL_allLN_convregular1e-2_DYTlearning_alphachecking
sessionname=I2CL_allLN_convregular1e-2_MinuesentropyFlatness
sessionname=I2CL_allLN_convregular_testagain
sessionname=I2CL_allLN_convregular_sharpnessencoding_softplu
sessionname=I2CL_convregular_sharpnessencoding_N1e-2C_flat1e-3
sessionname=I2CL_adding_demonstration_epoch50_conver_regular
sessionname=version4_1sample_0demon
File=bash_script/bash_sub_script/sub_layernorm_adaptation.sh

MultiRun ${sessionname} ${File} ${Param} ${Tunning}
done
