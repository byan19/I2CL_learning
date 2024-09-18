#!/bin/bash
MultiRun (){
sessionname=$1
File=$2
GPU=$3
Tunning=$4
echo "${sessionname}"

screen -dmS $sessionname

screen -S $sessionname -X stuff "module load Anaconda3/2024.02-1^M"
screen -S $sessionname -X stuff "eval "$(conda shell.bash hook)"^M"
screen -S $sessionname -X stuff "cd /well/clifton/users/acr511/project/I2CL_learning^M"
#screen -S $sessionname -X stuff "source activate distill^M"
screen -S $sessionname -X stuff "source activate i2cl_env^M"
sleep 1.0s
echo ${File}
screen -S $sessionname -X stuff "sbatch -p gpu_short --gres gpu:a100-pcie-80gb:1 ${File} ${GPU} ${sessionname} ${Tunning}^M"
}

GPU_list=(0 1 2  )
Tunning_list=(7 14 21 )

GPU_list=(2 )

for ((idx=0; idx<${#GPU_list[@]}; idx++)); do
GPU=${GPU_list[$idx]}
Tunning=${Tunning_list[$idx]}

sessionname=layernorm_SAM1e2

File=bash_script/bash_sub_script/sub_layernorm_adaptation.sh

MultiRun ${sessionname} ${File} ${GPU} ${Tunning}
done