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
screen -S $sessionname -X stuff "source activate distillation_a100^M"
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

sessionname=cifar100_20ipc_MTT
sessionname=implicit_cifar10ipc10_reproduce_cur50_realnoise
sessionname=baseline_withlong_iteration
sessionname=biaggregation_3gradient
sessionname=biaggregation_avg_3gradient_datm_flat_m09
sessionname=opMatching_step20_ep5_lr1000_bfm06_param1momentum10_m04_cosine
sessionname=opMatching_step20_ep5_lr1000_bfm06_param0momentum001_no
sessionname=opMatching_step20_ep2_lr1000_bfm06_param1momentum001_l2l2
sessionname=momentum06_recap_ep3_00005l2_09995param
sessionname=bmrcnew_anacondatest

File=bash_script/bash_sub_script/sub_run_distillplus_cifar10_3ipc.sh
File=bash_script/bash_sub_script/sub_run_distillplus_cifar10_5ipc.sh
File=bash_script/bash_sub_script/sub_run_distillplus_cifar100_10ipc.sh
File=bash_script/bash_sub_script/sub_run_baseline_tiny_3ipc.sh
File=bash_script/bash_sub_script/sub_run_distillplus_cifar100_20ipc.sh
File=bash_script/bash_sub_script/sub_run_distillplus_cifar100_10ipc.sh
File=bash_script/bash_sub_script/sub_run_distill_implicit_tiny_10ipc.sh
File=bash_script/bash_sub_script/sub_run_distill_biaggregation_cifar10_10ipc.sh
File=bash_script/bash_sub_script/sub_run_distill_implicit_cifar100_1ipc.sh
File=bash_script/bash_sub_script/sub_run_distill_biaggregation_cifar10_10ipc.sh
File=bash_script/bash_sub_script/sub_run_distill_biaggregation_cifar100_10ipc.sh
File=bash_script/bash_sub_script/sub_run_distill_implicit_cifar10_10ipc.sh
File=bash_script/bash_sub_script/sub_run_baseline.sh
File=bash_script/bash_sub_script/sub_run_distill_momentummatching_cifar10_10ipc.sh

MultiRun ${sessionname} ${File} ${GPU} ${Tunning}
done