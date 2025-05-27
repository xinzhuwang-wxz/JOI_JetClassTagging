#!/bin/bash
###### Part 1 ######
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=higgsgpu
#SBATCH --job-name=JoInfo
#SBATCH --ntasks=16
#SBATCH --output=/hpcfs/cepc/higgsgpu/wangxinzhu/JOI/CEPC-Jet-Origin-Identification/training/Infer_joi/logs/log.log
#SBATCH --mem-per-cpu=24576
#SBATCH --error=/hpcfs/cepc/higgsgpu/wangxinzhu/JOI/CEPC-Jet-Origin-Identification/training/Infer_joi/logs/error.log
#SBATCH --gres=gpu:v100:1

###### Part 2 ######


#source /hpcfs/cepc/higgsgpu/wangyuexin/clusterpid/projects/PID/env.sh
#conda activate weaver

source /hpcfs/cepc/higgsgpu/wangxinzhu/JOI/CEPC-Jet-Origin-Identification/training/env.sh
conda activate weaver_bamboo





weaver --predict --data-test '/cefs/higgs/zhuyifan/sample/training/input/input/merge/test/*' \
 --data-config /hpcfs/cepc/higgsgpu/wangxinzhu/JOI/CEPC-Jet-Origin-Identification/training/yaml/JetClass_M11.yaml \
 --network-config /hpcfs/cepc/higgsgpu/wangxinzhu/JOI/CEPC-Jet-Origin-Identification/training/networks/example_ParticleTransformer.py \
 --model-prefix /hpcfs/cepc/higgsgpu/zhangkl/training/net_best_epoch_state.pt \
 --batch-size 512 \
 --predict-output /hpcfs/cepc/higgsgpu/wangxinzhu/JOI/CEPC-Jet-Origin-Identification/training/Infer_joi/output/infer.root \
 --log /hpcfs/cepc/higgsgpu/wangxinzhu/JOI/CEPC-Jet-Origin-Identification/training/Infer_joi/logs/Log.log \
 --tensorboard TB