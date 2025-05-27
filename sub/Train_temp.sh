#!/bin/bash

###### Part 1 ######
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=higgsgpu
#SBATCH --job-name=JOBNAME
#SBATCH --ntasks=16
#SBATCH --output=logs/log.log
#SBATCH --mem-per-cpu=24576
#SBATCH --gres=gpu:v100:1
#SBATCH --error=logs/error.log

###### Part 2 ######
srun -l hostname

/usr/bin/nvidia-smi -L

echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"


#source /hpcfs/cepc/higgsgpu/wangyuexin/clusterpid/projects/PID/env.sh
#conda activate weaver

source /hpcfs/cepc/higgsgpu/wangxinzhu/JOI/CEPC-Jet-Origin-Identification/training/env.sh
conda activate weaver_bamboo

# shellcheck disable=SC2145
echo "args: $@"
_DIR=MAINDIR

DATA_DIR=${DATADIR}
[[ -z $DATA_DIR ]] && echo "No DATADIR specified!" && exit 1

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
NGPUS=1
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi


# PN, PFN, PCNN, ParT, MIParT
model=MODEL
extraopts=""
if [[ "$model" == "ParT" ]]; then
    modelopts="${_DIR}/networks/example_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
#    lr="1e-3"
elif [[ "$model" == "ParT-FineTune" ]]; then
    modelopts="${_DIR}/networks/example_ParticleTransformer_finetune.py --use-amp --optimizer-option weight_decay 0.01"
#    lr="1e-4"
    extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none"
elif [[ "$model" == "PN" ]]; then
    modelopts="${_DIR}/networks/example_ParticleNet.py"
#    lr="1e-2"
elif [[ "$model" == "PN-FineTune" ]]; then
    modelopts="${_DIR}/networks/example_ParticleNet_finetune.py"
#    lr="1e-3"
    extraopts="--optimizer-option lr_mult (\"fc_out.*\",50) --lr-scheduler none"
elif [[ "$model" == "MIParT" ]]; then
    modelopts="${_DIR}/networks/example_MIParticleTransformer.py"
#    lr="1e-3"
elif [[ "$model" == "MIParT-L-FineTune" ]]; then
    modelopts="${_DIR}/networks/example_MIParticleTransformer_L_finetune.py"
#    lr="0.00016"
    extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none"
elif [[ "$model" == "PFN" ]]; then
    modelopts="${_DIR}/networks/example_PFN.py"
#    lr="2e-2"
    extraopts="--batch-size 4096"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="${_DIR}/networks/example_PCNN.py"
#    lr="2e-2"
    extraopts="--batch-size 4096"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "kinpidplus", "full"
# QG --> kinpid
# Top --> kin
FEATURE_TYPE=FEATURETYPE
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kinpid"

if [[ "${FEATURE_TYPE}" == "kin" ]]; then
    pretrain_type="kin"
elif [[ "${FEATURE_TYPE}" =~ ^(kinpid|kinpidplus)$ ]]; then
    pretrain_type="kinpid"
else
    echo "feature type ${FEATURE_TYPE}"
fi

if [[ "$model" == "ParT-FineTune" ]]; then
    modelopts+=" --load-model-weights ${_DIR}/models/ParT_${pretrain_type}.pt"
fi
if [[ "$model" == "PN-FineTune" ]]; then
    modelopts+=" --load-model-weights ${_DIR}/models/ParticleNet_${pretrain_type}.pt"
fi
if [[ "$model" == "MIParT-FineTune" ]]; then
    modelopts+=" --load-model-weights ${_DIR}/models/MIParT_${pretrain_type}.pt"
fi


if [[ $DATA_DIR == *QuarkGluon ]]; then
    # shellcheck disable=SC2125
    dataopts="--data-train "${DATA_DIR}/train_file_*.parquet" --data-test "${DATA_DIR}/test_file_*.parquet""

elif [[ $DATA_DIR == *merge ]]; then
    # shellcheck disable=SC2125
    dataopts="--data-train "B:${DATA_DIR}/bb/merge_bb_*.root" \
    "C:${DATA_DIR}/cc/merge_cc_*.root" \
    "G:${DATA_DIR}/gg/merge_gg_*.root" \
    "Bbar:${DATA_DIR}/bbbar/merge_bbbar_*.root" \
    "Cbar:${DATA_DIR}/ccbar/merge_ccbar_*.root" \
    "D:${DATA_DIR}/dd/merge_dd_*.root" \
    "Dbar:${DATA_DIR}/ddbar/merge_ddbar_*.root" \
    "U:${DATA_DIR}/uu/merge_uu_*.root" \
    "Ubar:${DATA_DIR}/uubar/merge_uubar_*.root" \
    "S:${DATA_DIR}/ss/merge_ss_*.root" \
    "Sbar:${DATA_DIR}/ssbar/merge_ssbar_*.root" \
    --data-val "${DATA_DIR}/val/*" \
    --data-test "${DATA_DIR}/test/*""
else
    echo "Invalid data directory $DATA_DIR ! "
    exit 1
fi

$CMD \
    --data-config CONFIGPATH --network-config ${modelopts} \
    --model-prefix pt/net \
    $dataopts $batchopts \
    --num-workers 0 --fetch-step 1 --in-memory --train-val-split SPLIT \
    --batch-size BATCHSIZE --samples-per-epoch SAMPLEperEPOCH --samples-per-epoch-val SAMPLEPerEPOCHval --num-epochs EPOCHS --gpus 0 \
    --start-lr LR --optimizer ranger --log logs/Log.log --predict-output output/pred.root \
    --tensorboard TB \
    ${extraopts}
