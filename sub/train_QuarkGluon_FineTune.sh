#!/bin/bash
cd ..
make_dir=QuarkGluonFinetune_check
data_dir=DATADIR_QuarkGluon
epochs=20
lr_values=(0.00001)
batch_sizes=(100)

model=MIParT-L-FineTune
feature_type=kinpid

config_path=`pwd`/yaml/qg_kinpid.yaml

split=0.8889
sample_per_epoch=1600000
sample_per_epoch_val=200000


timestamp=$(date "+%Y-%m-%d_%H-%M")

DIR=`pwd`

if [ -d "${DIR}/${make_dir}" ]; then
    cd ${make_dir}
else
    mkdir $make_dir
    cd $make_dir
fi

for lr in "${lr_values[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        job_name="${make_dir}_${model}_${feature_type}_lr${lr}_bs${bs}_e${epochs}_${timestamp}"

        if [ ! -d "$job_name" ]; then
            mkdir "$job_name"
        fi

        cd $job_name
        cp -r $DIR/sub/Train_temp.sh .
        sed -i "s:MAINDIR:$DIR:g" Train_temp.sh
        sed -i "s:DATADIR:$data_dir:g" Train_temp.sh
        sed -i "s:MODEL:$model:g" Train_temp.sh
        sed -i "s:FEATURETYPE:$feature_type:g" Train_temp.sh
        sed -i "s:CONFIGPATH:$config_path:g" Train_temp.sh
        sed -i "s:EPOCHS:$epochs:g" Train_temp.sh
        sed -i "s:LR:$lr:g" Train_temp.sh
        sed -i "s:BATCHSIZE:$bs:g" Train_temp.sh
        sed -i "s:SPLIT:$split:g" Train_temp.sh
        sed -i "s:SAMPLEperEPOCH:$sample_per_epoch:g" Train_temp.sh
        sed -i "s:SAMPLEPerEPOCHval:$sample_per_epoch_val:g" Train_temp.sh
        sed -i "s:JOBNAME:$job_name:g" Train_temp.sh

        sbatch Train_temp.sh

        cd ..
    done
done

cd $DIR
