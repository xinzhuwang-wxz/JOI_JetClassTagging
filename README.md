
environment

option1: create your own environment 

1: make sure you have  miniconda/anaconda installed (https://www.anaconda.com/docs/getting-started/miniconda/install)

2: Run the following command:

    conda env create -f environment.yml

option2:  use my environment: weaver_bamboo

1: source env.sh to both activate the shell and set the date dir 

2: weaver_bamboo env activate automatically after sub job



Basic information

/JOI_JetClassTagging

	/model

	/networks

	/sub

	/yaml

	env.sh

	environment.yml

1: /model

	Model dir contains pre-training model for fine-tune on QG/Top dataset

2: /networks

	Networks dir contains PN, ParT, MIParT... networks and the finetune version

3: /sub

	This directory provides utilities to manage and submit multiple SLURM jobs automatically.

- Train_temp_class.sh: a SLURM job script template that defines the training environment and execution commands.
- train_CepcJetClass.sh: batch submission scripts that loop over lr, batch_sizes, generate job-specific scripts from the template, and submit them using sbatch.
- Infer_JOI.sh for inference from model
- Train_temp.sh, train_QuarkGluon.sh, train_QuarkGluon_FineTune.sh for QG datasets

4:/yaml

	This dir contains yaml for weaver

- JetClass_M11.yaml: for JetClassTagging, without PID
- JetClass_M11_full.yaml: for JetClassTagging, with PID

5: env.sh

	To activate shell and set the path of datasets(JetClass, TopLandscape and QuarkGluon)

6: environment.yml

	To create own environment 



RunRunRun

1: download the training dir

	


    git clone https://github.com/xinzhuwang-wxz/JOI_JetClassTagging.git
    cd JOI_JetClassTagging



2: replace the data path in env.sh

    source /hpcfs/cepc/higgsgpu/wangxinzhu/miniconda3/bin/activate
    
    export DATADIR_JetClass=  # replace
    export DATADIR_TopLandscape=
    export DATADIR_QuarkGluon=
    
    

then

    source env.sh

3: For JetClassTagging

focus on the head of /sub/train_CepcJetClass.sh

    #!/bin/bash
    cd ..
    make_dir=CepcJetClassM11Allgg  # create a dir named by this in /training
    data_dir=DATADIR_JetClass  # already set in env.sh
    epochs=30
    lr_values=(0.001 0.002 0.003)  # now supports loop lr and batch
    batch_sizes=(256 512)
    
    model=ParT # support ParT, MIParT and PN
    feature_type=full  # support full, kin
    
    
    config_path=`pwd`/yaml/JetClass_M11_full.yaml  # need to modity if kin
    
    sample_per_epoch=$((5*1000*1024))
    sample_per_epoch_val=$((2*1000*1024))
    
    

NOTE:

a. epochs, model, feature_type can be changed but now we don't support loop these params like lr and batch

b. when feature_type set to full, then config_path:

    config_path=`pwd`/yaml/JetClass_M11_full.yaml

   when feature_type set to kin, the config_path:

    config_path=`pwd`/yaml/JetClass_M11.yaml

4: if every thing has done, just
    cd sub
    chmod +x train_CepcJetClass.sh
    chmod +x Train_temp_class.sh
    
    ./train_CepcJetClass.sh

then the make_dir will created in /JOI_JetClassTagging 

