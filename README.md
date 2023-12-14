# Calibrating Visual Question Answering models

## Description
In this project, we aim to achieve better calibrated Visual Question Answering (VQA) models. Our approach involves modifying and enhancing existing models to improve their reliability and accuracy in interpreting and answering visual queries. The project is inspired by and builds upon the methodologies presented in [reliable_vqa](https://github.com/facebookresearch/reliable_vqa).

<!-- Project description and usage instructions will be updated soon. -->

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
To set up this project, you will need to follow the installation instructions for MMF (Modular Framework for Vision & Language Research). Please refer to the MMF installation guide at the MMF Installation Link.


## Usage

###Training Models

Our project offers various scripts to train models under different configurations:

1. Training Vanilla Model:
```
python bash_run_train.py --batch_size $bs --model model_name --selector "maxprob" --config "defaults.yaml" --ename exp_name --rid 0 --trainer "mmf" --run_type 'train-val' --user_dir path/to/user_dir  --save_dir_root path/to/save_dir_root --data_dir path/to/data_dir --json
```
2. Training by GLF loss:
```
python bash_run_train.py --model model_name --selector "maxprob" --config "gfl.yaml" --ename "GLF_lambda${lambda}"  --rid $rid --trainer "mmf" --trainer_param $lambda --run_type 'train-val' --user_dir path/to/user_dir  --save_dir_root path/to/save_dir_root --data_dir path/to/data_dir --json #--resume 'current' --resume_dir /home/mm3424/Saves/reliable_vqa/$model+maxprob/DRO_lambda${lambda}_bs$bs-$rid --json --args "checkpoint.reset.optimizer=False" #--args "training.max_updates=360000"
```

to resume training, add the following arguments:
```
--resume 'current' --resume_dir path/to/experiment_dir
```



This repository is a modified version of [reliable_vqa](https://github.com/facebookresearch/reliable_vqa).
