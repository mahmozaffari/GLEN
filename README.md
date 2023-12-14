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

**Training Models**

We offer a variety of scripts for training models under different configurations:

1. Training Vanilla Model:
```
python bash_run_train.py --batch_size $bs --model model_name --selector "maxprob" --config "defaults.yaml" --ename exp_name --rid 0 --trainer "mmf" --run_type 'train-val' --user_dir path/to/user_dir  --save_dir_root path/to/save_dir_root --data_dir path/to/data_dir --json
```
2. Training with GLF loss:
```
python bash_run_train.py --model model_name --selector "maxprob" --config "gfl.yaml" --ename "GLF_lambda${lambda}"  --rid $rid --trainer "mmf" --trainer_param $lambda --run_type 'train-val' --user_dir path/to/user_dir  --save_dir_root path/to/save_dir_root --data_dir path/to/data_dir --json
```

to resume training, add the following arguments:
```
...   --resume 'current' --resume_dir path/to/experiment_dir
```
3. Testing Models:

For VQA-v2 dataset:
```
python bash_run_train.py --batch_size batch_size --model model_name --selector "maxprob" --config "defaults.yaml" --ename exp_name   --rid $rid --trainer "mmf" --run_type 'val-test' --user_dir path/to/user_dir  --save_dir_root path/to/save_dir_root --resume 'best' --resume_dir path/to/experiment_dir --save_logits --json --add_mc 
```
For AdVQA dataset, append:
```
... --args "dataset_config.vqa2_extended.annotations.test='data/val2017_advqa.npy'"
```
For movie_mcan model, use 'data/val2017_advqa_clip.npy'.

4. Low-rank Factorization:
Low-rank factorizes the model's layers specified in the corresponding config file (default is set to final layer), then evaluates the low-rank factorized model.

Specify a ```compress_ratio``` between 0 and 1 for a pre-trained model.
```
python bash_run_train.py --model model_name --selector "maxprob" --config "defaults_compress.yaml" --ename compress_exp_name  --rid 0 --trainer "compress" --compress_trainer_param compress_ratio  --run_type 'test-val' --user_dir path/to/user_dir  --save_dir_root path/to/save_dir_root --data_dir path/to/data_dir --resume 'best' --resume_dir path/to/trained_model_experiment_name  --save_logits --json --add_mc
```

5. Ensembling:
Before executing the ensembling command, it's essential to first evaluate the models using the ```--save_logits``` argument. This process stores the output logits in the directory /path/to/experiment_name/vqa2_extended_modelname_seed/reports/logits. Once these logits are saved, they can be combined through ensembling and subsequently evaluated using the command provided below:
```
python bash_run_train.py --model model_name --selector "maxprob" --config "ensemble.yaml" --trainer 'ensemble' --ename exp_name --rid 0  --run_type 'val-test' --user_dir path/to/user_dir  --save_dir_root path/to/save_dir_root --data_dir path/to/data_dir --json  --add_mc --args "ensemble.params=['path/to/model1','path/to/model2','path/to/model3']"
```
```path/to/model``` arguments must be the paths to the reports folders as mentioned above.




This repository is a modified version of [reliable_vqa](https://github.com/facebookresearch/reliable_vqa).
