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
python bash_run_train.py --model model_name --selector "maxprob" --config "gfl.yaml" --ename "GLF_lambda${lambda}"  --rid $rid --trainer "mmf" --trainer_param $lambda --run_type 'train-val' --user_dir path/to/user_dir  --save_dir_root path/to/save_dir_root --data_dir path/to/data_dir --json
```

to resume training, add the following arguments:
```
...   --resume 'current' --resume_dir path/to/experiment_dir
```
3. Testing Models:

To test on the VQA-v2 dataset run the following command:
```
python bash_run_train.py --batch_size batch_size --model model_name --selector "maxprob" --config "defaults.yaml" --ename exp_name   --rid $rid --trainer "mmf" --run_type 'val-test' --user_dir path/to/user_dir  --save_dir_root path/to/save_dir_root --resume 'best' --resume_dir path/to/experiment_dir --save_logits --json --add_mc 
```
to test on the AdVQA dataset, add the following to the end of the command:
```
... --args "dataset_config.vqa2_extended.annotations.test='data/val2017_advqa.npy'"
```
for the movie_mcan model, use 'data/val2017_advqa_clip.npy' in the above command.

4. Low-rank Factorize:
```

```


6. Ensembling:
```
python bash_run_train.py --model model_name --selector "maxprob" --config "ensemble.yaml" --trainer 'ensemble' --ename exp_name --rid 0  --run_type 'val-test' --user_dir $user_dir  --save_dir_root $save_dir_root --data_dir $data_dir --json  --add_mc --args $ensemble_params
```




This repository is a modified version of [reliable_vqa](https://github.com/facebookresearch/reliable_vqa).
