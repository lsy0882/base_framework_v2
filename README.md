# Deep Learning Base Framework (made by Sangyoun Lee)
<br>

## Guide
### 0. Base
* Code architecture diagram ![Code Architecture Diagram](https://github.com/lsy0882/MDFD/releases/download/0.0.1/Code_Architecture_Diagram.png)
<br>

### 1. Git Clone / Conda Virtual Environment Setup
* It is assumed that git and conda installation and setup have been completed.
```shell
(Caution) Install the version of python & torch that matches your OS/GPU environment.
(Note) The version of torch used for the project is as follows.

# Git clone 
cd <your_dir_path>
git clone https://github.com/lsy0882/base_framework.git
cd <your_dir_path>/base_framework

# Conda virtual environment setup
conda create -n base_framework python=3.9
conda activate base_framework
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
<br>

### 2. Training
* "train.sh" file setup
```shell
# Edit the contents of file using an editor such as vim or vi.
vim train.sh

# Explanation of arguments within train.sh
python3 main.py \
  --mode train \ # choose the one ['train', 'test'].
  --model Hw1p2Net # choose the model class name(= model directory name)
```
* "models/<model_name>/configs.yaml" file setup
```shell
# Edit the contents of file using an editor such as vim or vi.
vim models/<model_name>/configs.yaml

# Explanation of keys and values within config.yaml
wandb:
  login: 
    key: "<your wandb login key>"
  init: ### Ref: https://docs.wandb.ai/ref/python/init
    project: "<your project name>" 
    entity: "<your wandb profile name>"
    save_code: true ### Don't change
    group: "" ### Ref: https://docs.wandb.ai/guides/runs/grouping
    job_type: "<purpose of this code" ### e.g: "data-preprocessing", "training", etc...
    tags: ["Hw1p2Net", "Small"] ### e.g: [Network, Size, etc...]
    name: "Hw1p2Net_Small_v1.0.0" ### "Network"_"Size"_"Version" | Version policy: v{Architecture change}_{Method/Block change}_{Layer/Minor change}
    notes: "Testing wandb setting" ### Insert changes(plz write details)
    dir: "./wandb" ### Don't change
    resume: "auto" ### Don't change
    reinit: false ### Don't change
    magic: null ### Don't change
    config_exclude_keys: [] ### Don't change
    config_include_keys: [] ### Don't change
    anonymous: null ### Don't change
    mode: "online" ### Don't change
    allow_val_change: true ### Don't change
    force: false ### Don't change
    sync_tensorboard: false ### Don't change
    monitor_gym: false ### Don't change
    config: ### Record and use all parameters and variables
      dataset:
        name: "hw1p2"
        modality: "audio"
        data_path: "/home/lsy/hw1p2"
        context: 20
      dataloader:
        batch_size: 1024
        pin_memory: true
        num_workers: 0
      model:
        # input_size: 100
        # output_size: 100
        options:
          test_ignore_this_var: true
      criterion: ### Ref: https://pytorch.org/docs/stable/nn.html#loss-functions
        name: "CrossEntropyLoss" ### Choose a torch.nn's class(=attribute) e.g. ["CrossEntropyLoss", "MSELoss", "Custom", ...] / You can build your criterion :)
      optimizer: ### Ref: https://pytorch.org/docs/stable/optim.html#algorithms
        name: "Adamw" ### Choose a torch.optim's class(=attribute) e.g. ["Adam", "Adamw", "SGD", ...] / You can build your optimizer :)
        Adam: ### Add or modify instance & args using reference link
          lr: 1.0e-3
          weight_decay: 1.0e-2
        AdamW:
          lr: 1.0e-3
          weight_decay: 1.0e-2
        SGD:
          lr: 1.0e-3
          momentum: 0.9
          weight_decay: 1.0e-2
        Custom:
          custom_arg1:
          custom_arg2:
      scheduler: ### Ref(& find "How to adjust learning rate"): https://pytorch.org/docs/stable/optim.html#algorithms
        name: "StepLR" ### Choose a torch.optim.lr_scheduler's class(=attribute) e.g. ["StepLR", "ReduceLROnPlateau", "Custom"] / You can build your scheduler :)
        StepLR: ### Add or modify instance & args using reference link
          step_size: 5
          gamma: 0.9
        ReduceLROnPlateau:
          mode: "min"
          min_lr: 1.0e-10
          factor: 0.9
          patience: 5
        Custom:
          custom_arg1:
          custom_arg2:
      trainer:
        epoch: 100
        gpuid: "0" ### "0"(single-gpu) or "0, 1" (multi-gpu)
```
* run .sh file to train
```shell
sh train.sh
```