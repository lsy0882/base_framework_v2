import os
import torch
from loguru import logger
from .dataset import get_dataloaders
from .model import Model
from .engine import Engine
from utils import util_system, util_implement
from utils.decorators import *

# Setup logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/system_log.log")
logger.add(log_file_path, level="DEBUG", mode="w")

@logger_wraps()
def main(args):
    
    ''' Build Setting '''
    # Call configuration file (configs.yaml)
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
    yaml_dict = util_system.parse_yaml(yaml_path)
    
    # Run wandb and get configuration
    wandb_run = util_system.wandb_setup(yaml_dict)
    config = wandb_run.config if wandb_run else yaml_dict["wandb"]["init"]["config"] # wandb login success or fail
    
    # Call DataLoader [train / valid / test / etc...]
    dataloaders = get_dataloaders(args, config["dataset"], config["dataloader"])
    
    ''' Build Model '''
    # Call network model
    model = Model(**config["model"])
    if wandb_run: util_system.log_model_information_to_wandb(wandb_run, model, (2, 32000), os.path.dirname(os.path.abspath(__file__))) # Record artifact on wandb for logging model architecture

    ''' Build Engine '''
    # Call gpu id & device
    gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
    device = torch.device(f'cuda:{gpuid[0]}')
    
    # Call Implement [criterion / optimizer / scheduler]
    criterions = util_implement.CriterionFactory(config["criterion"], device).get_criterions()
    optimizers = util_implement.OptimizerFactory(config["optimizer"], model.parameters()).get_optimizers()
    schedulers = util_implement.SchedulerFactory(config["scheduler"], optimizers).get_schedulers()
    
    # Call & Run Engine
    engine = Engine(config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device, wandb_run)
    engine.run()