import hydra
import os
import numpy as np
import random
import torch
import warnings
import wandb

from datetime import datetime
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import Self

print(f"CUDA available: {torch.cuda.is_available()}")

class BasicDataset(Dataset):
    def __init__(self: Self, x: Tensor, y: Tensor):
        super().__init__()
        self.x = x
        self.y = y.float()
    
    def __len__(self: Self):
        return self.x.size(0)
    
    def __getitem__(self: Self, index: int):
        return self.x[index], self.y[index]


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg: DictConfig):
    """Main function to run the training loop and stuff for 
    basic toy example. 
    """
    timestamp = datetime.now().strftime("%d_%b_%Y__%Hh%Mm")
    data_path = os.path.join(cfg.data.data_dir, cfg.data.data_file)
    data_path = to_absolute_path(data_path)
    raw_data = torch.load(data_path)
    group_name = cfg.run_name + "_" + timestamp
    
    torch.backends.cudnn.deterministic = True
    
    for seed in cfg.seeds:
        print(f'\n{'='*60}')
        print(f'Running Seed {seed} for group {group_name}')
        print(f'\n{'='*60}')
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
        name = cfg.run_name + f'_seed_{seed}_{timestamp}'
        logger = WandbLogger(**wandb_dict, name=name, config=config_dict, group=group_name, job_type=cfg.job_type) 
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        training_set = BasicDataset(raw_data['training_x'], raw_data['training_y'])
        test_expl_1 = BasicDataset(raw_data['test1_x'], raw_data['test1_y'])
        test_expl_2 = BasicDataset(raw_data['test2_x'], raw_data['test2_y'])
        
        train_loader = DataLoader(training_set, cfg.data.batch_size)
        test_loader_1 = DataLoader(test_expl_1, cfg.data.batch_size)
        test_loader_2 = DataLoader(test_expl_2, cfg.data.batch_size)
        
        model = instantiate(cfg.model)
        trainer = Trainer(**cfg.trainer, default_root_dir='../../checkpoints/', logger=logger)
        trainer.fit(model, train_dataloaders=train_loader)
        
        test_metrics_1 = trainer.test(model, test_loader_1, verbose=False)
        test_metrics_2 = trainer.test(model, test_loader_2, verbose=False)
        
        table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
        table.add_data('Test set Expl. 1', test_metrics_1[0]['test_loss'], test_metrics_1[0]['test_acc'])
        table.add_data('Test set Expl. 2', test_metrics_2[0]['test_loss'], test_metrics_2[0]['test_acc'])
        logger.experiment.log({'Test Sets Table': table})

if __name__ == '__main__':
    main()