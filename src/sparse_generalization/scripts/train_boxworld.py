import dill
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
from torch.utils.data import DataLoader, random_split

from sparse_generalization.utils.datasets import BasicDataset

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings(
    "ignore", ".*can be accelerated via the 'torch-scatter' package*"
)

print(f"CUDA available: {torch.cuda.is_available()}")


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg: DictConfig):
    """Main function to run the training loop and stuff for 
    box world example. 
    """
    timestamp = datetime.now().strftime("%d_%b_%Y__%Hh%Mm")
    data_path = os.path.join(cfg.data.data_dir, cfg.data.data_file)
    data_path = to_absolute_path(data_path)
    with open(data_path, 'rb') as file:
        dataset = dill.load(file)
    group_name = cfg.run_name + "_" + timestamp
    dataset = BasicDataset(dataset['X'], dataset['Y'])
    
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.seeds is None:
        print(f'\n{'='*60}')
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
        name = cfg.run_name + f'_seed_{timestamp}'
        logger = WandbLogger(**wandb_dict, name=name, config=config_dict, group=group_name) 
        
        training_set, test_set = random_split(dataset, [1-cfg.data.test_size, cfg.data.test_size])
        
        train_loader = DataLoader(training_set, cfg.data.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, cfg.data.batch_size, shuffle=True)
        
        model = instantiate(cfg.model)
        print(f"{'='*60}")
        print(model)
        trainer = Trainer(**cfg.trainer, default_root_dir='../../checkpoints/', logger=logger)
        trainer.fit(model, train_dataloaders=train_loader)
        
        test_metrics = trainer.test(model, test_loader, verbose=False)
        
        table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
        table.add_data('Test set', test_metrics[0]['test_loss'], test_metrics[0]['test_acc'])
        logger.experiment.log({'Test Sets Table': table})
        logger.experiment.finish()
    else:
        torch.backends.cudnn.deterministic = True
        for seed in cfg.seeds:
            print(f'\n{'='*60}')
            print(f'Running Seed {seed} for group {group_name}')
            print(f'\n{'='*60}')
            config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
            name = cfg.run_name + f'_seed_{seed}_{timestamp}'
            logger = WandbLogger(**wandb_dict, name=name, config=config_dict, group=group_name) 
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            generator = torch.Generator().manual_seed(seed)
            training_set, test_set = random_split(dataset, [1-cfg.data.test_size, cfg.data.test_size], generator=generator)
        
            train_loader = DataLoader(training_set, cfg.data.batch_size, shuffle=True, generator=generator)
            test_loader = DataLoader(test_set, cfg.data.batch_size, shuffle=True, generator=generator)
            
            model = instantiate(cfg.model)
            trainer = Trainer(**cfg.trainer, default_root_dir='../../checkpoints/', logger=logger)
            trainer.fit(model, train_dataloaders=train_loader)
            
            test_metrics = trainer.test(model, test_loader, verbose=False)
            
            table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
            table.add_data('Test set', test_metrics[0]['test_loss'], test_metrics[0]['test_acc'])
            logger.experiment.log({'Test Sets Table': table})
            logger.experiment.finish()

if __name__ == '__main__':
    main()