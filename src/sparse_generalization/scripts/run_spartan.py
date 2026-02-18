import dill
import hydra
import gc
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
from torch.utils.data import DataLoader
from torchsummary import summary

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
        data = dill.load(file)
    group_name = cfg.run_name + "_" + timestamp
    dataset = BasicDataset(data['X_train'], data['Y_train'])
    test_dataset_ind = BasicDataset(data['X_test_ind'], data['Y_test_ind'])
    test_dataset_ood = BasicDataset(data['X_test_ood'], data['Y_test_ood'])
    
    print(OmegaConf.to_yaml(cfg, resolve=True))


    if cfg.seeds is None:
        print(f'\n{'='*60}')
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
        name = cfg.run_name + f'_seed_{timestamp}'
        logger = WandbLogger(**wandb_dict, name=name, config=config_dict, group=group_name) 
        
        train_loader = DataLoader(dataset, cfg.data.batch_size, shuffle=True)
        test_loader_ind = DataLoader(test_dataset_ind, cfg.data.batch_size, shuffle=False)
        test_loader_ood = DataLoader(test_dataset_ood, cfg.data.batch_size, shuffle=False)
        
        model = instantiate(cfg.model)
        model.logger = logger
        print(summary(model, (10, 10, 3), device='cuda'))
        model.fit(dataloader=train_loader, num_epochs=cfg.trainer.max_epochs)
        
        if cfg.save:
            torch.save(model.state_dict(), f'checkpoints/{cfg.run_name}_{timestamp}.pt')
        
        test_metrics_1 = model.test('id', test_loader_ind)
        test_metrics_2 = model.test('ood', test_loader_ood)
        
        table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
        table.add_data('Test set ID', test_metrics_1['loss'], test_metrics_1['acc'])
        table.add_data('Test set OOD', test_metrics_2['loss'], test_metrics_2['acc'])
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
        
            train_loader = DataLoader(dataset, cfg.data.batch_size, shuffle=True, pin_memory=True)
            test_loader_ind = DataLoader(test_dataset_ind, cfg.data.batch_size, shuffle=False, pin_memory=True)
            test_loader_ood = DataLoader(test_dataset_ood, cfg.data.batch_size, shuffle=False, pin_memory=True)
            
            model = instantiate(cfg.model)
            model = model.to(cfg.model.device)
            model.logger = logger
            # print(summary(model, (10, 10, 3), device=cfg.model.device))
            model.fit(dataloader=train_loader, num_epochs=cfg.trainer.max_epochs)
            
            if cfg.save:
                torch.save(model.state_dict(), f'checkpoints/{cfg.run_name}_{timestamp}.pt')
            
            test_metrics_1 = model.test('id', test_loader_ind)
            test_metrics_2 = model.test('ood', test_loader_ood)
            
            table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
            table.add_data('Test set ID', test_metrics_1['loss'], test_metrics_1['acc'])
            table.add_data('Test set OOD', test_metrics_2['loss'], test_metrics_2['acc'])
            logger.experiment.log({'Test Sets Table': table})
            logger.experiment.finish()
            
            del model
            del logger
            del train_loader
            torch.cuda.empty_cache() 
            gc.collect()

if __name__ == '__main__':
    main()