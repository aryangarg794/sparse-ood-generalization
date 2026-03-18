import dill
import hydra
import os
import numpy as np
import random
import torch
import warnings
import wandb
import gc

from datetime import datetime
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from sparse_generalization.models.transformer import TransformerLit
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
    file_names = {
        
    }
    data_path = cfg.data.data_dir
    X_train = torch.tensor(np.load(to_absolute_path(data_path + '/grid_tensors_train.npy')))
    Y_train = torch.tensor(np.load(to_absolute_path(data_path + '/grid_labels_train.npy')))
    X_test_ID = torch.tensor(np.load(to_absolute_path(data_path + '/grid_tensors_train.npy')))
    Y_test_ID = torch.tensor(np.load(to_absolute_path(data_path + '/grid_labels_train.npy')))
    X_test_A = torch.tensor(np.load(to_absolute_path(data_path + '/grid_tensors_train.npy')))
    Y_test_A = torch.tensor(np.load(to_absolute_path(data_path + '/grid_labels_train.npy')))
    X_test_B = torch.tensor(np.load(to_absolute_path(data_path + '/grid_tensors_train.npy')))
    Y_test_B = torch.tensor(np.load(to_absolute_path(data_path + '/grid_labels_train.npy')))
    group_name = cfg.run_name + "_" + timestamp
    dataset = BasicDataset(X_train, Y_train)
    test_dataset_id = BasicDataset(X_test_ID, Y_test_ID)
    test_dataset_a = BasicDataset(X_test_A, Y_test_A)
    test_dataset_b = BasicDataset(X_test_B, Y_test_B)
    
    print(OmegaConf.to_yaml(cfg, resolve=True))


    if cfg.seeds is None:
        print(f'\n{'='*60}')
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
        name = cfg.run_name + f'_seed_{timestamp}'
        logger = WandbLogger(**wandb_dict, name=name, config=config_dict, group=group_name) 
        
        train_loader = DataLoader(dataset, cfg.data.batch_size, shuffle=True)

        test_loaders = []
        for test_dataset in [test_dataset_id, test_dataset_a, test_dataset_b]:
            test_loaders.append(DataLoader(test_dataset, cfg.data.batch_size))
        
        model = instantiate(cfg.model)
        print(f"{'='*60}")
        print(model)
        trainer = Trainer(**cfg.trainer, logger=logger)
        trainer.fit(model, train_dataloaders=train_loader)
        
        if cfg.save:
            torch.save(model.model.state_dict(), f'checkpoints/{cfg.run_name}_{timestamp}.pt')
        
        model.test_name = 'In-Distribution'
        test_metrics_id = trainer.test(model, test_loaders[0], verbose=False)
        model.test_name = 'OOD: A'
        test_metrics_col = trainer.test(model, test_loaders[1], verbose=False)
        model.test_name = 'OOD: B'
        test_metrics_pair = trainer.test(model, test_loaders[2], verbose=False)

        
        # table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
        # table.add_data('Test set ID', test_metrics_1[0]['test_loss'], test_metrics_1[0]['test_acc'])
        # table.add_data('Test set OOD', test_metrics_2[0]['test_loss'], test_metrics_2[0]['test_acc'])
        # logger.experiment.log({'Test Sets Table': table})
        logger.experiment.finish()
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        results = {}

        for seed in cfg.seeds:
            results[seed] = {}
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
        
            train_loader = DataLoader(dataset, cfg.data.batch_size, shuffle=True, generator=generator)
                
            test_loaders = []
            for test_dataset in [test_dataset_id, test_dataset_a, test_dataset_b]:
                test_loaders.append(DataLoader(test_dataset, cfg.data.batch_size))
            
            model = instantiate(cfg.model)
            trainer = Trainer(**cfg.trainer, logger=logger)
            model.num_train_batches = len(train_loader)
    
            trainer.fit(model, train_dataloaders=train_loader)
            if cfg.save:
                torch.save(model.model.state_dict(), f'checkpoints/{cfg.run_name}_{timestamp}.pt')
        
            model.test_name = 'id'
            test_metrics_id = trainer.test(model, test_loaders[0], verbose=False)
            model.test_name = 'A'
            test_metrics_col = trainer.test(model, test_loaders[1], verbose=False)
            model.test_name = 'B'
            test_metrics_pair = trainer.test(model, test_loaders[2], verbose=False)
            
            results[seed]['train_loss'] = model.losses
            results[seed]['train_acc'] = model.accs
            
            if isinstance(cfg.model, TransformerLit):
                results[seed]['train_masks'] = model.masks
                results[seed]['train_sparse'] = model.sparses
                results[seed]['test_masks'] = model.masks_test
                
            results[seed]['test_id'] = test_metrics_id
            results[seed]['test_A'] = test_metrics_col
            results[seed]['test_B'] = test_metrics_pair
            
            # table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
            # table.add_data('Test set ID', test_metrics_1[0]['test_loss_id'], test_metrics_1[0]['test_acc_id'])
            # table.add_data('Test set OOD', test_metrics_2[0]['test_loss_ood'], test_metrics_2[0]['test_acc_ood'])
            # logger.experiment.log({'Test Sets Table': table})
            logger.experiment.finish()
            
            del model
            del logger
            del train_loader
            torch.cuda.empty_cache() 
            gc.collect()

        with open(f'results/{group_name}.pl', 'wb') as file:
            dill.dump(results, file)
            file.close()
if __name__ == '__main__':
    main()