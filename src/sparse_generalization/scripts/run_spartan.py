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
    val_dataset_id = BasicDataset(data['X_val_id'], data['Y_val_id'])
    test_dataset_id = BasicDataset(data['X_test_id'], data['Y_test_id'])
    val_dataset_col = BasicDataset(data['X_val_col'], data['Y_val_col'])
    test_dataset_col = BasicDataset(data['X_test_col'], data['Y_test_col'])
    val_dataset_pair = BasicDataset(data['X_val_pair'], data['Y_val_pair'])
    test_dataset_pair = BasicDataset(data['X_test_pair'], data['Y_test_pair'])
    val_dataset_dist = BasicDataset(data['X_val_dist'], data['Y_val_dist'])
    test_dataset_dist = BasicDataset(data['X_test_dist'], data['Y_test_dist'])
    val_dataset_comb = BasicDataset(data['X_val_comb'], data['Y_val_comb'])
    test_dataset_comb = BasicDataset(data['X_test_comb'], data['Y_test_comb'])
    
    print(OmegaConf.to_yaml(cfg, resolve=True))


    if cfg.seeds is None:
        print(f'\n{'='*60}')
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
        name = cfg.run_name + f'_seed_{timestamp}'
        logger = WandbLogger(**wandb_dict, name=name, config=config_dict, group=group_name) 
        
        train_loader = DataLoader(dataset, cfg.data.batch_size, shuffle=True)
        val_loaders = []
        for dataset in [val_dataset_id, val_dataset_col, val_dataset_pair, val_dataset_dist, val_dataset_comb]:
            val_loaders.append(DataLoader(dataset, cfg.data.batch_size))
            
        test_loaders = []
        for dataset in [test_dataset_id, test_dataset_col, test_dataset_pair, test_dataset_dist, test_dataset_comb]:
            test_loaders.append(DataLoader(dataset, cfg.data.batch_size))
        
        model = instantiate(cfg.model)
        model.logger = logger
        print(summary(model, (10, 10, 3), device='cuda'))
        model.fit(dataloader=train_loader, num_epochs=cfg.trainer.max_epochs)
        
        if cfg.save:
            torch.save(model.state_dict(), f'checkpoints/{cfg.run_name}_{timestamp}.pt')
        
        test_metrics_id = model.test('id', test_loaders[0])
        test_metrics_col = model.test('col', test_loaders[1])
        test_metrics_pair = model.test('pair', test_loaders[2])
        test_metrics_dist = model.test('dist', test_loaders[3])
        test_metrics_comb = model.test('comb', test_loaders[4])
        
        # table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
        # table.add_data('Test set ID', test_metrics_1['loss'], test_metrics_1['acc'])
        # table.add_data('Test set OOD', test_metrics_2['loss'], test_metrics_2['acc'])
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
            val_loaders = []
            for val_dataset in [val_dataset_id, val_dataset_col, val_dataset_pair, val_dataset_dist, val_dataset_comb]:
                val_loaders.append(DataLoader(val_dataset, cfg.data.batch_size))
                
            test_loaders = []
            for test_dataset in [test_dataset_id, test_dataset_col, test_dataset_pair, test_dataset_dist, test_dataset_comb]:
                test_loaders.append(DataLoader(test_dataset, cfg.data.batch_size))
            
            model = instantiate(cfg.model)
            model = model.to(cfg.model.device)
            model.logger = logger
            # print(summary(model, (10, 10, 3), device=cfg.model.device))
            losses, accs, sparses, mask_edges, attn_edges, losses_test, accs_test, attn_test, masks_test = model.fit(
                dataloader=train_loader, num_epochs=cfg.trainer.max_epochs,
                testloaders=val_loaders)
            
            test_metrics_id = model.test('id', test_loaders[0])
            test_metrics_col = model.test('col', test_loaders[1])
            test_metrics_pair = model.test('pair', test_loaders[2])
            test_metrics_dist = model.test('dist', test_loaders[3])
            test_metrics_comb = model.test('comb', test_loaders[4])
            
            if cfg.save:
                torch.save(model.state_dict(), f'checkpoints/{cfg.run_name}_{timestamp}.pt')
            
            results[seed]['train_loss'] = losses
            results[seed]['train_acc'] = accs
            results[seed]['train_sparse'] = sparses
            results[seed]['train_masks'] = mask_edges
            results[seed]['train_attns'] = attn_edges
                
            results[seed]['test_losses'] = losses_test
            results[seed]['test_accs'] = accs_test
            results[seed]['test_attns'] = attn_test
            results[seed]['test_masks'] = masks_test
            
            results[seed]['test_id'] = test_metrics_id
            results[seed]['test_col'] = test_metrics_col
            results[seed]['test_pair'] = test_metrics_pair
            results[seed]['test_dist'] = test_metrics_dist
            results[seed]['test_comb'] = test_metrics_comb
            
            # table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
            # table.add_data('Test set ID', test_metrics_1['loss'], test_metrics_1['acc'])
            # table.add_data('Test set OOD', test_metrics_2['loss'], test_metrics_2['acc'])
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