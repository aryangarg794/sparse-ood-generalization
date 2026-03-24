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
    data_path = os.path.join(cfg.data.data_dir, f"boxworld_v2_train_{cfg.data.size}_pairs{cfg.data.num_pairs}.pl")
    data_path = to_absolute_path(data_path)
    
    with open(data_path, 'rb') as file:
        train_data = dill.load(file)
        file.close()

    group_name = cfg.run_name + "_" + timestamp
    dataset = BasicDataset(train_data['X_train'], train_data['Y_train'])

    test_id_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_test_id_pairs{cfg.data.num_pairs}.pl"))
    with open(test_id_path, 'rb') as file:
        test_id = dill.load(file)
        file.close()

    val_id_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_val_id_pairs{cfg.data.num_pairs}.pl"))
    with open(val_id_path, 'rb') as file:
        val_id = dill.load(file)
        file.close()

    test_col_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_test_col_pairs{cfg.data.num_pairs}.pl"))
    with open(test_col_path, 'rb') as file:
        test_col = dill.load(file)
        file.close()

    val_col_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_val_col_pairs{cfg.data.num_pairs}.pl"))
    with open(val_col_path, 'rb') as file:
        val_col = dill.load(file)
        file.close()

    test_pair_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_test_pair_pairs{cfg.data.num_pairs}.pl"))
    with open(test_pair_path, 'rb') as file:
        test_pair = dill.load(file)
        file.close()

    val_pair_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_val_pair_pairs{cfg.data.num_pairs}.pl"))
    with open(val_pair_path, 'rb') as file:
        val_pair = dill.load(file)
        file.close()

    test_dist_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_test_dist_pairs{cfg.data.num_pairs}.pl"))
    with open(test_dist_path, 'rb') as file:
        test_dist = dill.load(file)
        file.close()

    val_dist_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_val_dist_pairs{cfg.data.num_pairs}.pl"))
    with open(val_dist_path, 'rb') as file:
        val_dist = dill.load(file)
        file.close()

    test_comb_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_test_comb_pairs{cfg.data.num_pairs}.pl"))
    with open(test_comb_path, 'rb') as file:
        test_comb = dill.load(file)
        file.close()

    val_comb_path = to_absolute_path(os.path.join(cfg.data.data_dir, f"boxworld_v2_val_comb_pairs{cfg.data.num_pairs}.pl"))
    with open(val_comb_path, 'rb') as file:
        val_comb = dill.load(file)
        file.close()

    val_dataset_id = BasicDataset(val_id['X_train'], val_id['Y_train'])
    test_dataset_id = BasicDataset(test_id['X_train'], test_id['Y_train'])
    val_dataset_col = BasicDataset(val_col['X_col'], val_col['Y_col'])
    test_dataset_col = BasicDataset(test_col['X_col'], test_col['Y_col'])
    val_dataset_pair = BasicDataset(val_pair['X_pair'], val_pair['Y_pair'])
    test_dataset_pair = BasicDataset(test_pair['X_pair'], test_pair['Y_pair'])
    val_dataset_dist = BasicDataset(val_dist['X_dist'], val_dist['Y_dist'])
    test_dataset_dist = BasicDataset(test_dist['X_dist'], test_dist['Y_dist'])
    val_dataset_comb = BasicDataset(val_comb['X_comb'], val_comb['Y_comb'])
    test_dataset_comb = BasicDataset(test_comb['X_comb'], test_comb['Y_comb'])
    
    
    print(OmegaConf.to_yaml(cfg, resolve=True))


    if cfg.seeds is None:
        print(f'\n{'='*60}')
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
        name = cfg.run_name + f'_seed_{timestamp}'
        logger = WandbLogger(**wandb_dict, name=name, config=config_dict, group=group_name) 
        
        train_loader = DataLoader(dataset, cfg.data.batch_size, shuffle=True)
        val_loaders = []
        for val_dataset in [val_dataset_id, val_dataset_col, val_dataset_pair, val_dataset_dist, val_dataset_comb]:
            val_loaders.append(DataLoader(val_dataset, 256))
            
        test_loaders = []
        for test_dataset in [test_dataset_id, test_dataset_col, test_dataset_pair, test_dataset_dist, test_dataset_comb]:
            test_loaders.append(DataLoader(test_dataset, 512))
        
        model = instantiate(cfg.model)
        print(f"{'='*60}")
        print(model)
        trainer = Trainer(**cfg.trainer, logger=logger)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
        
        if cfg.save:
            torch.save(model.model.state_dict(), f'checkpoints/{cfg.run_name}_{timestamp}.pt')
        
        model.test_name = 'In-Distribution'
        test_metrics_id = trainer.test(model, test_loaders[0], verbose=False)
        model.test_name = 'OOD: Colors'
        test_metrics_col = trainer.test(model, test_loaders[1], verbose=False)
        model.test_name = 'OOD: Num pairs'
        test_metrics_pair = trainer.test(model, test_loaders[2], verbose=False)
        model.test_name = 'OOD: Distractors'
        test_metrics_dist = trainer.test(model, test_loaders[3], verbose=False)
        model.test_name = 'OOD: Combined'
        test_metrics_comb = trainer.test(model, test_loaders[4], verbose=False)
        
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
            val_loaders = []
            for val_dataset in [val_dataset_id, val_dataset_col, val_dataset_pair, val_dataset_dist, val_dataset_comb]:
                val_loaders.append(DataLoader(val_dataset, 256))
                
            test_loaders = []
            for test_dataset in [test_dataset_id, test_dataset_col, test_dataset_pair, test_dataset_dist, test_dataset_comb]:
                test_loaders.append(DataLoader(test_dataset, 512))
            
            model = instantiate(cfg.model)
            trainer = Trainer(**cfg.trainer, logger=logger)
            model.num_train_batches = len(train_loader)
            model.num_val_batches = len(val_loaders[0])
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
            if cfg.save:
                torch.save(model.model.state_dict(), f'checkpoints/{cfg.run_name}_{timestamp}.pt')
        
            model.test_name = 'id'
            test_metrics_id = trainer.test(model, test_loaders[0], verbose=False)
            model.test_name = 'col'
            test_metrics_col = trainer.test(model, test_loaders[1], verbose=False)
            model.test_name = 'pair'
            test_metrics_pair = trainer.test(model, test_loaders[2], verbose=False)
            model.test_name = 'dist'
            test_metrics_dist = trainer.test(model, test_loaders[3], verbose=False)
            model.test_name = 'comb'
            test_metrics_comb = trainer.test(model, test_loaders[4], verbose=False)
            
            results[seed]['train_loss'] = model.losses
            results[seed]['train_acc'] = model.accs
            
            if isinstance(cfg.model, TransformerLit):
                results[seed]['train_masks'] = model.masks
                results[seed]['train_sparse'] = model.sparses
                results[seed]['test_masks'] = model.masks_test
                
            results[seed]['val_losses'] = model.losses_test
            results[seed]['val_accs'] = model.accs_test
            
            results[seed]['test_id'] = test_metrics_id
            results[seed]['test_col'] = test_metrics_col
            results[seed]['test_pair'] = test_metrics_pair
            results[seed]['test_dist'] = test_metrics_dist
            results[seed]['test_comb'] = test_metrics_comb
            
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