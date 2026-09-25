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

from sparse_generalization.utils.dataloading import get_shapes_datasets
from sparse_generalization.models.transformer import TransformerLit
from sparse_generalization.utils.datasets import BasicDataset
from sparse_generalization.utils.parallel import (
    captured_output,
    lightning_progress_callbacks,
    run_seeds,
    send_to_console,
    set_current_seed,
    worker_slot,
)

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings(
    "ignore", ".*can be accelerated via the 'torch-scatter' package*"
)

if __name__ == "__main__":  # spawned workers import this module as __mp_main__
    print(f"CUDA available: {torch.cuda.is_available()}")


def run_seed(config_dict: dict, seed: int, timestamp: str, group_name: str, data: tuple):
    cfg = OmegaConf.create(config_dict)
    dataset, val_sets, test_sets, anti_dataset = data
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_results = {}
    set_current_seed(seed)
    if worker_slot() is None:
        print(f"\n{'='*60}")
        print(f"Running Seed {seed} for group {group_name}")
        print(f"\n{'='*60}")
    wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
    name = cfg.run_name + f"_seed_{seed}_{timestamp}"
    logger = WandbLogger(**wandb_dict, name=name, config=config_dict, group=group_name)
    with captured_output():
        _ = logger.experiment  # start wandb now so its banner isn't printed over the progress bars

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(dataset, cfg.data.batch_size, shuffle=True, generator=generator)

    val_loaders = []
    for val_dataset in val_sets:
        val_loaders.append(DataLoader(val_dataset, 512))

    test_loaders = []
    for test_dataset in test_sets:
        test_loaders.append(DataLoader(test_dataset, 1024))

    anti_loader = DataLoader(anti_dataset, 1024)

    model = instantiate(cfg.model)(val_to_name=cfg.data.val_to_name)
    trainer = Trainer(**cfg.trainer, logger=logger, callbacks=lightning_progress_callbacks())
    model.num_train_batches = len(train_loader)
    model.num_val_batches = len(val_loaders[0])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
    if cfg.save:
        trainer.save_checkpoint(f"checkpoints/{cfg.run_name}_{timestamp}.ckpt")

    for i, name in enumerate(cfg.data.val_to_name.values()):
        model.test_name = name
        test_metrics = trainer.test(model, test_loaders[i], verbose=False)
        seed_results[f"test_{name}"] = test_metrics

    seed_results["train_loss"] = model.losses
    seed_results["train_acc"] = model.accs

    if isinstance(cfg.model, TransformerLit):
        seed_results["train_masks"] = model.masks
        seed_results["train_sparse"] = model.sparses
        seed_results["test_masks"] = model.masks_test

    seed_results["val_losses"] = model.losses_test
    seed_results["val_accs"] = model.accs_test

    seed_results["anti_results"] = model.test_anti(anti_loader)

    with captured_output() as wandb_summary:
        logger.experiment.finish()
    if wandb_summary is not None:
        send_to_console(f"{'='*60}\nSeed {seed} finished ({group_name})\n{wandb_summary.getvalue()}")

    del model
    del logger
    del train_loader
    torch.cuda.empty_cache()
    gc.collect()

    return seed_results


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    """Main function to run the training loop and stuff for
    box world example.
    """
    timestamp = datetime.now().strftime("%d_%b_%Y__%Hh%Mm")

    dataset, val_sets, test_sets, anti_dataset = instantiate(cfg.data.data_func)(
        compute_mask=False
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))
    group_name = cfg.run_name + "_" + timestamp

    if cfg.seeds is None:
        print(f"\n{'='*60}")
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_dict = OmegaConf.to_container(
            cfg.wandb, resolve=True, throw_on_missing=True
        )
        name = cfg.run_name + f"_seed_{timestamp}"
        logger = WandbLogger(
            **wandb_dict, name=name, config=config_dict, group=group_name
        )

        train_loader = DataLoader(dataset, cfg.data.batch_size, shuffle=True)

        test_loaders = []
        for test_dataset in test_sets:
            test_loaders.append(DataLoader(test_dataset, cfg.data.batch_size))

        model = instantiate(cfg.model)
        print(f"{'='*60}")
        print(model)
        trainer = Trainer(**cfg.trainer, logger=logger)
        trainer.fit(model, train_dataloaders=train_loader)

        if cfg.save:
            torch.save(
                model.model.state_dict(), f"checkpoints/{cfg.run_name}_{timestamp}.pt"
            )

        model.test_name = "In-Distribution"
        test_metrics_id = trainer.test(model, test_loaders[0], verbose=False)
        model.test_name = "OOD: A"
        test_metrics_col = trainer.test(model, test_loaders[1], verbose=False)
        model.test_name = "OOD: B"
        test_metrics_pair = trainer.test(model, test_loaders[2], verbose=False)

        # table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
        # table.add_data('Test set ID', test_metrics_1[0]['test_loss'], test_metrics_1[0]['test_acc'])
        # table.add_data('Test set OOD', test_metrics_2[0]['test_loss'], test_metrics_2[0]['test_acc'])
        # logger.experiment.log({'Test Sets Table': table})
        logger.experiment.finish()
    else:
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        data = (dataset, val_sets, test_sets, anti_dataset)
        seed_outputs = run_seeds(
            run_seed,
            [(config_dict, seed, timestamp, group_name, data) for seed in cfg.seeds],
            cfg.num_concurrent,
        )
        results = dict(zip(cfg.seeds, seed_outputs))

        with open(f"results/{cfg.run_name}.pl", "wb") as file:
            dill.dump(results, file)
            file.close()


if __name__ == "__main__":
    main()
