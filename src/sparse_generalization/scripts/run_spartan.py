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
from functools import partial
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchinfo import summary

from sparse_generalization.models.generative import FlowSpartan
from sparse_generalization.models.hypernet import HyperNetSpartan
from sparse_generalization.models.ensemble import Ensemble
from sparse_generalization.models.cond_spartan import ConditionalSPARTAN
from sparse_generalization.models.spartan import SPARTAN
from sparse_generalization.utils.parallel import (
    captured_output,
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


def run_seed(config_dict: dict, seed: int, timestamp: str, group_name: str, flow_model: bool, data: tuple):
    cfg = OmegaConf.create(config_dict)
    dataset, val_sets, test_sets, anti_dataset = data
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_results = {}
    gens = None

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
    model = model.to(model.device)
    model.logger = logger
    if flow_model:
        (
            losses,
            accs,
            sparses,
            gens,
            mask_edges,
            attn_edges,
            losses_test,
            accs_test,
            attn_test,
            masks_test,
        ) = model.fit(
            dataloader=train_loader,
            num_epochs=cfg.trainer.max_epochs,
            testloaders=val_loaders,
        )
    else:
        (
            losses,
            accs,
            sparses,
            mask_edges,
            attn_edges,
            losses_test,
            accs_test,
            attn_test,
            masks_test,
        ) = model.fit(
            dataloader=train_loader,
            num_epochs=cfg.trainer.max_epochs,
            testloaders=val_loaders,
        )

    if cfg.save:
        torch.save(
            {"model": model.state_dict(), "hparams": model.hyper_params},
            f"checkpoints/{cfg.run_name}_seed{seed}_{timestamp}.pt",
        )

    hparams = model.hyper_params
    seed_results["train_loss"] = losses
    seed_results["train_acc"] = accs
    seed_results["train_sparse"] = sparses
    seed_results["train_masks"] = mask_edges
    seed_results["train_attns"] = attn_edges
    seed_results["train_gen"] = gens

    seed_results["val_losses"] = losses_test
    seed_results["val_accs"] = accs_test
    seed_results["val_attns"] = attn_test
    seed_results["val_masks"] = masks_test

    for i, name in enumerate(cfg.data.val_to_name.values()):
        model.test_name = name
        test_metrics = model.test(name, test_loaders[i])
        seed_results[f"test_{name}"] = test_metrics

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

    return seed_results, hparams


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%d_%b_%Y__%Hh%Mm")
    group_name = cfg.run_name + "_" + timestamp

    test_model = instantiate(cfg.model)(val_to_name=cfg.data.val_to_name)
    flow_model = isinstance(test_model, FlowSpartan) or isinstance(test_model, HyperNetSpartan)
    spartan_model = isinstance(test_model, SPARTAN)
    ensemble_model = isinstance(test_model, Ensemble)
    cond_model = isinstance(test_model, ConditionalSPARTAN)
    del test_model
    dataset, val_sets, test_sets, anti_dataset = instantiate(cfg.data.data_func)(
        compute_mask=cfg.model.compute_mask if spartan_model else False
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))

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
        val_loaders = []
        for dataset in val_sets:
            val_loaders.append(DataLoader(dataset, cfg.data.batch_size))

        test_loaders = []
        for dataset in test_sets:
            test_loaders.append(DataLoader(dataset, cfg.data.batch_size))

        model = instantiate(cfg.model)
        model.logger = logger
        print(summary(model, (10, 10, 3), device=model.device))
        model.fit(dataloader=train_loader, num_epochs=cfg.trainer.max_epochs)

        if cfg.save:
            torch.save(model.state_dict(), f"checkpoints/{cfg.run_name}_{timestamp}.pt")

        test_metrics_id = model.test("id", test_loaders[0])
        test_metrics_col = model.test("col", test_loaders[1])
        test_metrics_pair = model.test("pair", test_loaders[2])
        test_metrics_dist = model.test("dist", test_loaders[3])
        test_metrics_comb = model.test("comb", test_loaders[4])

        # table = wandb.Table(columns=['Dataset', 'Loss', 'Acc'])
        # table.add_data('Test set ID', test_metrics_1['loss'], test_metrics_1['acc'])
        # table.add_data('Test set OOD', test_metrics_2['loss'], test_metrics_2['acc'])
        # logger.experiment.log({'Test Sets Table': table})
        logger.experiment.finish()
    else:
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        data = (dataset, val_sets, test_sets, anti_dataset)
        seed_outputs = run_seeds(
            run_seed,
            [(config_dict, seed, timestamp, group_name, flow_model, data) for seed in cfg.seeds],
            cfg.num_concurrent,
        )

        results = {}
        for seed, (seed_results, hparams) in zip(cfg.seeds, seed_outputs):
            results[seed] = seed_results
            results["hparams"] = hparams

        with open(f"results/{cfg.run_name}.pl", "wb") as file:
            dill.dump(results, file)
            file.close()


if __name__ == "__main__":
    main()
