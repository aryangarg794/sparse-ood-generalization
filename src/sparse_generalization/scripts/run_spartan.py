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
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from torch.utils.data import DataLoader
from torchsummary import summary

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings(
    "ignore", ".*can be accelerated via the 'torch-scatter' package*"
)

print(f"CUDA available: {torch.cuda.is_available()}")


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    """Main function to run the training loop and stuff for
    box world example.
    """
    timestamp = datetime.now().strftime("%d_%b_%Y__%Hh%Mm")
    group_name = cfg.run_name + "_" + timestamp

    dataset, val_sets, test_sets = instantiate(cfg.data.data_func)()

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
        print(summary(model, (10, 10, 3), device="cuda"))
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        results = {}
        for seed in cfg.seeds:
            results[seed] = {}

            print(f"\n{'='*60}")
            print(f"Running Seed {seed} for group {group_name}")
            print(f"\n{'='*60}")
            config_dict = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            wandb_dict = OmegaConf.to_container(
                cfg.wandb, resolve=True, throw_on_missing=True
            )
            name = cfg.run_name + f"_seed_{seed}_{timestamp}"
            logger = WandbLogger(
                **wandb_dict, name=name, config=config_dict, group=group_name
            )

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            generator = torch.Generator().manual_seed(seed)

            train_loader = DataLoader(
                dataset, cfg.data.batch_size, shuffle=True, generator=generator
            )
            val_loaders = []
            for val_dataset in val_sets:
                val_loaders.append(DataLoader(val_dataset, 512))

            test_loaders = []
            for test_dataset in test_sets:
                test_loaders.append(DataLoader(test_dataset, 1024))

            model = instantiate(cfg.model)(val_to_name=cfg.data.val_to_name)
            model = model.to(cfg.model.device)
            model.logger = logger
            # print(ModelSummary(model, max_depth=-1))
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
                    {"model": model.state_dict(), "hparams": cfg.model},
                    f"checkpoints/{cfg.run_name}_{timestamp}.pt",
                )

            results[seed]["train_loss"] = losses
            results[seed]["train_acc"] = accs
            results[seed]["train_sparse"] = sparses
            results[seed]["train_masks"] = mask_edges
            results[seed]["train_attns"] = attn_edges

            results[seed]["val_losses"] = losses_test
            results[seed]["val_accs"] = accs_test
            results[seed]["val_attns"] = attn_test
            results[seed]["val_masks"] = masks_test

            for i, name in enumerate(cfg.data.val_to_name.values()):
                model.test_name = name
                test_metrics = model.test(name, test_loaders[i])
                results[seed][f"test_{name}"] = test_metrics

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

        with open(f"results/{cfg.run_name}.pl", "wb") as file:
            dill.dump(results, file)
            file.close()


if __name__ == "__main__":
    main()
