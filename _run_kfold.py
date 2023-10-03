import os, json
import pandas as pd
import numpy as np
import pickle

import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from os import path
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset, TensorDataset, ConcatDataset
from torchmetrics.classification.accuracy import Accuracy

from pytorch_lightning import LightningDataModule, seed_everything, Trainer
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.demos.boring_classes import Net
from pytorch_lightning.demos.mnist_datamodule import MNIST
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn

from transformers import AutoTokenizer

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils.normalize_process import inverse_scaling, norm_shuffledSet, norm_shuffledSet_merge
from utils.utils import load_hparams, load_protdata, DictX, make_sequential
from modules.kfold_module import BaseKFoldDataModule, KfoldClearanceDataModule, KfoldClearanceModel
from modules.models import RegressionPredictModel, DTIPredictionModel
from utils.draw_plot import draw_plot



DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")

#############################################################################################
#                           KFold Loop / Cross Validation Example                           #
# This example demonstrates how to leverage Lightning Loop Customization introduced in v1.5 #
# Learn more about the loop structure from the documentation:                               #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html                  #
#############################################################################################


#############################################################################################
#                           Step 3 / 5: Implement the EnsembleVotingModel module            #
# The `EnsembleVotingModel` will take our custom LightningModule and                        #
# several checkpoint_paths.                                                                 #
#                                                                                           #
#############################################################################################


class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str], prot_embed, scaler) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.criterior = torch.nn.SmoothL1Loss()
        self.prot_embed = prot_embed
        self.scaler = scaler

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        logits = torch.stack([m(batch[0], batch[1], self.prot_embed) for m in self.models]).mean(0)
        loss = self.criterior(logits, batch[2])

        y_pred = inverse_scaling(logits, self.scaler, log_scale=False)
        y_label = inverse_scaling(batch[2], self.scaler, log_scale=False)

        self.log("test_loss", loss)
        self.log("test_MSE", mean_squared_error(y_label, y_pred))


#############################################################################################
#                           Step 4 / 5: Implement the  KFoldLoop                            #
# From Lightning v1.5, it is possible to implement your own loop. There is several steps    #
# to do so which are described in detail within the documentation                           #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html.                 #
# Here, we will implement an outer fit_loop. It means we will implement subclass the        #
# base Loop and wrap the current trainer `fit_loop`.                                        #
#############################################################################################


#############################################################################################
#                     Here is the `Pseudo Code` for the base Loop.                          #
# class Loop:                                                                               #
#                                                                                           #
#   def run(self, ...):                                                                     #
#       self.reset(...)                                                                     #
#       self.on_run_start(...)                                                              #
#                                                                                           #
#        while not self.done:                                                               #
#            self.on_advance_start(...)                                                     #
#            self.advance(...)                                                              #
#            self.on_advance_end(...)                                                       #
#                                                                                           #
#        return self.on_run_end(...)                                                        #
#############################################################################################


class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.

        # the test loop normally expects the model to be the pure LightningModule, but since we are running the
        # test loop during fitting, we need to temporarily unpack the wrapped module
        wrapped_model = self.trainer.strategy.model
        self.trainer.strategy.model = self.trainer.strategy.lightning_module
        self.trainer.test_loop.run()
        self.trainer.strategy.model = wrapped_model
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths,
                                                self.trainer.datamodule.prot_embeds["embeds"],
                                                self.trainer.datamodule.test_scaler)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)



#############################################################################################
#                           Step 5 / 5: Connect the KFoldLoop to the Trainer                #
# After creating the `KFoldDataModule` and our model, the `KFoldLoop` is being connected to #
# the Trainer.                                                                              #
# Finally, use `trainer.fit` to start the cross validation training.                        #
#############################################################################################

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    config = load_hparams('config/3_config_hparam_BA.json')
    config = DictX(config)

    seed_everything(seed=config.num_seed)

    train_path = os.path.join(config.data_folder, config.train_file)
    test_path = os.path.join(config.data_folder, config.test_file)
    additional_datafile = os.path.join(config.data_folder, config.additional_datafile)

    ## -- Datamodule setting and load training data -- ##
    drug_tokenizer = AutoTokenizer.from_pretrained(config.d_model_name)
    prot_tokenizer = AutoTokenizer.from_pretrained(config.p_model_name)

    load_model = f"{config.load_model}.ckpt"
    model = KfoldClearanceModel(load_model, config.lr, config.dropout, config.act_func, config.affinity_sorting, 
                                config.smiles_decoder_use, config.use_Fup)
    datamodule = KfoldClearanceDataModule(config.num_seed, drug_tokenizer, prot_tokenizer, config.drug_length, config.prot_length, 
                                train_path, test_path, additional_datafile, config.prot_path, 
                                config.batch_size, config.num_workers, 
                                config.train_seprate,config.valid_source, config.valid_rate, config.valid_shuffle,
                                config.use_additionalSet)
    trainer = Trainer(
        max_epochs=config.max_epoch,
        num_sanity_val_steps=0,
        devices=config.gpu_id,
        accelerator="gpu",
        strategy="ddp",
    )

    internal_fit_loop = trainer.fit_loop
    
    fold_export_path = "./fold_export/"
    if not os.path.exists(fold_export_path):
        os.makedirs(fold_export_path)

    trainer.fit_loop = KFoldLoop(5, export_path=fold_export_path)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)
