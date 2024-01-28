import os
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset, TensorDataset

from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.module import LightningModule

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils.normalize_process import inverse_scaling, norm_shuffledSet
from modules.LayerModule import make_sequential
from utils.preprocess import load_protdata

from modules.models import RegressionPredictModel, DTIPredictionModel


#############################################################################################
#                           Step 1 / 5: Define KFold DataModule API                         #
# Our KFold DataModule requires to implement the `setup_folds` and `setup_fold_index`       #
# methods.                                                                                  #
#############################################################################################


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


#############################################################################################
#                           Step 2 / 5: Implement the KFoldDataModule                       #
# The `KFoldDataModule` will take a train and test dataset.                                 #
# On `setup_folds`, folds will be created depending on the provided argument `num_folds`    #
# Our `setup_fold_index`, the provided train dataset will be split accordingly to        #
# the current fold split.                                                                   #
#############################################################################################

@dataclass
class KfoldClearanceDataModule(BaseKFoldDataModule):
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None
    
    def __init__(self, num_seed: int, drug_tokenizer, prot_tokenizer, drug_length: int, prot_length: int, 
                 train_path: str, test_path: str, additional_datafile:str, prot_path: str, 
                 batch_size: int, num_workers: int, 
                 train_seprate: float,valid_source:str, valid_rate:float, valid_shuffle:bool,
                 use_additionalSet:bool = False):
        super().__init__()
        self.num_seed = num_seed

        self.drug_tokenizer = drug_tokenizer
        self.prot_tokenizer = prot_tokenizer

        self.drug_length = drug_length
        self.prot_length = prot_length

        self.train_path = train_path
        self.test_path = test_path
        self.prot_path = prot_path
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_seprate = train_seprate
        self.valid_source = valid_source
        self.valid_rate = valid_rate
        self.valid_shuffle = valid_shuffle

        self.use_additionalSet = use_additionalSet
        self.additional_datafile = additional_datafile

        self.df_prepared = False
    
    def prepare_data(self) -> None:
        # download the data.
        if self.df_prepared is False:
            ## -- load train dataset -- ##
            train_load = pd.read_csv(self.train_path)

            if self.use_additionalSet:
                train_load_additional = pd.read_csv(self.additional_datafile)

                train_load = train_load.loc[:, [col for col in train_load.columns if col == 'Clint' or col == 'SMILES']]
                train_load = pd.concat([train_load, train_load_additional])

            else:
                train_load = train_load.loc[:, [col for col in train_load.columns if col == 'Clint' or col == 'SMILES' or col == 'Fup']]
                    
            train_load.drop_duplicates(['SMILES'], keep='last', inplace=True)
            train_load.reset_index(drop=True, inplace=True)

            test_load = pd.read_csv(self.test_path)
            test_load = test_load.loc[:, [col for col in test_load.columns if col == 'Clint' or col == 'SMILES' or col == 'Fup']]

            if self.valid_source == "train":
                train_norm, train_scaler, test_norm, test_scaler = norm_shuffledSet(train_load, test_load, self.num_seed)
            else:
                train_norm, train_scaler, test_norm, test_scaler = norm_shuffledSet(train_load, test_load, self.num_seed, self.valid_rate)

            train_datas = np.array(train_norm['SMILES']).tolist()
            test_datas = np.array(test_norm['SMILES']).tolist()

            self.train_scaler = train_scaler
            self.test_scaler = test_scaler

            ## -- load Protein dataset -- ##
            prot_datas = load_protdata(self.prot_path, True)
            
            print("\n\nPrepare_data_start!!\n\n")

            train_obj = self.drug_tokenizer(train_datas, padding='max_length', 
                                        max_length=self.drug_length, truncation=True, return_tensors="pt")
            test_obj = self.drug_tokenizer(test_datas, padding='max_length', 
                                        max_length=self.drug_length, truncation=True, return_tensors="pt")
            
            ## -- prot dataset tokenization -- ##
            self.prot_obj = self.prot_tokenizer(np.array(prot_datas['aas']).tolist(), padding='max_length', 
                                        max_length=self.prot_length, truncation=True, return_tensors="pt")
            
            train_labels = torch.from_numpy(np.array(train_norm['Clint']))
            test_labels = torch.from_numpy(np.array(test_norm['Clint']))

            train_Fup = torch.from_numpy(np.array(train_norm['Fup']))
            test_Fup = torch.from_numpy(np.array(test_norm['Fup']))

            self.train_set = TensorDataset(train_obj['input_ids'], train_obj['attention_mask'], train_Fup, train_labels)
            self.test_set = TensorDataset(test_obj['input_ids'], test_obj['attention_mask'], test_Fup, test_labels)

            self.df_prepared = True

    def setup(self, stage: str) -> None:
        # load the data
        self.train_dataset = self.train_set
        self.test_dataset = Subset(self.test_set, np.arange(len(self.test_set)))

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    


class KfoldClearanceModel(LightningModule):
    def __init__(self, model_file, lr, dropout, act_func: list, affinity_sorting: bool = False, smiles_decoder_use:bool = False, use_Fup:bool = False):
        super().__init__()
        
        self.save_hyperparameters()
        self.lr = lr
        self.criterior = torch.nn.SmoothL1Loss()
        
        self.affinity_sorting = affinity_sorting
        self.smiles_decoder_use = smiles_decoder_use
        self.use_Fup = use_Fup

        ## -- Biomarker model selection code  -- ##
        """ DTI : interaction train
            Regression : binding affinity train 
        """      
        ckpt_file = os.path.join("./checkpoint", model_file)
        if model_file == "biomarker.ckpt":
            self.model = DTIPredictionModel.load_from_checkpoint(ckpt_file)
        else:
            self.model = RegressionPredictModel.load_from_checkpoint(ckpt_file)

        hook_layer = 3
        self.model.model_freeze(hook_layer)    

        ## -- Output decoder setting -- ##
        smiles_dim  = self.model.d_model.config.hidden_size
        smiles_outputdim = 512
        self.smiles_decoder = nn.Sequential(nn.Linear(smiles_dim, 1024),
                                           nn.ReLU(),
                                           nn.Dropout(dropout),
                                           nn.Linear(1024, smiles_outputdim),
                                           nn.ReLU(),
                                           nn.Dropout(dropout))
        
        ## -- Fup MLP Setting -- ##
        self.Fup_MLP = nn.Sequential(nn.Linear(1, 128),
                                                nn.ReLU(),
                                                nn.Dropout(dropout),
                                                nn.Linear(128, 256),
                                                nn.ReLU(),
                                                nn.Dropout(dropout),
                                                nn.Linear(256, smiles_outputdim))


        ## -- clearance decoder setting  -- ##
        prot_embedlen = 20 if affinity_sorting else 23
        aff_dim = self.model.decoder[hook_layer].out_features * prot_embedlen

        if not self.smiles_decoder_use:
            decoder_dim = smiles_dim + aff_dim + smiles_outputdim if use_Fup else smiles_dim + aff_dim
        else:
            decoder_dim = 2 * smiles_outputdim + aff_dim if use_Fup else smiles_outputdim + aff_dim

        self.output_decoder1 = make_sequential(decoder_dim, 512, act_func[0])
        self.output_decoder2 = make_sequential(512, 256, act_func[1])
        self.output_decoder3 = make_sequential(256, 32, act_func[2])

        self.Linear = nn.Linear(32,1)

    def forward(self, smiles_input, prot_input, Fup_input):
        smiles_logits, affinity_logits, outputs = self.model(smiles_input, prot_input)

        if self.affinity_sorting:
            chunk_size = [9,10,4]
            chunk_init = 0
            sorted_result = list()

            for chunk_len in chunk_size:
                chunk_end = chunk_init + chunk_len

                CYP_sort = [output.squeeze()[chunk_init:chunk_end].detach().tolist() for output in outputs]
                affinity_logits_sort = [output[chunk_init:chunk_end].detach().tolist() for output in affinity_logits]

                for idx, _ in enumerate(CYP_sort):
                    min_idx = CYP_sort[idx].index(min(CYP_sort[idx]))
                    affinity_logits_sort[idx].pop(min_idx)

                sorted_result.append(torch.tensor(affinity_logits_sort).to(self._device))
                chunk_init = chunk_end

            affinity_embed = torch.cat((sorted_result[0], sorted_result[1], sorted_result[2]), dim=1)
            affinity_embed = affinity_embed.view([affinity_embed.shape[0], -1])

        else:
            affinity_embed = torch.stack(affinity_logits)
            affinity_embed = affinity_embed.view([affinity_embed.shape[0], -1])

        ## --  smiles decoder use -- ##
        if not self.smiles_decoder_use:
            outputs = torch.cat((smiles_logits, affinity_embed), dim=1)
        else:
            smiles_output = self.smiles_decoder(smiles_logits)
            outputs = torch.cat((smiles_output, affinity_embed), dim=1)

        if self.use_Fup:
            Fup_input = Fup_input.unsqueeze(dim=1)
            Fup_Embed = self.Fup_MLP(Fup_input)
            outputs = torch.cat((outputs, Fup_Embed), dim=1)

        outputs = self.output_decoder1(outputs)
        outputs = self.output_decoder2(outputs)
        outputs = self.output_decoder3(outputs)
        outputs = self.Linear(outputs)

        return outputs

    def training_step(self, batch):
        smiles_input = {"input_ids": batch[0], "attention_mask": batch[1]}
        Fup_input = batch[2].type(torch.FloatTensor).to(self._device)
        labels = batch[3].type(torch.FloatTensor).to(self._device)
        # fup_labels = batch[3].type(torch.FloatTensor).to(self._device)
        
        prot_input = self.trainer.datamodule.prot_obj.to(self._device)
        outputs = self(smiles_input, prot_input, Fup_input)
        outputs = outputs.squeeze(dim=1)

        loss = self.criterior(outputs, labels)
        # fup_loss = self.criterior(outputs[1], fup_labels)

        self.log("train_loss", loss)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        smiles_input = {"input_ids": batch[0], "attention_mask": batch[1]}
        Fup_input = batch[2].type(torch.FloatTensor).to(self._device)
        labels = batch[3].type(torch.FloatTensor).to(self._device)

        prot_input = self.trainer.datamodule.prot_obj.to(self._device)
        outputs = self(smiles_input, prot_input, Fup_input)
        outputs = outputs.squeeze(dim=1)

        loss = self.criterior(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)

        return {"outputs": outputs, "labels": labels}

    def validation_step_end(self, outputs):
        return {"outputs": outputs["outputs"], "labels": outputs["labels"]}

    def validation_epoch_end(self, outputs):
        preds = torch.as_tensor(torch.cat([output['outputs'] for output in outputs], dim=0))
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0))

        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()

        valid_scaler = self.trainer.datamodule.train_scaler

        y_pred = inverse_scaling(y_pred, valid_scaler, log_scale=False)
        y_label = inverse_scaling(y_label, valid_scaler, log_scale=False)
        
        MSE_score = mean_squared_error(y_label, y_pred)
        MAE_score = mean_absolute_error(y_label, y_pred)
        R2_score = r2_score(y_label, y_pred)

        self.log("valid_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_R2", R2_score, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        smiles_input = {"input_ids": batch[0], "attention_mask": batch[1]}
        Fup_input = batch[2].type(torch.FloatTensor).to(self._device)
        labels = batch[3].type(torch.FloatTensor).to(self._device)

        prot_input = self.trainer.datamodule.prot_obj.to(self._device)
        outputs = self(smiles_input, prot_input, Fup_input)
        outputs = outputs.squeeze(dim=1)

        loss = self.criterior(outputs, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"outputs": outputs,  "labels": labels}

    def test_step_end(self, outputs):
        return {"outputs": outputs["outputs"], "labels": outputs["labels"]}

    def test_epoch_end(self, outputs):
        preds = torch.as_tensor(torch.cat([output['outputs'] for output in outputs], dim=0))
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0))

        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()
        
        test_scaler = self.trainer.datamodule.test_scaler

        y_pred = inverse_scaling(y_pred, test_scaler, log_scale=False)
        y_label = inverse_scaling(y_label, test_scaler, log_scale=False)

        MSE_score = mean_squared_error(y_label, y_pred)
        MAE_score = mean_absolute_error(y_label, y_pred)
        R2_score = r2_score(y_label, y_pred)

        self.test_result = {'preds': y_pred, 'labels': y_label}
        self.test_log = {'rmse': MSE_score, 'r2': R2_score}

        self.log("test_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_R2", R2_score, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        # param_optimizer = list(self.named_parameters())
        # no_decay = ["bias", "gamma", "beta"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #         "weight_decay_rate": 0.01
        #     },
        #     {
        #         "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        #         "weight_decay_rate": 0.0
        #     },
        # ]
        # optimizer = torch.optim.AdamW(
        #     optimizer_grouped_parameters,
        #     lr=self.lr,
        # )

        optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer