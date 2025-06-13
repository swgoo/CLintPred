import os
import pandas as pd
import numpy as np

from typing import Optional
from itertools import cycle, product

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from transformers import AutoTokenizer

from utils.normalize_process import normalization, inverse_scaling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from dataclasses import dataclass


class BiomakerDataset(Dataset):
    def __init__(self, chem_obj, prot_obj):
        self.pair_ids, self.pair_mask = self.make_pairset(chem_obj, prot_obj)

    def make_pairset(self, chem_obj, prot_obj):
        chem_ids = list(range(0, len(chem_obj["input_ids"])))
        marked_chem_obj = list(zip(chem_ids, chem_obj["input_ids"]))

        pair_ids = list(product(*[marked_chem_obj, prot_obj["input_ids"]]))
        pair_mask = list(product(*[chem_obj["attention_mask"], prot_obj["attention_mask"]]))

        return pair_ids, pair_mask
        
    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, idx):
        ids = self.pair_ids[idx][0][0]
        chem_obj = {"input_ids" : self.pair_ids[idx][0][1], "attention_mask": self.pair_mask[idx][0]}
        prot_obj = {"input_ids" : self.pair_ids[idx][1], "attention_mask": self.pair_mask[idx][1]}

        return ids, chem_obj, prot_obj


class BiomakerDataModule(pl.LightningDataModule):
    def __init__(self, chem_obj:pd.DataFrame, prot_obj:pd.DataFrame,
                 config):
        super().__init__()
        self.num_seed = config.num_seed

        self.chem_obj = chem_obj
        self.prot_obj = prot_obj

        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.df_prepared = False
    
    def setup(self, stage: str) -> None:
        # load the data
        self.predict_dataset = BiomakerDataset(self.chem_obj, self.prot_obj)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    


@dataclass
class clearanceDataset(Dataset):
    def __init__(self, data, labels = None, features = None):
        self.data = data
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        data = {"input_ids": self.data["input_ids"][idx], 
                "attention_mask": self.data["attention_mask"][idx]}
        
        if self.features is not None and self.labels is not None:
            labels = self.labels[idx]
            features = self.features[idx]
            return data, labels, features
        elif self.features is not None:
            features = self.features[idx]
            return data, features
        elif self.labels is not None:
            labels = self.labels[idx]
            return data, labels
        else:
            return data
        

class clearanceDatamodule(pl.LightningDataModule):
    def __init__(self, chem_tokenizer, config,
                 df_trainData=None, df_testData=None, df_predictData=None,
                 df_trainLabel=None, df_testLabel=None, df_predictLabel=None,
                 train_scaler=None, test_scaler=None, predict_scaler=None,
                 df_trainfeatures=None, df_testfeatures=None, df_predictfeatures=None):
        super().__init__()
        self.chem_tok = chem_tokenizer
        
        self.max_length = config.chem_max
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.df_trainData, self.df_testData, self.df_predictData = df_trainData, df_testData, df_predictData
        self.df_trainLabel, self.df_testLabel, self.df_predictLabel = df_trainLabel, df_testLabel, df_predictLabel
        self.train_scaler, self.test_scaler, self.predict_scaler = train_scaler, test_scaler, predict_scaler
        
        self.df_trainfeatures, self.df_testfeatures, self.df_predictfeatures = df_trainfeatures, df_testfeatures, df_predictfeatures
        
        self.is_load_file = False

    def prepare_data(self):
        if not self.is_load_file:
            if self.df_trainData is not None:
                train_set = np.array(self.df_trainData["SMILES"]).tolist()
                self.train_pos = int(len(self.df_trainData) * 0.8)
                self.train_obj = self.chem_tok(train_set[:self.train_pos], padding='max_length', 
                                max_length=self.max_length, truncation=True, return_tensors="pt")
                self.valid_obj = self.chem_tok(train_set[self.train_pos:], padding='max_length', 
                                max_length=self.max_length, truncation=True, return_tensors="pt")
                self.train_label = torch.from_numpy(np.array(self.df_trainLabel[:self.train_pos])) 
                self.valid_label = torch.from_numpy(np.array(self.df_trainLabel[self.train_pos:]))
                self.train_features = torch.from_numpy(np.array(self.df_trainfeatures[:self.train_pos])) if self.df_trainfeatures is not None else None
                self.valid_features = torch.from_numpy(np.array(self.df_trainfeatures[self.train_pos:])) if self.df_trainfeatures is not None else None
            
            if self.df_testData is not None:
                test_set = np.array(self.df_testData["SMILES"]).tolist()
                self.test_obj = self.chem_tok(test_set, padding='max_length', 
                                max_length=self.max_length, truncation=True, return_tensors="pt")
                self.test_label = torch.from_numpy(np.array(self.df_testLabel))
                self.test_features = torch.from_numpy(np.array(self.df_testfeatures)) if self.df_testfeatures is not None else None

            if self.df_predictData is not None:
                predict_set = np.array(self.df_predictData["SMILES"]).tolist()
                self.predict_obj = self.chem_tok(predict_set, padding='max_length', 
                                max_length=self.max_length, truncation=True, return_tensors="pt")
                self.predict_label = torch.from_numpy(np.array(self.df_predictLabel)) if self.df_predictLabel is not None else None
                self.predict_features = torch.from_numpy(np.array(self.df_predictfeatures)) if self.df_predictfeatures is not None else None

            self.is_load_file = True

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            if self.df_trainData is not None:
                self.train_dataset = clearanceDataset(self.train_obj, self.train_label, self.train_features)
            if self.df_trainData is not None:
                self.valid_dataset = clearanceDataset(self.valid_obj, self.valid_label, self.valid_features)
        
        if self.df_testData is not None:
            self.test_dataset = clearanceDataset(self.test_obj, self.test_label, self.test_features)

        if stage == 'predict' or stage is None:
            if self.df_predictData is not None:
                self.pred_dataset = clearanceDataset(self.predict_obj, self.predict_label, self.predict_features)

    def train_dataloader(self):
        if self.df_trainData is not None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return None

    def val_dataloader(self):
        if self.df_trainData is not None:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return None
        
    def test_dataloader(self):
        if self.df_testData is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return None

    def predict_dataloader(self):
        if self.df_predictData is not None:
            return DataLoader(self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return None
    
    
class clearanceKfoldDatamodule(pl.LightningDataModule):
    def __init__(self, chem_tokenizer, config,
                 df_trainData, df_testData,
                 df_trainLabel, df_testLabel,
                 train_scaler, test_scaler,
                 df_trainfeatures = None, df_testfeatures = None, kfold = 5):
        super().__init__()
        self.chem_tok = chem_tokenizer
        
        self.max_length = config.chem_max
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.df_trainData, self.df_testData = df_trainData, df_testData
        self.df_trainLabel, self.df_testLabel = df_trainLabel, df_testLabel
        self.train_scaler, self.test_scaler = train_scaler, test_scaler
        
        self.df_trainfeatures, self.df_testfeatures = df_trainfeatures, df_testfeatures
        self.kfold = kfold

    def setup(self, stage: Optional[str] = None) -> None:
        ## -- train/test smile dataset setting -- ##
        train_set, test_set = np.array(self.df_trainData["SMILES"]).tolist(), np.array(self.df_testData["SMILES"]).tolist()
        
        ## -- train/test SMILES dataset tokenization -- ##
        train_encoding = self.chem_tok(train_set, padding='max_length', 
                        max_length=self.max_length, truncation=True, return_tensors="pt")
        test_encoding = self.chem_tok(test_set, padding='max_length', 
                        max_length=self.max_length, truncation=True, return_tensors="pt")
        
        ## -- train/test Label setting -- ##
        train_label = torch.from_numpy(np.array(self.df_trainLabel)) 
        test_label = torch.from_numpy(np.array(self.df_testLabel))
        
        ## -- feature data setting -- ##
        if self.df_trainfeatures is not None:
            train_features = torch.from_numpy(np.array(self.df_trainfeatures)) 
            self.trainset = clearanceDataset(train_encoding, train_label, train_features)
            # self.trainset = TensorDataset(train_encoding['input_ids'], train_encoding['attention_mask'], train_label, train_features)
        else:
            self.trainset = clearanceDataset(train_encoding, train_label)
            # self.trainset = TensorDataset(train_encoding['input_ids'], train_encoding['attention_mask'], train_label)
            
        if self.df_testfeatures is not None:
            test_features = torch.from_numpy(np.array(self.df_testfeatures))
            self.testset = clearanceDataset(test_encoding, test_label, test_features)
            # self.testset = TensorDataset(test_encoding['input_ids'], test_encoding['attention_mask'], test_label, test_features)
        else:
            self.testset = clearanceDataset(test_encoding, test_label)
            # self.testset = TensorDataset(test_encoding['input_ids'], test_encoding['attention_mask'], test_label)
            
        self.splits = list(KFold(n_splits=self.kfold, shuffle=True).split(self.trainset))

            
        
    def train_dataloader(self, fold_index):
        train_idx, val_idx = self.splits[fold_index]
        train_dataset = torch.utils.data.Subset(self.trainset, train_idx)
        val_dataset = torch.utils.data.Subset(self.trainset, val_idx)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
        
    def test_dataloader(self):
        self.test_loader = DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers)
        return self.test_loader