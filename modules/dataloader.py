import os
import pandas as pd
import numpy as np

from typing import Optional
from itertools import cycle, product

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from transformers import AutoTokenizer

from utils.normalize_process import normalization, inverse_scaling
from sklearn.model_selection import train_test_split

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
    def __init__(self, data, features, labels):
        self.data = data
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        data = {"input_ids": self.data["input_ids"][idx], 
                "attention_mask": self.data["attention_mask"][idx]}
        
        features = self.features[idx]
        label = self.labels[idx]

        return data, features, label
        

class clearanceDatamodule(pl.LightningDataModule):
    def __init__(self,
                 df_trainfeatures:pd.DataFrame, df_testfeatures:pd.DataFrame, 
                 df_trainlabels, df_testlabels,
                 train_scaler, test_scaler,
                 batch_size: int, num_workers: int, 
                 train_obj, test_obj = None):
        super().__init__()
        self.df_trainfeatures = df_trainfeatures
        self.df_testfeatures = df_testfeatures

        self.df_trainlabels = df_trainlabels
        self.df_testlabels = df_testlabels

        self.train_scaler = train_scaler
        self.test_scaler = test_scaler
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_obj = train_obj
        self.test_obj = test_obj

        self.is_load_file = False
        self.sep_test = False

    def prepare_data(self):
        if not self.is_load_file:
            if self.test_obj != None:
                self.train_pos = int(len(self.train_obj['input_ids']) * 0.8)

                self.train_data = {"input_ids":self.train_obj["input_ids"][:self.train_pos], 
                                "attention_mask": self.train_obj["attention_mask"][:self.train_pos]}
                self.valid_data = {"input_ids":self.train_obj["input_ids"][self.train_pos:], 
                                "attention_mask": self.train_obj["attention_mask"][self.train_pos:]}
                
                self.train_labels = torch.from_numpy(np.array(self.df_trainlabels[:self.train_pos]))
                self.valid_labels = torch.from_numpy(np.array(self.df_trainlabels[self.train_pos:]))

                self.train_features = torch.from_numpy(np.array(self.df_trainfeatures[:self.train_pos])) 
                self.valid_features = torch.from_numpy(np.array(self.df_trainfeatures[self.train_pos:]))
                
                self.test_data = self.test_obj
                self.test_labels = torch.from_numpy(np.array(self.df_testlabels))
                self.test_features = torch.from_numpy(np.array(self.df_testfeatures))

            else:
                self.train_pos = int(len(self.train_obj['input_ids']) * 0.6)
                self.valid_pos = self.train_pos + int(len(self.train_obj['input_ids']) * 0.2)

                self.train_data = {"input_ids":self.train_obj["input_ids"][:self.train_pos], 
                                "attention_mask": self.train_obj["attention_mask"][:self.train_pos]}
                self.valid_data = {"input_ids":self.train_obj["input_ids"][self.train_pos:self.valid_pos], 
                                "attention_mask": self.train_obj["attention_mask"][self.train_pos:self.valid_pos]}
                self.test_data = {"input_ids":self.train_obj["input_ids"][self.valid_pos:], 
                                "attention_mask": self.train_obj["attention_mask"][self.valid_pos:]}
                
                self.train_labels = torch.from_numpy(np.array(self.df_trainlabels[:self.train_pos]))
                self.valid_labels = torch.from_numpy(np.array(self.df_trainlabels[self.train_pos:self.valid_pos]))
                self.test_labels = torch.from_numpy(np.array(self.df_trainlabels[self.valid_pos:]))

                self.train_features = torch.from_numpy(np.array(self.df_trainfeatures[:self.train_pos])) 
                self.valid_features = torch.from_numpy(np.array(self.df_trainfeatures[self.train_pos:self.valid_pos]))
                self.test_features = torch.from_numpy(np.array(self.df_trainfeatures[self.valid_pos:]))

                self.sep_test = True

            self.is_load_file = True

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = clearanceDataset(self.train_data, self.train_features, self.train_labels)
            self.valid_dataset = clearanceDataset(self.valid_data, self.valid_features, self.valid_labels)
        
        self.test_dataset = clearanceDataset(self.test_data, self.test_features, self.test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)