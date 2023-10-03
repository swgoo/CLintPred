import torch
import torch.nn as nn

import pytorch_lightning as pl

from utils.utils import *
from transformers import AutoConfig, RobertaModel, BertModel


activation = {}

def get_hiddenlayerweight(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


class transformerEncoder(pl.LightningModule):
    def __init__(self, input_dim:int, num_heads:int = 2, num_layers:int = 18, act_function:str = "relu", dropout:float = 0.1):
        super().__init__()
        act_func = get_actfunction(act_function.lower())

        transEncoderLayer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=input_dim * num_heads, 
                                                       dropout=dropout, activation=act_func, batch_first=True)
        self.TransEncoder = nn.TransformerEncoder(transEncoderLayer, num_layers=num_layers)

        self.pooling_layer = nn.Linear(input_dim, input_dim)
        # self.activation1 = nn.ReLU()
        # self.dropout1=nn.Dropout(0.1)

        # self.outputLayer1 = nn.Linear(input_dim, 256)
        # self.activation1 = nn.ReLU()
        # self.dropout1=nn.Dropout(0.1)

        # self.outputLayer2 = nn.Linear(256, 1)

    def forward(self, src):
        # src = src.permute(1, 0, 2)
        SMILES_hidden_states = self.TransEncoder(src)

        smiles_sequence = SMILES_hidden_states[:, 0,:]
        outputs = self.pooling_layer(smiles_sequence)

        # outputs = self.activation1(outputs)
        # outputs = self.dropout1(outputs)
        
        # outputs = self.outputLayer1(outputs)
        # outputs = self.activation1(outputs)
        # outputs = self.dropout1(outputs)

        # predict_score = self.outputLayer2(outputs)

        return smiles_sequence, outputs
    

class transformerDecoder(pl.LightningModule):
    def __init__(self, input_dim:int, num_heads:int = 2, num_layers:int = 18, act_function:str = "relu", dropout:float = 0.1):
        super().__init__()
        act_func = get_actfunction(act_function.lower())

        TransDecoderLayer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=input_dim * num_heads, dropout=dropout, activation=act_func, batch_first=True)
        self.TransDecoder = nn.TransformerDecoder(TransDecoderLayer, num_layers=num_layers)

        self.pooling_layer = nn.Linear(input_dim, input_dim)
        # self.activation1 = nn.ReLU()
        # self.dropout1=nn.Dropout(0.1)

        self.decoder1 = nn.Linear(input_dim, 256)
        self.activation1 = nn.ReLU()
        self.dropout1=nn.Dropout(0.1)

        self.decoder2 = nn.Linear(256, 1)

    def forward(self, src, mem):
        # src = src.permute(1, 0, 2)
        # SMILES_hidden_states = self.TransDecoder(src, mem)
        SMILES_hidden_states = self.TransDecoder(src, mem)
        outputs = self.pooling_layer(SMILES_hidden_states)

        # outputs = self.activation1(outputs)
        # outputs = self.dropout1(outputs)
        
        outputs = self.decoder1(outputs)
        outputs = self.activation1(outputs)
        outputs = self.dropout1(outputs)

        predict_score = self.decoder2(outputs)

        return SMILES_hidden_states, predict_score[:, -1, :]