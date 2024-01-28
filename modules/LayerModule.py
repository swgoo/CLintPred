import copy
from collections import OrderedDict

import torch.nn as nn

ACT2CLS = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

ACT2FN = ClassInstantier(ACT2CLS)

def get_actfunction(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
    

def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.layer = newModuleList

    return copyOfModel

    

def make_sequential(input_dim:int, output_dim:int, act_function:str = "relu", dropout:float = 0.1, shift:float = 0.0):
    act_func = get_actfunction(act_function.lower())
    sequencial = nn.Sequential(nn.Linear(input_dim, output_dim),
                                act_func,
                                nn.Dropout(dropout))

    return sequencial


def make_encoder(input_dim:int, num_heads:int, num_layers:int, act_function:str = "relu", dropout:float = 0.1):
    act_func = get_actfunction(act_function.lower())
    # sequence_pos_encoding = PositionalEncoding(input_dim, dropout) 
    seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=input_dim * num_heads, dropout=dropout, activation=act_func) 
    seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=num_layers)

    return seqTransEncoder

def make_decoder(input_dim:int, num_heads:int, num_layers:int, act_function:str = "relu", dropout:float = 0.1):
    act_func = get_actfunction(act_function.lower())
    # sequence_pos_encoding = PositionalEncoding(input_dim, dropout) 
    seq_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=input_dim * num_heads, dropout=dropout, activation=act_func) 
    seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer, num_layers=num_layers)

    return seqTransDecoder