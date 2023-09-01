import os
import pandas as pd
import numpy as np
import json, copy

from easydict import EasyDict
from collections import OrderedDict

import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from utils.draw_plot import draw_plot


## -- protatin sequence type -- ##
prot_type = ["AAS_CYP9", "UGT_TYPE", "SULTs"]

## --  chemical Compound Feature type -- ##
features_columns = ["logP", "Fup"]
rdkit_columns = ["MW_rdkit", "HBD_rdkit", "HBA_rdkit", "NRB_rdkit", "RF_rdkit", "PSA_rdkit"]
default_columns = ["SMILES", "Clint"]


ACT2CLS = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:

            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2FN = ClassInstantier(ACT2CLS)



def load_hparams(file_path):
    hparams = EasyDict()
    with open(file_path, 'r') as f:
        hparams = json.load(f)
    return hparams


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


def load_protdata(file_path:str, extend_protType:bool = False):
    df_protData = pd.read_csv(file_path)

    if not extend_protType:
        df_protData= df_protData[df_protData["type"] == prot_type[0]]
    
    df_protData["aas"]  = [' '.join(list(aas)) for aas in df_protData["aas"]]

    return df_protData


def load_chemdata(data_path:str, feature_type:str = "default", scale = True):
    df_loadData = pd.read_csv(data_path)

    

    if feature_type.lower() == "default":
        df_loadData = df_loadData[(df_loadData['Clint'] <= 500)]
        df_loadData['Clint'] = np.log1p(df_loadData['Clint'])
        df_loadData = df_loadData[default_columns].dropna(axis=0).reset_index(drop=True)

    elif feature_type.lower() == "parts":
        df_loadData = df_loadData[(df_loadData['Clint'] <= 500) & (df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        df_loadData['Clint'] = np.log1p(df_loadData['Clint'])

        df_loadData = df_loadData[default_columns + features_columns].dropna(axis=0)
        df_loadData = df_loadData.drop(df_loadData[df_loadData["logP"] == "None"].index).reset_index(drop=True)
        
    else:
        df_loadData = df_loadData[(df_loadData['Clint'] <= 500) & (df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        df_loadData['Clint'] = np.log1p(df_loadData['Clint'])

        df_loadData = df_loadData[default_columns + features_columns + rdkit_columns].dropna(axis=0)
        df_loadData = df_loadData.drop(df_loadData[df_loadData["logP"] == "None"].index).reset_index(drop=True)

    ## -- log scaled dataset augmentation--  ##
    # df_augmentedData = df_loadData[(df_loadData['Clint'] >= 0.5)]
    # df_loadData = pd.concat([df_loadData, df_augmentedData], axis=0)
    
    datacols = list(df_loadData.columns[1:])
    data_scaler = MinMaxScaler()

    if scale:
        df_loadData[datacols] = df_loadData[datacols].astype('float')
        scaled_data = data_scaler.fit_transform(df_loadData[datacols])
        df_loadData[datacols] = scaled_data

    return df_loadData, data_scaler


def load_pairchemdata(train_path:str, test_path:str, feature_type:str = "default", scale = True):
    df_trainData, df_testData = pd.read_csv(train_path), pd.read_csv(test_path)

    # df_trainData = df_trainData[(df_trainData['Clint'] <= 500) & (df_trainData['Fup'] >= 0.01) & (df_trainData['Fup'] <= 0.99)]
    # df_testData = df_testData[(df_testData['Clint'] <= 500) & (df_testData['Fup'] >= 0.01) & (df_testData['Fup'] <= 0.99)]

    # df_trainData = df_trainData[(df_trainData['Clint'] <= 500)]
    # df_testData = df_testData[(df_testData['Clint'] <= 500)]

    df_trainData['Clint'] = np.log1p(df_trainData['Clint'])
    df_testData['Clint'] = np.log1p(df_testData['Clint'])

    if feature_type.lower() == "default":
        df_trainData = df_trainData[default_columns].dropna(axis=0).reset_index(drop=True)
        df_testData = df_testData[default_columns].dropna(axis=0).reset_index(drop=True)

    elif feature_type.lower() == "parts":
        df_trainData = df_trainData[default_columns + features_columns].dropna(axis=0).reset_index(drop=True)

        df_testData = df_testData[default_columns + features_columns].dropna(axis=0)
        df_testData = df_testData.drop(df_testData[df_testData["logP"] == "None"].index).reset_index(drop=True)
        
    else:
        df_trainData = df_trainData[default_columns + features_columns + rdkit_columns].dropna(axis=0).reset_index(drop=True)

        df_testData = df_testData[default_columns + features_columns + rdkit_columns].dropna(axis=0)
        df_testData = df_testData.drop(df_testData[df_testData["logP"] == "None"].index).reset_index(drop=True)
    
    datacols = list(df_trainData.columns[1:])
    train_scaler, test_scaler = MinMaxScaler(), MinMaxScaler()

    if scale:
        df_trainData[datacols], df_testData[datacols] = df_trainData[datacols].astype('float'), df_testData[datacols].astype('float')
        train_data, test_data = train_scaler.fit_transform(df_trainData[datacols]), test_scaler.fit_transform(df_testData[datacols])
        df_trainData[datacols], df_testData[datacols] = train_data, test_data

    return df_trainData, df_testData, train_scaler, test_scaler


def inverse_data(df_data:pd.DataFrame, scaler:MinMaxScaler):
    inverted_data = scaler.inverse_transform(df_data)
    df_data[:] = inverted_data
    
    return df_data


def save_result(config, result_dir, model_result:dict,  model_metric:dict):
    submit = {"Clint": model_result['labels'], "predict": model_result['preds']}
    submit = pd.DataFrame.from_dict(submit)

    result_name = f"lr_{config.lr}_r2_{model_metric['r2']:.3f}"
    result_path = f"results/{result_dir}/{result_name}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    submit_config = {"load_model": config.checkpoint_name, "lr":config.lr, "dropout":config.dropout,
                    "MSE": str(model_metric['rmse']), "MAE": str(model_metric['MAE']), "r2": str(model_metric['r2'])}

    with open(os.path.join(result_path,"config.json"), 'w', encoding='utf-8') as mf:
        json.dump(submit_config, mf, indent='\t')

    result_file = f'{result_path}/results.csv'
    submit.to_csv(result_file, index=False)
    print('Done.')
    
    save_file = os.path.join(result_path, "figure.png")
    draw_plot(result_file, save_file)
