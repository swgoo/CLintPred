import os
import pandas as pd
import numpy as np
import json, copy

from easydict import EasyDict
from collections import OrderedDict

import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from utils.draw_plot import draw_plot, draw_boxplot


## -- protatin sequence type -- ##
prot_type = ["AAS_CYP9", "UGT_TYPE", "SULTs"]

## --  chemical Compound Feature type -- ##
features_columns = ["logP", "Fup"]
rdkit_columns = ["MW_rdkit", "HBD_rdkit", "HBA_rdkit", "NRB_rdkit", "RF_rdkit", "PSA_rdkit"]
default_columns = ["SMILES", "Clint"]

rangeLabel_col = ["MW_range", "PSA_range", "NRB_range", "HBA_range", "HBD_range", "LogP_range"]

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


def load_chemdata(df_loadData:pd.DataFrame, train_affinity:pd.DataFrame, feature_type:str = "default", scale = True, augmentation = False):
    df_loadData = df_loadData[(df_loadData['Clint'] <= 500)]
    df_loadData['Clint'] = np.log1p(df_loadData['Clint'])
    df_rdkitLabel = rdkit_rangeLabel(df_loadData)

    if feature_type.lower() == "default":
        df_loadData = df_loadData[default_columns].dropna(axis=0).reset_index(drop=True)

    elif feature_type.lower() == "features":
        df_loadData = df_loadData[(df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        df_loadData = df_loadData[(df_loadData['logP'] >= -2.0) & (df_loadData['logP'] <= 6.0)]
        df_loadData = df_loadData[default_columns + features_columns].dropna(axis=0)
        df_loadData = df_loadData.drop(df_loadData[df_loadData["logP"] == "None"].index).reset_index(drop=True)
    
    elif feature_type.lower() == "rdkit":
        df_loadData = df_loadData[(df_loadData['MW_rdkit'] < 1000.0)]
        df_loadData = df_loadData[(df_loadData['HBA_rdkit'] <= 10.0)]
        df_loadData = df_loadData[(df_loadData['HBD_rdkit'] <= 5.0)]
        df_loadData = df_loadData[default_columns + rdkit_columns].dropna(axis=0).reset_index(drop=True)
        
    else:
        df_loadData = df_loadData[(df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        df_loadData = df_loadData[(df_loadData['logP'] >= -2.0) & (df_loadData['logP'] <= 6.0)]

        df_loadData = df_loadData[(df_loadData['MW_rdkit'] < 1000.0)]
        df_loadData = df_loadData[(df_loadData['HBA_rdkit'] <= 10.0)]
        df_loadData = df_loadData[(df_loadData['HBD_rdkit'] <= 5.0)]

        df_loadData = df_loadData[default_columns + features_columns + rdkit_columns].dropna(axis=0)
        df_loadData = df_loadData.drop(df_loadData[df_loadData["logP"] == "None"].index).reset_index(drop=True)

    df_affinityData = train_affinity[train_affinity["SMILES"].isin(list(df_loadData["SMILES"]))].reset_index(drop = True)
    df_rdkitLabel = df_rdkitLabel[df_rdkitLabel["SMILES"].isin(list(df_loadData["SMILES"]))].reset_index(drop = True)

    ## -- log scaled dataset augmentation--  ##
    if augmentation is True:
        df_augmentedData = df_loadData[(df_loadData['Clint'] >= 0.5)]
        df_augmentedFeature = df_affinityData[df_affinityData["SMILES"].isin(list(df_augmentedData["SMILES"]))]
        df_augmentedrdkitLabel = df_rdkitLabel[df_rdkitLabel["SMILES"].isin(list(df_augmentedData["SMILES"]))]

        df_loadData = pd.concat([df_loadData, df_augmentedData], axis=0).reset_index(drop = True)
        df_affinityData = pd.concat([df_affinityData, df_augmentedFeature], axis=0).reset_index(drop = True)
        df_rdkitLabel = pd.concat([df_rdkitLabel, df_augmentedrdkitLabel], axis=0).reset_index(drop = True)
        
    df_rdkitLabel = df_rdkitLabel[rangeLabel_col]
    
    datacols = list(df_loadData.columns[1:])
    data_scaler = MinMaxScaler()

    if scale:
        df_loadData[datacols] = df_loadData[datacols].astype('float')
        scaled_data = data_scaler.fit_transform(df_loadData[datacols])
        df_loadData[datacols] = scaled_data

    return df_loadData, data_scaler, df_affinityData, df_rdkitLabel


# def load_pairchemdata(train_path:str, test_path:str, feature_type:str = "default", scale = True):
#     df_trainData, df_testData = pd.read_csv(train_path), pd.read_csv(test_path)

#     # df_trainData = df_trainData[(df_trainData['Clint'] <= 500) & (df_trainData['Fup'] >= 0.01) & (df_trainData['Fup'] <= 0.99)]
#     # df_testData = df_testData[(df_testData['Clint'] <= 500) & (df_testData['Fup'] >= 0.01) & (df_testData['Fup'] <= 0.99)]

#     # df_trainData = df_trainData[(df_trainData['Clint'] <= 500)]
#     # df_testData = df_testData[(df_testData['Clint'] <= 500)]

#     df_trainData['Clint'] = np.log1p(df_trainData['Clint'])
#     df_testData['Clint'] = np.log1p(df_testData['Clint'])

#     if feature_type.lower() == "default":
#         df_trainData = df_trainData[default_columns].dropna(axis=0).reset_index(drop=True)
#         df_testData = df_testData[default_columns].dropna(axis=0).reset_index(drop=True)

#     elif feature_type.lower() == "parts":
#         df_trainData = df_trainData[default_columns + features_columns].dropna(axis=0).reset_index(drop=True)

#         df_testData = df_testData[default_columns + features_columns].dropna(axis=0)
#         df_testData = df_testData.drop(df_testData[df_testData["logP"] == "None"].index).reset_index(drop=True)
        
#     else:
#         df_trainData = df_trainData[default_columns + features_columns + rdkit_columns].dropna(axis=0).reset_index(drop=True)

#         df_testData = df_testData[default_columns + features_columns + rdkit_columns].dropna(axis=0)
#         df_testData = df_testData.drop(df_testData[df_testData["logP"] == "None"].index).reset_index(drop=True)
    
#     datacols = list(df_trainData.columns[1:])
#     train_scaler, test_scaler = MinMaxScaler(), MinMaxScaler()

#     if scale:
#         df_trainData[datacols], df_testData[datacols] = df_trainData[datacols].astype('float'), df_testData[datacols].astype('float')
#         train_data, test_data = train_scaler.fit_transform(df_trainData[datacols]), test_scaler.fit_transform(df_testData[datacols])
#         df_trainData[datacols], df_testData[datacols] = train_data, test_data

#     return df_trainData, df_testData, train_scaler, test_scaler


def inverse_data(df_data:pd.DataFrame, scaler:MinMaxScaler):
    inverted_data = scaler.inverse_transform(df_data)
    df_data[:] = inverted_data
    
    return df_data


def rdkit_rangeLabel(df_feature:pd.DataFrame):
    df_feature[rangeLabel_col] = np.NaN

    MW_range = range(200, 601, 100)
    PSA_range = [50, 75, 100, 150]
    NRB_range = [3,5,7,10]
    HBA_range = [1,3,5,7,10]
    HBD_range = [1,3,5,7,10]
    LogP_range = range(0, 5)

    MW_label = ["<200", "200-300", "300-400", "400-500", "500-600",">=600"]
    PSA_label = ["<50", "50-75", "75-100", "100-150", ">=150"]
    NRB_label = ["<3", "3-5", "5-7", "7-10", ">=10"]
    HBA_label = ["<1", "1-3", "3-5", "5-7", "7-10", ">=10"]
    HBD_label = ["<1", "1-3", "3-5", "5-7", "7-10", ">=10"]
    LogP_label = ["<0", "0-1", "1-2", "2-3", "3-4",">=4"]

    ## -- make MW_rdkit range dataset -- ##
    for idx, _ in enumerate(MW_range):
        if idx == 0:
            df_feature["MW_range"][df_feature[df_feature["MW_rdkit"] < MW_range[idx]].index] = MW_label[idx]
        else:
            df_feature["MW_range"][df_feature[(df_feature["MW_rdkit"] >= MW_range[idx-1]) & (df_feature["MW_rdkit"] < MW_range[idx])].index] = MW_label[idx]

            if idx == (len(MW_range)-1):
                df_feature["MW_range"][df_feature[df_feature["MW_rdkit"] >= MW_range[idx]].index] = MW_label[idx+1]


    for idx, _ in enumerate(PSA_range):
        if idx == 0:
            df_feature["PSA_range"][df_feature[df_feature["PSA_rdkit"] < PSA_range[idx]].index] = PSA_label[idx]
        else:
            df_feature["PSA_range"][df_feature[(df_feature["PSA_rdkit"] >= PSA_range[idx-1]) & (df_feature["PSA_rdkit"] < PSA_range[idx])].index] = PSA_label[idx]

            if idx == (len(PSA_range)-1):
                df_feature["PSA_range"][df_feature[df_feature["PSA_rdkit"] >= PSA_range[idx]].index] = PSA_label[idx+1]


    for idx, _ in enumerate(NRB_range):
        if idx == 0:
            df_feature["NRB_range"][df_feature[df_feature["NRB_rdkit"] < NRB_range[idx]].index] = NRB_label[idx]
        else:
            df_feature["NRB_range"][df_feature[(df_feature["NRB_rdkit"] >= NRB_range[idx-1]) & (df_feature["NRB_rdkit"] < NRB_range[idx])].index] = NRB_label[idx]

            if idx == (len(NRB_range)-1):
                df_feature["NRB_range"][df_feature[df_feature["NRB_rdkit"] >= NRB_range[idx]].index] = NRB_label[idx+1]


    for idx, _ in enumerate(HBA_range):
        if idx == 0:
            df_feature["HBA_range"][df_feature[df_feature["HBA_rdkit"] < HBA_range[idx]].index] = HBA_label[idx]
        else:
            df_feature["HBA_range"][df_feature[(df_feature["HBA_rdkit"] >= HBA_range[idx-1]) & (df_feature["HBA_rdkit"] < HBA_range[idx])].index] = HBA_label[idx]

            if idx == (len(HBA_range)-1):
                df_feature["HBA_range"][df_feature[df_feature["HBA_rdkit"] >= HBA_range[idx]].index] = HBA_label[idx+1]


    for idx, _ in enumerate(HBD_range):
        if idx == 0:
            df_feature["HBD_range"][df_feature[df_feature["HBD_rdkit"] < HBD_range[idx]].index] = HBD_label[idx]
        else:
            df_feature["HBD_range"][df_feature[(df_feature["HBD_rdkit"] >= HBD_range[idx-1]) & (df_feature["HBD_rdkit"] < HBD_range[idx])].index] = HBD_label[idx]

            if idx == (len(HBD_range)-1):
                df_feature["HBD_range"][df_feature[df_feature["HBD_rdkit"] >= HBD_range[idx]].index] = HBD_label[idx+1]


    for idx, _ in enumerate(LogP_range):
        if idx == 0:
            df_feature["LogP_range"][df_feature[df_feature["logP_rdkit"] < LogP_range[idx]].index] = LogP_label[idx]
        else:
            df_feature["LogP_range"][df_feature[(df_feature["logP_rdkit"] >= LogP_range[idx-1]) & (df_feature["logP_rdkit"] < LogP_range[idx])].index] = LogP_label[idx]

            if idx == (len(LogP_range)-1):
                df_feature["LogP_range"][df_feature[df_feature["logP_rdkit"] >= LogP_range[idx]].index] = LogP_label[idx+1]

    return df_feature


def save_result(config, result_dir, df_result:pd.DataFrame,  model_metric:dict, df_train:pd.DataFrame = None, df_test:pd.DataFrame = None):
    result_name = f"{config.model_type}_{config.train_dataType}_r2_{model_metric['r2']:.3f}"
    result_path = f"results/{result_dir}/{result_name}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if df_train is not None:
        df_train.to_csv(f'{result_path}/train.csv', index=False) 
    if df_test is not None:
        df_test.to_csv(f'{result_path}/test.csv', index=False)

    submit_config = {"model_type":config.model_type, "chem_model":config.chem_model,
                     "load_model": config.checkpoint_name, "feature_data": config.feature_type, "test_rate":config.test_rate,
                     "data_augmentation" : config.augmentation, "protainData_extended" : config.extend_protType,
                     "chem_max" : config.chem_max, "batch_size" : config.batch_size,
                     "lr":config.lr, "dropout":config.dropout,
                     "MSE": str(model_metric['rmse']), "MAE": str(model_metric['MAE']), "r2": str(model_metric['r2'])}

    with open(os.path.join(result_path,"config.json"), 'w', encoding='utf-8') as mf:
        json.dump(submit_config, mf, indent='\t')

    result_file = f'{result_path}/results.csv'
    df_result.to_csv(result_file, index=False)
    print('Done.')
    
    save_file = os.path.join(result_path, "figure.png")
    draw_plot(result_file, save_file)

    # draw_boxplot(result_file, save_path)
    
