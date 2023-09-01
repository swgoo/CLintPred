import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def norm_Clint_scaler(df_data, is_testCode = False):
    # df_norm = df_data[(df_data['logP'] >= -2) & (df_data['logP'] <= 8) & (v['Clint'] <= 500) & (df_data['Fup'] >= 0.01) & (df_load['Fup'] <= 0.99)]
    df_norm = df_data[(df_data['Clint'] <= 500) & (df_data['Fup'] >= 0.01) & (df_data['Fup'] <= 0.99)]

    df_norm['Clint'] = np.log1p(df_norm['Clint'])
    if not is_testCode:
        df_norm['Clint'], scaler = scaling(df_norm['Clint'])
        return df_norm, scaler
    else:
        df_norm['Clint'], df_mean, df_std= scaling_test(df_norm['Clint'])
        return df_norm, [df_mean, df_std]


def norm_shuffledSet_merge(df_train, df_test):
    # df_train.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0 ,inplace=True)
    # df_test.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0 ,inplace=True)

    merged_dataset = pd.concat([df_train, df_test])
    merged_dataset.reset_index(drop=True, inplace=True)

    # df_merged_norm, merged_scaler = norm_Clint_scaler(merged_dataset)
    df_merged_norm, merged_scaler = norm_Clint_scaler(merged_dataset, is_testCode=True)

    # df_train_norm = df_merged_norm.iloc[:len(df_train),:]
    # df_test_norm = df_merged_norm.iloc[len(df_train):,:]

    df_train_norm = df_merged_norm.sample(frac=0.8, random_state=42)
    df_test_norm = df_merged_norm.drop(df_train_norm.index)

    return df_train_norm, df_test_norm, merged_scaler


def norm_shuffledSet(df_train:pd.DataFrame, df_test:pd.DataFrame, random_seed:int, shuffle_rate = None):
    # df_train.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0 ,inplace=True)
    # df_test.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0 ,inplace=True)

    if shuffle_rate is None:
        df_train_norm, train_scaler = norm_Clint_scaler(df_train)
        df_test_norm, test_scaler = norm_Clint_scaler(df_test)
    
    else:
        valid = df_test.sample(frac=shuffle_rate, random_state=random_seed)

        df_test = df_test.drop(valid.index)
        # df_test = df_test.loc[~df_test["SMILES"].isin(valid["SMILES"].tolist())]
        df_merge_train = pd.concat([df_train, valid])

        df_merge_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        df_train_norm, train_scaler = norm_Clint_scaler(df_merge_train)
        df_test_norm, test_scaler = norm_Clint_scaler(df_test)

        # df_train_norm, train_scaler = norm_Clint_scaler(df_merge_train, is_testCode=True)
        # df_test_norm, test_scaler = norm_Clint_scaler(df_test, is_testCode=True)

    return df_train_norm, train_scaler, df_test_norm, test_scaler
    # return df_train_norm, df_test_norm


def normalization(df_load, rm_duplicates:bool = False):
    ## -- Read dataset drop duplication and Nan data-- ##
    if not "SMILES" in df_load.columns:
        df_load.rename(columns = {'SMILES_rdkit_final':'SMILES'},inplace=True)
        
    df_load.rename(columns = {'Clint.invitro.':'Clint'},inplace=True)
    df_load.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0 ,inplace=True)

    ## -- Remove duplicates SMILES Data -- ##
    if rm_duplicates:
        df_load.drop_duplicates(['SMILES'])
    
    ## -- Check Attributes Values -- ##
    # df_norm = df_load[(df_load['logP'] >= -2) & (df_load['logP'] <= 8) & (df_load['Clint'] <= 500) & (df_load['Fup'] >= 0.01) & (df_load['Fup'] <= 0.99)]
    df_norm = df_load[(df_load['Clint'] <= 500) & (df_load['Fup'] >= 0.01) & (df_load['Fup'] <= 0.99)]
    
    # ## -- Clearance Normalization -- ##
    df_norm['Clint'] = np.log1p(df_norm['Clint'])

    # ## -- F_up Normalization -- ##
    df_norm['Fup'] = np.log1p(df_norm['Fup'])
    scaling_info = {"Clint": None, "Fup": None, "logP": None}

    df_norm['Clint'], scaler_clint = scaling(df_norm['Clint'])
    df_norm['Fup'], scaler_fup = scaling(df_norm['Fup'])
    df_norm['logP'], scaler_logp = scaling(df_norm['logP'])
    
    return df_norm, scaler_clint


def normalization_logscale(df_load):
    ## -- Read dataset drop duplication and Nan data-- ##
    if not "SMILES" in df_load.columns:
        df_load.rename(columns = {'SMILES_rdkit_final':'SMILES'},inplace=True)
        
    df_load.rename(columns = {'Clint.invitro.':'Clint'},inplace=True)
    df_load.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0 ,inplace=True)

    ## -- Check Attributes Values -- ##
    # df_norm = df_load[(df_load['logP'] >= -2) & (df_load['logP'] <= 8) & (df_load['Clint'] <= 500) & (df_load['Fup'] >= 0.01) & (df_load['Fup'] <= 0.99)]
    df_norm = df_load[(df_load['Clint'] <= 500) & (df_load['Fup'] >= 0.01) & (df_load['Fup'] <= 0.99)]
    
    # ## -- Clearance Normalization -- ##
    df_norm['Clint'] = np.log1p(df_norm['Clint'])
    
    return df_norm


def scaling_test(df_data):
    dataset_train = df_data.astype(np.float64)

    features = dataset_train
    data_mean = features.mean(axis=0)
    data_std = features.std(axis=0)
    features = (features-data_mean)/data_std
    dataset_train = features

    return dataset_train, data_mean, data_std


def scaling(df_data):
    scaler = MinMaxScaler(feature_range = (0,1))
    data_reshape = df_data.values.reshape(-1,1)
    scaler_fit = scaler.fit(data_reshape)
    scaling_data = scaler_fit.transform(data_reshape)
    scaling_data = scaling_data.flatten()

    return scaling_data, scaler_fit


def inverse_scaling(df_datas, scaler_fit, log_scale=False):
    df_datas = df_datas.reshape(-1,1)
    inverse_scale = scaler_fit.inverse_transform(df_datas)

    if log_scale:
        inverse_scale = np.expm1(inverse_scale)

    inverse_scale = inverse_scale.flatten()

    return inverse_scale

def inverse_normalize(df_norm):
    df_norm["Clint"] = np.expm1(df_norm["Clint"]) - 0.01
    df_norm['Fup'] = np.log1p(df_norm['Fup']) - 0.001

    # df_norm['logP'] = np.log1p(df_norm['logP']) - 0.01

    return df_norm