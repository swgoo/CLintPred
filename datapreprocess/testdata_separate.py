import os

import pandas as pd
import numpy as np


def make_testdata(df_load, rm_duplicates:bool = False):
    ## -- Read dataset drop duplication and Nan data-- ##
    if not "SMILES" in df_load.columns:
        df_load.rename(columns = {'SMILES_rdkit_final':'SMILES'},inplace=True)
    
    df_load.rename(columns = {'Clint.invivo.':'Clint'},inplace=True)
    df_load.drop("Clint.invitro.", axis=1, inplace=True)
    df_load.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0,inplace=True)

    ## -- Remove duplicates SMILES Data -- ##
    if rm_duplicates:
        df_load.drop_duplicates(['SMILES'])

    return df_load


def make_traindata(df_load, rm_duplicates:bool = False):
    ## -- Read dataset drop duplication and Nan data-- ##
    if not "SMILES" in df_load.columns:
        df_load.rename(columns = {'SMILES_rdkit_final':'SMILES'},inplace=True)
        
    # df_load.rename(columns = {'Clint.invivo.':'Clint'},inplace=True)
    # df_load.drop("Clint.invitro.", axis=1, inplace=True)
    # df_load.dropna(subset=['logP','Clint','Fup'], axis=0,inplace=True)

    df_load = df_load.drop(df_load[df_load['Clint.invivo.'].notna()].index)
    
    df_load.rename(columns = {'Clint.invitro.':'Clint'},inplace=True)
    df_load.drop("Clint.invivo.", axis=1, inplace=True)
    df_load.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0,inplace=True)

    ## -- Remove duplicates SMILES Data -- ##
    if rm_duplicates:
        df_load.drop_duplicates(['SMILES'])

    return df_load


def convert_trainset(df_load):
    df_trainset = df_load.drop(df_load[df_load['Clint.invivo.'].notna()].index)

    df_trainset.drop("Clint.invivo.", axis=1, inplace=True)
    df_trainset.rename(columns = {'Clint.invitro.':'Clint'},inplace=True)
    return df_trainset


def convert_testset(df_load):
    df_testset = df_load.drop("Clint.invitro.", axis=1)
    
    df_testset.rename(columns = {'Clint.invivo.':'Clint'},inplace=True)
    df_testset.dropna(subset=['Clint'], axis=0, inplace=True)
    return df_testset

def convert_testset_invitro(df_load):
    df_testset = df_load.drop(df_load[df_load['Clint.invivo.'].isna()].index)

    df_testset.drop("Clint.invivo.", axis=1, inplace=True)
    df_testset.rename(columns = {'Clint.invitro.':'Clint'},inplace=True)
    return df_testset


def make_NoneOptDataset(df_load, rm_duplicates:bool = False):
    ## -- Read dataset drop duplication and Nan data-- ##
    if not "SMILES" in df_load.columns:
        df_load.rename(columns = {'SMILES_rdkit_final':'SMILES'},inplace=True)

    ## -- Remove duplicates SMILES Data -- ##
    if rm_duplicates:
        df_load.drop_duplicates(['SMILES'])

    ## -- delete duplicate invivo and invitro -- ##
    df_trainset = convert_trainset(df_load)
    df_testset = convert_testset(df_load)

    df_trainset.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0, inplace=True)
    df_testset.dropna(subset=['SMILES','logP','Clint','Fup'], axis=0, inplace=True)

    return df_trainset, df_testset

def make_pairedOptDataset(df_load, rm_duplicates:bool = False):
    ## -- Read dataset drop duplication and Nan data-- ##
    if not "SMILES" in df_load.columns:
        df_load.rename(columns = {'SMILES_rdkit_final':'SMILES'},inplace=True)

    df_load.dropna(subset=['SMILES','Clint.invitro.'], axis=0, inplace=True)
    
    ## -- Remove duplicates SMILES Data -- ##
    if rm_duplicates:
        df_load.drop_duplicates(['SMILES'])

    ## -- delete duplicate invivo and invitro -- ##
    df_trainset = convert_trainset(df_load)
    df_testset = convert_testset(df_load)

    return df_trainset, df_testset

def make_OnlyVitroDataset(df_load, rm_duplicates:bool = False):
    ## -- Read dataset drop duplication and Nan data-- ##
    if not "SMILES" in df_load.columns:
        df_load.rename(columns = {'SMILES_rdkit_final':'SMILES'},inplace=True)

    ## -- Remove duplicates SMILES Data -- ##
    if rm_duplicates:
        df_load.drop_duplicates(['SMILES'])

    ## -- delete duplicate invivo and invitro -- ##
    df_trainset = convert_trainset(df_load)
    df_testset = convert_testset_invitro(df_load)

    df_trainset.dropna(subset=['SMILES','Clint'], axis=0, inplace=True)
    df_testset.dropna(subset=['SMILES','Clint'], axis=0, inplace=True)

    return df_trainset, df_testset



if __name__ == "__main__":
    loaddata_path = "../dataset"
    dataset_file = os.path.join(loaddata_path, "train221027.csv")
    df_loaddata = pd.read_csv(dataset_file)

    ## -- 0_NoneOpt make df_train, df_test dataset -- ##
    save_path = "../dataset/0_NoneOpt"
    df_trainset, df_testset = make_NoneOptDataset(df_loaddata)
    train_file, test_file = os.path.join(save_path, "train_data_221027.csv"), os.path.join(save_path, "test_data_221027.csv")
    
    df_trainset.to_csv(train_file, index=False)
    df_testset.to_csv(test_file, index=False)

    ## -- 1_Paired make df_train, df_test dataset -- ##
    save_path = "../dataset/1_paired"
    df_trainset, df_testset = make_pairedOptDataset(df_loaddata)
    train_file, test_file = os.path.join(save_path, "train_data_221027.csv"), os.path.join(save_path, "test_data_221027.csv")

    df_trainset.to_csv(train_file, index=False)
    df_testset.to_csv(test_file, index=False)

    ## -- 2_Only Usetvitro dataset make df_train, df_test dataset -- ##
    save_path = "../dataset/2_onlyVitroset"
    df_trainset, df_testset = make_OnlyVitroDataset(df_loaddata)
    train_file, test_file = os.path.join(save_path, "train_data_221027.csv"), os.path.join(save_path, "test_data_221027.csv")

    df_trainset.to_csv(train_file, index=False)
    df_testset.to_csv(test_file, index=False)

    