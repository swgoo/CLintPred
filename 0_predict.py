import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoConfig, RobertaModel, BertModel

from utils.utils import load_hparams, load_protdata, DictX
from modules.dataloader import BiomakerDataModule
from modules.biomarker import BiomarkerModel, BindingAffinityModel

from sklearn.preprocessing import StandardScaler, MinMaxScaler

## --  chemical Compound Feature type -- ##
features_columns = ["logP", "Fup"]
rdkit_columns = ["MW_rdkit", "HBD_rdkit", "HBA_rdkit", "NRB_rdkit", "RF_rdkit", "PSA_rdkit"]
default_columns = ["SMILES", "Clint"]

prot_type = ["AAS_CYP9", "UGT_TYPE", "SULTs"]


def pred_affinities(pred_affinity, affinity_columns):
    ids = np.array(torch.cat([output['ids'] for output in pred_affinity], dim=0)).tolist()
    outputs = np.array(torch.cat([output['outputs'] for output in pred_affinity], dim=0)).tolist()
    pred_list = {"ids": ids, "outputs":outputs}

    df_result = pd.DataFrame(columns=affinity_columns)
    
    start_idx = 0
    for idx in range(1, len(pred_list['ids'])):
        if pred_list["ids"][idx-1] != pred_list["ids"][idx]:
            affinity_dataset = [affinities for affinities in pred_list["outputs"][start_idx : idx]] 
            df_result.loc[len(df_result)] = affinity_dataset

            start_idx = idx
    
    affinity_dataset = [affinities for affinities in pred_list["outputs"][start_idx :]] 
    df_result.loc[len(df_result)] = affinity_dataset

    return df_result

if __name__ == "__main__":
    config = load_hparams('./config/1_config_train.json')
    config = DictX(config)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    pl.seed_everything(seed=config.num_seed)

    ## -- load protain dataset -- ##
    prot_path = os.path.join(config.data_folder, config.prot_file)
    df_protData = pd.read_csv(prot_path)

    if not config.extend_protType:
        df_protData= df_protData[df_protData["type"] == prot_type[0]]

    affinity_columns = list(df_protData["name"])

    ## -- Load SMILES dataset -- ##
    train_path = os.path.join(config.data_folder, config.train_file)
    test_path = os.path.join(config.data_folder, config.test_file)

    df_trainData = pd.read_csv(train_path)
    df_testData = pd.read_csv(test_path)

    df_trainData = df_trainData[default_columns + features_columns].dropna(axis=0).reset_index(drop=True)
    df_testData = df_testData[default_columns + features_columns].dropna(axis=0).reset_index(drop=True)
    df_testData = df_testData.drop(df_testData[df_testData["logP"] == "None"].index).reset_index(drop=True)
    
    datacols = list(df_trainData.columns[1:])
    train_scaler, test_scaler = MinMaxScaler(), MinMaxScaler()

    if config.scale:
        df_trainData[datacols], df_testData[datacols] = df_trainData[datacols].astype('float'), df_testData[datacols].astype('float')
        train_data = train_scaler.fit_transform(df_trainData[datacols])
        test_data = test_scaler.fit_transform(df_testData[datacols])

        df_trainData[datacols] = train_data
        df_testData[datacols] = test_data

    ## -- model declaration -- ##
    checkpoint_name = "biomarker"
    ckpt_file = os.path.join("./checkpoint", f"{checkpoint_name}.ckpt")
    cls_model = BiomarkerModel.load_from_checkpoint(ckpt_file)

    checkpoint_name = "regression_bindingDB"
    ckpt_file = os.path.join("./checkpoint", f"{checkpoint_name}.ckpt")
    reg_model = BindingAffinityModel.load_from_checkpoint(ckpt_file)

    # pt_file = os.path.join("./checkpoint", f"{checkpoint_name}.pt")

    # if not os.path.exist(pt_file):
    #     ckpt_file = os.path.join("./checkpoint", f"{checkpoint_name}.ckpt")
    #     model = BiomarkerModel.load_from_checkpoint(ckpt_file)

    #     torch.save({"chem_model": model.d_model.state_dict(),
    #                 "prot_model": model.p_model.state_dict(),
    #                 "decoder": model.decoder.state_dict()}, './checkpoint/biomarker.pt')

    # else:
    #     model = BiomarkerModel()
    #     checkpoint = torch.load(pt_file)

    #     model.d_model.load_state_dict(checkpoint["chem_model"])
    #     model.p_model.load_state_dict(checkpoint["prot_model"])
    #     model.decoder.load_state_dict(checkpoint["decoder"])

    train_datamodule = BiomakerDataModule(config.num_seed, df_trainData, df_protData, config)
    test_datamodule = BiomakerDataModule(config.num_seed, df_testData, df_protData, config)

    trainer = pl.Trainer(accelerator='gpu',
                        strategy='dp', 
                        devices=config.gpu_id)
    # trainer = pl.Trainer(accelerator='cpu')
    
    
    train_preds = trainer.predict(cls_model, train_datamodule)
    test_preds = trainer.predict(cls_model, test_datamodule)
    
    # ids = torch.as_tensor(torch.cat([output['ids'] for output in predict_result], dim=0))
    # outputs = torch.as_tensor(torch.cat([output['outputs'] for output in predict_result], dim=0))

    
    # ids_list, outputs_list = np.array(ids).tolist(), np.array(outputs).tolist()
    # df_result = pd.DataFrame([datas for datas in zip(ids_list, outputs_list)], columns= ["ids", 'affinity'])

    train_affinities = pred_affinities(train_preds, affinity_columns)
    test_affinities = pred_affinities(test_preds, affinity_columns)

    if config.feature_type.lower() != "default":
        df_trainfeatures = pd.concat([train_affinities,df_trainData.iloc[:10, 2:]], axis=1)
        df_testfeatures = pd.concat([test_affinities,df_trainData.iloc[:10, 2:]], axis=1)
    else:
        df_trainfeatures = train_affinities
        df_testfeatures = test_affinities
        
    # df_result.to_csv("./result/Clint_test/affinity.csv")
    print(test_affinities)

    
