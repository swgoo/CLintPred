import os, copy
import pandas as pd
import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from transformers import AutoTokenizer

import wandb

from utils.utils import DictX, load_hparams, load_protdata, load_chemdata, save_result, inverse_data
from modules.dataloader import BiomakerDataModule, clearanceDatamodule
from modules.biomarker import BiomarkerModel, BindingAffinityModel
from modules.clearance import clearanceDecoderModel, clearanceEncoderModel


## -- Logger Type -- ##
TENSOR = 0
WANDB = 1
SWEEP = 2


def concat_predData(pred_result, affinity_columns):
    df_result = pd.DataFrame(columns=affinity_columns)

    ids = np.array(torch.cat([output['ids'] for output in pred_result], dim=0)).tolist()
    outputs = np.array(torch.cat([output['outputs'] for output in pred_result], dim=0)).tolist()
    pred_list = {"ids": ids, "outputs":outputs}

    start_idx = 0
    for idx in range(1, len(pred_list['ids'])):
        if pred_list["ids"][idx-1] != pred_list["ids"][idx]:
            affinity_dataset = [affinities for affinities in pred_list["outputs"][start_idx : idx]] 
            df_result.loc[len(df_result)] = affinity_dataset

            start_idx = idx
    
    affinity_dataset = [affinities for affinities in pred_list["outputs"][start_idx :]] 
    df_result.loc[len(df_result)] = affinity_dataset

    return df_result


def pred_affinityData(model, affinity_path, affinity_columns, config, prot_obj, train_obj, test_obj=None):
    ## -- predict datset -- ##
    trainer = pl.Trainer(accelerator='gpu',
                            strategy='dp', 
                            devices=config.gpu_id)
    
    train_datamodule = BiomakerDataModule(train_obj, prot_obj, config)
    pred_train = trainer.predict(model, train_datamodule)

    train_affinities  = concat_predData(pred_train, affinity_columns)
    train_affinities.to_csv(f"{affinity_path}/{config.train_file}", index=False)

    if test_obj != None:
        test_datamodule = BiomakerDataModule(test_obj, prot_obj, config)
        pred_test = trainer.predict(model, test_datamodule)
        
        test_affinities = concat_predData(pred_test, affinity_columns)
        test_affinities.to_csv(f"{affinity_path}/{config.test_file}", index=False)
        
        return train_affinities, test_affinities
    
    return train_affinities


def main(config):
    try: 
        ##-- hyper param config file Load --##
        if run_type == TENSOR:
            config = DictX(config)
        else:
            if config is not None:
                wandb.init(config=config, project=project_name)
            else:
                wandb.init(settings=wandb.Settings(console='off'))  

            config = wandb.config

        log_path = config.log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if run_type == TENSOR:
            logger = TensorBoardLogger(save_dir=log_path, name=project_name)
        else:
            logger = WandbLogger(project=project_name, save_dir=log_path)

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        pl.seed_everything(seed=config.num_seed)

        ## -- tokenized dataset declaration -- ##
        chem_tokenizer = AutoTokenizer.from_pretrained(config.chem_model)
        prot_tokenizer = AutoTokenizer.from_pretrained(config.prot_model)


        ## -- load protain and chemical dataset -- ##
        prot_path = config.prot_file
        df_protData = load_protdata(prot_path, config.extend_protType)
        affinity_columns = list(df_protData["name"])

        train_path =  os.path.join(config.data_folder, config.train_dataType, config.train_file)
        test_path = os.path.join(config.data_folder, config.train_dataType, config.test_file)
        
        df_trainData, train_scaler = load_chemdata(train_path, config.feature_type, config.scale)
        

        ## -- Tokenization load Dataset -- ##
        prot_obj = prot_tokenizer(np.array(df_protData["aas"]).tolist(), padding='max_length', 
                                    max_length=config.prot_max, truncation=True, return_tensors="pt")
        train_obj = chem_tokenizer(np.array(df_trainData["SMILES"]).tolist(), padding='max_length', 
                                    max_length=config.chem_max, truncation=True, return_tensors="pt")
        

        ## -- Test data load and Tokenization load Dataset -- ##
        test_obj = None
        if os.path.exists(test_path):
            df_testData, test_scaler = load_chemdata(test_path, config.feature_type, config.scale, True)

            test_obj = chem_tokenizer(np.array(df_testData["SMILES"]).tolist(), padding='max_length', 
                                    max_length=config.chem_max, truncation=True, return_tensors="pt")
        
        datacols = list(df_trainData.columns[1:])
       
        
        ## -- model declaration -- ##
        checkpoint_name = str(config.checkpoint_name).lower()
        ckpt_file = os.path.join("./checkpoint", f"{checkpoint_name}.ckpt")

        if checkpoint_name == "biomarker":
            model = BiomarkerModel.load_from_checkpoint(ckpt_file)
        else:
            model = BindingAffinityModel.load_from_checkpoint(ckpt_file)


        ## -- binding affinity prediction -- ##
        affinity_path = os.path.join(config.affinity_folder, checkpoint_name, config.train_dataType)
        if not os.path.exists(affinity_path):
            os.makedirs(affinity_path)

        train_affinityFile, test_affinityFile = f"{affinity_path}/{config.train_file}", f"{affinity_path}/{config.test_file}"

        if not os.path.exists(train_affinityFile):
            if os.path.exists(test_path):
                train_affinity, test_affinity = pred_affinityData(model, affinity_path, affinity_columns, config, prot_obj, train_obj, test_obj)
                train_labels, test_labels = df_trainData["Clint"], df_testData["Clint"]

                df_trainfeatures = pd.concat([train_affinity, df_trainData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else train_affinity
                df_testfeatures = pd.concat([test_affinity, df_testData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else test_affinity

            else:
                train_affinity = pred_affinityData(model, affinity_path, affinity_columns, config, prot_obj, train_obj)
                train_labels = df_trainData["Clint"]
                df_trainfeatures = pd.concat([train_affinity, df_trainData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else train_affinity
            
        else:
            train_affinity = pd.read_csv(train_affinityFile)
            train_labels = df_trainData["Clint"]
            df_trainfeatures = pd.concat([train_affinity, df_trainData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else train_affinity
                
            if os.path.exists(test_path):
                test_affinity = pd.read_csv(test_affinityFile)
                test_labels = df_testData["Clint"]
                df_testfeatures = pd.concat([test_affinity, df_testData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else test_affinity

        ## -- predict Clearance -- ##
        # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=5, mode="min")

        trainer = pl.Trainer(
            devices=config.gpu_id,
            max_epochs=config.max_epoch,
            logger = logger,
            # callbacks=[early_stop_callback],
            num_sanity_val_steps=0,
            accelerator='gpu',
            strategy='dp'           
        )

        feature_dim = len(df_trainfeatures.columns)
        datamodule = clearanceDatamodule(df_trainfeatures, df_testfeatures, 
                                         train_labels, test_labels, train_scaler, test_scaler, 
                                         config.batch_size, config.num_workers, train_obj, test_obj)
        ## -- clearance model declaration -- ##
        clearance_model = clearanceDecoderModel(config.lr, config.dropout, config.chem_model, feature_dim,
                                                df_trainData[datacols], df_testData[datacols])
        # clearance_model = clearanceEncoderModel(config.lr, config.dropout, config.chem_model, feature_dim,
        #                                         df_trainData[datacols], df_testData[datacols])
        trainer.fit(clearance_model, datamodule)
        
        clearance_model.eval()
        trainer.test(clearance_model, datamodule)

        save_result(config, project_name, clearance_model.test_result, clearance_model.test_log)
    
    except Exception as e:
        print(e)



if __name__ == "__main__":
    #-- wandb Sweep Hyper Param Tuning --##
    run_type = TENSOR
    
    if run_type == SWEEP:
        config = load_hparams('config/sweep/1_config_train.json')
        project_name = config["name"]
        sweep_id = wandb.sweep(config, project=project_name)
        wandb.agent(sweep_id, main)
    
    else:
        config = load_hparams('config/1_config_train.json')
        project_name = config["name"]
        main(config)