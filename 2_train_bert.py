import os, copy
import pandas as pd
import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from transformers import AutoTokenizer

import wandb

from utils.utils import DictX, load_hparams, load_protdata, load_chemdata, save_result
from modules.dataloader import BiomakerDataModule, clearanceDatamodule
from modules.biomarker import BiomarkerModel, BindingAffinityModel
from modules.clearance import clearanceMLPModel, clearanceDecoderModel, clearanceEncoderModel


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


def pred_affinity(model, chem_tokenizer, prot_tokenizer, df_chem, df_prot, pred_file, config):

    ## -- predict datset -- ##
    trainer = pl.Trainer(accelerator='gpu',
                        strategy='dp', 
                        devices=config.gpu_id)

    train_obj = chem_tokenizer(np.array(df_chem["SMILES"]).tolist(), padding='max_length', 
                            max_length=config.chem_max, truncation=True, return_tensors="pt")
    prot_obj = prot_tokenizer(np.array(df_prot["aas"]).tolist(), padding='max_length', 
                            max_length=config.prot_max, truncation=True, return_tensors="pt")
    
    affinity_dm = BiomakerDataModule(train_obj, prot_obj, config)
    pred_train = trainer.predict(model, datamodule=affinity_dm)

    affinity_columns = list(df_prot["name"])
    affinities = concat_predData(pred_train, affinity_columns)
    affinities = pd.concat([df_chem["SMILES"], affinities], axis=1)

    affinities.to_csv(pred_file, index=False)

    return affinities


def main(config=None):
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

        if not os.path.exists(config.log_path):
            os.makedirs(config.log_path)

        if run_type == TENSOR:
            logger = TensorBoardLogger(save_dir=config.log_path, name=project_name)
        else:
            logger = WandbLogger(project=project_name, save_dir=config.log_path)

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        pl.seed_everything(seed=config.num_seed)

        ## -- tokenized dataset declaration -- ##
        chem_tokenizer = AutoTokenizer.from_pretrained(config.chem_model)
        prot_tokenizer = AutoTokenizer.from_pretrained(config.prot_model)

        
        ## -- load protain and chemical dataset -- ##
        prot_path = config.prot_file
        df_protData = load_protdata(prot_path, config.extend_protType)
        
        train_path =  os.path.join(config.data_folder, config.train_dataType, config.train_file)
        test_path = os.path.join(config.data_folder, config.train_dataType, config.test_file)


        ## -- binding affinity prediction -- ##
        checkpoint_name = str(config.checkpoint_name).lower()

        affinity_path = os.path.join(config.affinity_folder, checkpoint_name, config.train_dataType)
        if not os.path.exists(affinity_path):
            os.makedirs(affinity_path)

        train_affinityFile, test_affinityFile = f"{affinity_path}/{config.train_file}", f"{affinity_path}/{config.test_file}"

        ## -- model declaration -- ##
        if not os.path.exists(train_affinityFile) or not os.path.exists(test_affinityFile):
            ckpt_file = os.path.join("./checkpoint", f"{checkpoint_name}.ckpt")

            if checkpoint_name == "regression_bindingDB":
                model = BindingAffinityModel.load_from_checkpoint(ckpt_file)
            else:
                model = BiomarkerModel.load_from_checkpoint(ckpt_file)
            
        ## -- Predict binding Affinity -- ##
        df_train = pd.read_csv(train_path)
        train_affinity = pd.read_csv(train_affinityFile) if os.path.exists(train_affinityFile) else pred_affinity(model, chem_tokenizer, prot_tokenizer, 
                                                                                                                    df_train, df_protData,
                                                                                                                    train_affinityFile, config)
        
        if os.path.exists(test_path):
            df_test = pd.read_csv(test_path)
            test_affinity = pd.read_csv(test_affinityFile) if os.path.exists(test_affinityFile) else pred_affinity(model, chem_tokenizer, prot_tokenizer, 
                                                                                                                    df_test, df_protData,
                                                                                                                    test_affinityFile, config)
            
            df_trainData, train_scaler, df_trainAff, df_trainRdkitLabel = load_chemdata(df_train, train_affinity, config.feature_type, config.scale, config.augmentation)
            df_trainfeatures = pd.concat([df_trainAff.iloc[:, 1:], df_trainData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else df_trainAff.iloc[:, 1:]
            train_labels = df_trainData["Clint"]

            df_testData, test_scaler, df_testAff, df_testRdkitLabel = load_chemdata(df_test, test_affinity, config.feature_type, config.scale)
            df_testfeatures = pd.concat([df_testAff.iloc[:, 1:], df_testData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else df_testAff.iloc[:, 1:]
            test_labels = df_testData["Clint"]

        else:
            df_test = df_train.sample(frac=config.test_rate, random_state=config.num_seed)
            df_train = df_train.drop(df_test.index)
            df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

            df_trainData, train_scaler, df_trainAff, df_trainRdkitLabel = load_chemdata(df_train, train_affinity, config.feature_type, config.scale, config.augmentation)
            df_trainfeatures = pd.concat([df_trainAff.iloc[:, 1:], df_trainData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else df_trainAff.iloc[:, 1:]
            train_labels = df_trainData["Clint"]

            df_testData, test_scaler, df_testAff, df_testRdkitLabel = load_chemdata(df_test, train_affinity, config.feature_type, config.scale)
            df_testfeatures = pd.concat([df_testAff.iloc[:, 1:], df_testData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else df_testAff.iloc[:, 1:]
            test_labels = df_testData["Clint"]
        

        checkpoint_callback = ModelCheckpoint(f"{config.model_type}_{config.feature_type}_{config.num_seed}", save_top_k=1, monitor="valid_MSE", mode="min")

        trainer = pl.Trainer(
            devices=config.gpu_id,
            max_epochs=config.max_epoch,
            logger = logger,
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=0,
            accelerator='gpu',
            strategy='dp'           
        )

        datacols = list(df_trainData.columns[1:])
        feature_length = len(df_trainfeatures.columns)
        datamodule = clearanceDatamodule(chem_tokenizer, 
                                         df_trainData, df_testData,
                                         df_trainfeatures, df_testfeatures,
                                         train_labels, test_labels,
                                         train_scaler, test_scaler, 
                                         config.chem_max, config.batch_size, config.num_workers)
        ## -- clearance model declaration -- ##
        if config.model_type == "decoder":
            clearance_model = clearanceDecoderModel(config.lr, config.dropout, config.chem_model, feature_length,
                                                    df_trainData[datacols], df_testData[datacols])
        elif config.model_type == "encoder":
            clearance_model = clearanceEncoderModel(config.lr, config.dropout, config.chem_model, feature_length,
                                                    df_trainData[datacols], df_testData[datacols])
        else:
            clearance_model = clearanceMLPModel(config.lr, config.dropout, config.chem_model, feature_length,
                                                df_trainData[datacols], df_testData[datacols], config.act_func)
            # clearance_model = clearanceEncoderModel(config.lr, config.dropout, config.chem_model,
            #                                         df_trainData[datacols], df_testData[datacols])
            
        trainer.fit(clearance_model, datamodule)
        
        clearance_model.eval()
        trainer.test(clearance_model, datamodule)

        ## -- connecting result data to rdkit feature range -- ##
        dict_result = {"SMILES": df_testData["SMILES"], "predict": clearance_model.test_result['preds'], "Clint": clearance_model.test_result['labels']}
        df_result = pd.DataFrame.from_dict(dict_result)

        df_result = pd.concat([df_result, df_testRdkitLabel], axis=1)

        resultData_path = f"{project_name}_{config.checkpoint_name}_{config.feature_type}"
        save_result(config, resultData_path, df_result, clearance_model.test_log, df_trainData, df_testData)
    
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