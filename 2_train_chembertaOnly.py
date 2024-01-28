import os, copy
import pandas as pd
import numpy as np

from argparse import ArgumentParser

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from transformers import AutoTokenizer

import wandb

from utils.utils import DictX, load_hparams, save_result, generate_combinations
from utils.preprocess import load_protdata, load_chemdata, norm_dataset, get_rdkitlabel, get_affinitydata

from modules.dataloader import BiomakerDataModule, clearanceDatamodule
from modules.biomarker import BiomarkerModel, BindingAffinityModel
from modules.clearance import clearanceMLPModel, clearanceDecoderModel, clearanceEncoderModel


## -- Logger Type -- ##
TENSOR = 0
TENSOR_SWEEP = 1
WANDB = 2
SWEEP = 3


def chemBERTaOnly_function(config, logger):
    pl.seed_everything(seed=config.num_seed)
    chem_tokenizer = AutoTokenizer.from_pretrained(config.chem_model)
    
    ## -- dataset path -- ##
    train_path =  os.path.join(config.data_folder, config.train_dataType, config.train_file)
    test_path = os.path.join(config.data_folder, config.train_dataType, config.test_file)
    
    ## -- dataset load -- ##
    df_train = pd.read_csv(train_path)
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        
        ## -- test dataset load type setting -- ##
        testData_normtype = "all" if config.sampling_type.lower() == "all" else config.feature_type
        
        ## -- data preprocessing normalization, rdkit range setting -- ##
        df_trainData, train_scaler, _ = norm_dataset(df_train, config.feature_type, config.test_scaler)
        # df_trainRdkitLabel = get_rdkitlabel(df_trainData, config.augmentation)
        df_trainfeatures = df_trainData.iloc[:, 2:] if config.feature_type.lower() != "default" else None
        
        df_testData, test_scaler, _ = norm_dataset(df_test, testData_normtype, config.scale)
        df_testRdkitLabel = get_rdkitlabel(df_testData)
        df_testfeatures = df_testData.iloc[:, 2:] if config.feature_type.lower() != "default" else None
        
        ## -- label setting -- ##
        train_labels = df_trainData["Clint"]
        test_labels = df_testData["Clint"]
        
    else:
        ## -- dataset load -- ##
        df_test = df_train.sample(frac=config.test_rate, random_state=config.num_seed)
        df_train = df_train.drop(df_test.index)
        df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

        ## -- test dataset load type setting -- ##
        testData_normtype = "all" if config.sampling_type.lower() == "all" else config.feature_type
        
        ## -- data preprocessing normalization, rdkit range setting -- ##
        df_trainData, train_scaler, _ = norm_dataset(df_train, config.feature_type, config.scale)
        # df_trainRdkitLabel = get_rdkitlabel(df_trainData, config.augmentation)
        df_trainfeatures = df_trainData.iloc[:, 2:] if config.feature_type.lower() != "default" else None
        
        df_testData, test_scaler, _ = norm_dataset(df_test, testData_normtype, config.scale)
        df_testRdkitLabel = get_rdkitlabel(df_testData)
        df_testfeatures = df_testData.iloc[:, 2:] if config.feature_type.lower() != "default" else None
        
        ## -- label setting -- ##
        train_labels = df_trainData["Clint"]
        test_labels = df_testData["Clint"]
    
    option_name = ''
    if config.extend_protType == False:
        option_name += "Extend"
    if config.augmentation == False:
        option_name += "Aug"
    if option_name != '':
        option_name = "_Non"+option_name

    checkpoint_callback = ModelCheckpoint(filename=f"{config.model_type}_{config.feature_type}_{config.num_seed}{option_name}", 
                                          save_top_k=1, 
                                          monitor="valid_MSE", 
                                          mode="min")
    
    ## -- pytorch lightning trainer setting -- ##
    trainer = pl.Trainer(
        devices=args.gpus,
        max_epochs=config.max_epoch,
        logger = logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accelerator='gpu',
        strategy='dp'           
    )

    ## -- train, test column name setting -- ##
    train_datacols = list(df_trainData.columns[1:])
    test_datacols = list(df_testData.columns[1:])
    
    ## -- feature column name length -- ##
    feature_length = 0
    if df_trainfeatures is not None:
        feature_length = len(df_trainfeatures.columns)
    
    ## -- datamodule -- ##
    datamodule = clearanceDatamodule(chem_tokenizer, config, 
                                        df_trainData, df_testData,
                                        train_labels, test_labels,
                                        train_scaler, test_scaler,
                                        df_trainfeatures, df_testfeatures)
    
    ## -- model -- ##
    clearance_model = clearanceMLPModel(config.lr, config.dropout, config.chem_model, feature_length,
                                            df_trainData[train_datacols], df_testData[test_datacols], config.act_func)
        
    trainer.fit(clearance_model, datamodule)
    
    clearance_model.eval()
    trainer.test(clearance_model, datamodule)

    ## -- connecting result data to rdkit feature range -- ##
    dict_result = {"SMILES": df_testData["SMILES"], "predict": clearance_model.test_result['preds'], "Clint": clearance_model.test_result['labels']}
    df_result = pd.DataFrame.from_dict(dict_result)
    df_result = pd.concat([df_result, df_testRdkitLabel], axis=1)

    return df_result, clearance_model.valid_log, clearance_model.test_log, df_trainData, df_testData

    
def main(config=None):
    try: 
        ##-- hyper param config file Load --##
        if run_type == TENSOR or run_type == TENSOR_SWEEP:
            config = DictX(config)
        else:
            if config is not None:
                wandb.init(config=config, project=project_name)
            else:
                wandb.init(settings=wandb.Settings(console='off'))  

            config = wandb.config

        if not os.path.exists(config.log_path):
            os.makedirs(config.log_path)

        if run_type == TENSOR or run_type == TENSOR_SWEEP:
            logger = TensorBoardLogger(save_dir=config.log_path, name=project_name)
        else:
            logger = WandbLogger(project=project_name, save_dir=config.log_path)

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        outputs_list = list()
        path_name, _ = os.path.splitext(args.config)
        resultData_path = f"{path_name}_latest/"

        if run_type == TENSOR_SWEEP:
            list_sweep_options = generate_combinations(config)
            metric_columns = ["seed", "model_type", "chem_model", "checkpoint_name", "feature_type", "test_rate", "data_augmentation", "protainData_extended", "scale", \
                                "valid_MSE", "valid_MAE", "valid_r2", "valid_rm2", "valid_CI", "test_MSE", "test_MAE", "test_r2", "test_rm2", "test_CI"]
            df_metricData = pd.DataFrame(columns=metric_columns)
            
            for options in list_sweep_options:
                options = DictX(options)
                df_result, valid_log, test_log, df_trainData, df_testData = chemBERTaOnly_function(options, logger)
                outputs_list.append([df_result, valid_log, test_log, df_trainData, df_testData, options])

                ## -- metric csv save -- ##
                df_metricData.loc[len(df_metricData)] = [options.num_seed, options.model_type, options.chem_model, options.checkpoint_name, options.feature_type, options.test_rate, \
                                                        options.augmentation, options.extend_protType, options.scale, \
                                                        round(float(valid_log['rmse']), 5), round(float(valid_log['MAE']), 5), round(float(valid_log['r2']), 5), \
                                                        round(float(valid_log['rm2']), 5), round(float(valid_log['ci']), 5), \
                                                        round(float(test_log['rmse']), 5), round(float(test_log['MAE']), 5), round(float(test_log['r2']), 5), \
                                                        round(float(test_log['rm2']), 5), round(float(test_log['ci']), 5)]
                
                
                evaluation_path = f"results/{resultData_path}/"
                if not os.path.exists(evaluation_path):
                    os.makedirs(evaluation_path)

                df_metricData.to_csv(evaluation_path + f"result_metrics.csv", index=False)

        else:
            df_result, valid_log, test_log, df_trainData, df_testData = chemBERTaOnly_function(config, logger)
            outputs_list.append([df_result, valid_log, test_log, df_trainData, df_testData, config])
                
        print("---------------Chart Boundary setting!!--------------------")
        variance_boundary = 0.5
        for index, output in enumerate(outputs_list):
            variance_list =[abs(data - output[0]['Clint'][idx])/output[0]['predict'].mean() for idx, data in enumerate(output[0]['predict'])]
            output[0]["variance"] = variance_list
            
            if index ==  0:
                df_duplPos = output[0][output[0]["variance"] > variance_boundary]
            else:
                duplPos_temp = output[0][output[0]["variance"] > variance_boundary]
                df_duplPos = df_duplPos[df_duplPos["SMILES"].isin(duplPos_temp["SMILES"])]
            

        for output_result, output_devlog, output_testlog, output_traindata, output_testdata, output_options in outputs_list:
            print("---------------Save prediction Results!!--------------------")
            save_result(output_options, resultData_path, output_result, output_devlog, output_testlog, output_traindata, output_testdata, df_duplPos)
    
    except Exception as e:
        print(e)



if __name__ == "__main__":
    #-- wandb Sweep Hyper Param Tuning --##
    parser = ArgumentParser()
    parser.add_argument("--gpus", help="gpus", type=str, default="2,3,4,5,6,7")
    parser.add_argument("--config", help="config_file", type=str, default="1_config_chembertaOnly_sweep.json")
    args = parser.parse_args()

    run_type = TENSOR_SWEEP
    
    if run_type == SWEEP:
        config = load_hparams('config/sweep/1_config_train.json')
        project_name = config["name"]
        sweep_id = wandb.sweep(config, project=project_name)
        wandb.agent(sweep_id, main)
    
    else:
        config_file = f'config/{args.config}' if run_type == TENSOR_SWEEP else 'config/1_config_train.json'
        config = load_hparams(config_file)
        project_name = config["name"]
        main(config)