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


def predict_BindingAffinity(model, chem_tokenizer, prot_tokenizer, df_chem, df_prot, pred_file, config):

    ## -- predict datset -- ##
    trainer = pl.Trainer(accelerator='gpu',
                        strategy='dp', 
                        devices=args.gpus)

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
    
    
def train_function(config, logger):
    pl.seed_everything(seed=config.num_seed)

    ## -- tokenized dataset declaration -- ##
    chem_tokenizer = AutoTokenizer.from_pretrained(config.chem_model)
    prot_tokenizer = AutoTokenizer.from_pretrained(config.prot_model)

    ## -- load protain and chemical dataset -- ##
    prot_path = config.prot_file
    df_protData = load_protdata(prot_path, config.extend_protType)
    
    train_path =  os.path.join(config.data_folder, config.train_dataType, config.train_file)
    test_path = os.path.join(config.data_folder, config.train_dataType, config.test_file)

    ## -- binding affinity predict file path setting -- ##
    checkpoint_name = str(config.checkpoint_name).lower()
    affinity_path = os.path.join(config.affinity_folder, checkpoint_name, config.train_dataType)
    train_affinityFile, test_affinityFile = f"{affinity_path}/{config.train_file}", f"{affinity_path}/{config.test_file}"

    ## -- binding affinity model declaration -- ##
    if not os.path.exists(train_affinityFile) or not os.path.exists(test_affinityFile):
        if not os.path.exists(affinity_path):
            os.makedirs(affinity_path)

        ckpt_file = os.path.join("./checkpoint", f"{checkpoint_name}.ckpt")
        if checkpoint_name == "regression_bindingDB":
            model = BindingAffinityModel.load_from_checkpoint(ckpt_file)
        else:
            model = BiomarkerModel.load_from_checkpoint(ckpt_file)
        
    ## -- load SMILES dataset-- ##
    df_train = pd.read_csv(train_path)

    ## -- Predict or load Binding Affinity data-- ##
    train_affinity = pd.read_csv(train_affinityFile) if os.path.exists(train_affinityFile) else predict_BindingAffinity(model, chem_tokenizer, prot_tokenizer, 
                                                                                                                df_train, df_protData,
                                                                                                                train_affinityFile, config)
        
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        test_affinity = pd.read_csv(test_affinityFile) if os.path.exists(test_affinityFile) else predict_BindingAffinity(model, chem_tokenizer, prot_tokenizer, 
                                                                                                                df_test, df_protData,
                                                                                                                test_affinityFile, config)

                                                                                                                ## -- test dataset load type setting -- ##
        testData_normtype = "all" if config.sampling_type.lower() == "all" else config.feature_type

        ## -- data preprocessing normalization, rdkit range setting -- ##
        df_trainData, train_scaler, df_sampledtrain = norm_dataset(df_train, config.feature_type, config.scale, config.augmentation)
        df_trainAff = get_affinitydata(df_sampledtrain, train_affinity, config.augmentation)
        df_trainfeatures = pd.concat([df_trainAff.iloc[:, 1:], df_trainData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else df_trainAff.iloc[:, 1:]

        df_testData, test_scaler, df_sampledtest = norm_dataset(df_test, testData_normtype, config.scale)
        df_testAff = get_affinitydata(df_sampledtest, test_affinity)
        df_testRdkitLabel = get_rdkitlabel(df_sampledtest)
        df_testfeatures = pd.concat([df_testAff.iloc[:, 1:], df_testData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else df_testAff.iloc[:, 1:]
        
        train_labels = df_trainData["Clint"]
        test_labels = df_testData["Clint"]

    else:
        df_test = df_train.sample(frac=config.test_rate, random_state=config.num_seed)
        df_train = df_train.drop(df_test.index)
        df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

        testData_normtype = "all" if config.sampling_type.lower() == "all" else config.feature_type

        ## -- data preprocessing normalization, rdkit range setting -- ##
        df_trainData, train_scaler, df_sampledtrain = norm_dataset(df_train, config.feature_type, config.scale, config.augmentation)
        df_trainAff = get_affinitydata(df_sampledtrain, train_affinity, config.augmentation)
        df_trainfeatures = pd.concat([df_trainAff.iloc[:, 1:], df_trainData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else df_trainAff.iloc[:, 1:]

        df_testData, test_scaler, df_sampledtest = norm_dataset(df_test, testData_normtype, config.scale)
        df_testAff = get_affinitydata(df_sampledtest, train_affinity)
        df_testRdkitLabel = get_rdkitlabel(df_sampledtest)
        df_testfeatures = pd.concat([df_testAff.iloc[:, 1:], df_testData.iloc[:, 2:]], axis=1) if config.sampling_type.lower() != "default" else df_testAff.iloc[:, 1:]
        
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
    # test_datacols = list(df_testData.columns[1:])
    
    ## -- feature column name length -- ##
    feature_length = len(df_trainfeatures.columns)    
    
    ## -- datamodule -- ##
    datamodule = clearanceDatamodule(chem_tokenizer, config, 
                                        df_trainData=df_trainData, 
                                        df_trainLabel=train_labels, 
                                        train_scaler=train_scaler,
                                        df_trainfeatures=df_trainfeatures,)
    
    ## -- clearance model declaration -- ##
    if config.model_type.lower() == "decoder":
        clearance_model = clearanceDecoderModel(config.lr, config.dropout, config.chem_model, feature_length,
                                                df_trainData[train_datacols], df_testData[test_datacols])
    elif config.model_type.lower() == "encoder":
        clearance_model = clearanceEncoderModel(config.lr, config.dropout, config.chem_model, feature_length,
                                                df_trainData[train_datacols], df_testData[test_datacols])
    else:
        clearance_model = clearanceMLPModel(config.lr, config.dropout, config.chem_model, feature_length,
                                            config.act_func[0])
        
    trainer.fit(clearance_model, datamodule)
    
    # clearance_model.eval()
    # trainer.test(clearance_model, datamodule)

    ## -- connecting result data to rdkit feature range -- ##
    dict_result = {"SMILES": df_testData["SMILES"], "predict": clearance_model.test_result['preds'], "Clint": clearance_model.test_result['labels']}
    df_result = pd.DataFrame.from_dict(dict_result)

    df_result = pd.concat([df_result, df_testRdkitLabel[df_testRdkitLabel.columns[1:]]], axis=1)

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
        resultData_path = f"{path_name}_Commonsetting/"

        if run_type == TENSOR_SWEEP:
            list_sweep_options = generate_combinations(config)
            metric_columns = ["seed", "model_type", "chem_model", "checkpoint_name", "feature_type", "test_rate", "data_augmentation", "protainData_extended", "scale", \
                                "valid_MSE", "valid_MAE", "valid_r2", "valid_rm2", "valid_CI", "test_MSE", "test_MAE", "test_r2", "test_rm2", "test_CI"]
            df_metricData = pd.DataFrame(columns=metric_columns)
            
            for options in list_sweep_options:
                options = DictX(options)
                df_result, valid_log, test_log, df_trainData, df_testData = train_function(options, logger)
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
            df_result, valid_log, test_log, df_trainData, df_testData = train_function(config, logger)
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
    parser.add_argument("--gpus", help="gpus", type=str, default="0,1")
    parser.add_argument("--config", help="config_file", type=str, default="1_config_sweep.json")
    args = parser.parse_args()

    run_type = TENSOR
    
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