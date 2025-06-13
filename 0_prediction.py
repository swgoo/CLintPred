import os, json
import copy

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from argparse import ArgumentParser

from utils.utils import DictX
from utils.preprocess import load_protdata, get_affinitydata, norm_dataset, get_rdkitlabel
from modules.dataloader import BiomakerDataModule, clearanceDatamodule
from modules.biomarker import BiomarkerModel, BindingAffinityModel
from modules.clearance import clearanceMLPModel, clearanceDecoderModel, clearanceEncoderModel


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

def predict_BindingAffinity(model, chem_tokenizer, prot_tokenizer, df_chem, df_prot, result_file, config):

    trainer = pl.Trainer(accelerator='gpu', devices=config.gpus)

    train_obj = chem_tokenizer(np.array(df_chem["SMILES"]).tolist(), padding='max_length', 
                            max_length=config.chem_max, truncation=True, return_tensors="pt")
    prot_obj = prot_tokenizer(np.array(df_prot["aas"]).tolist(), padding='max_length', 
                            max_length=config.prot_max, truncation=True, return_tensors="pt")
    
    affinity_dm = BiomakerDataModule(train_obj, prot_obj, config)
    pred_train = trainer.predict(model, datamodule=affinity_dm)

    affinity_columns = list(df_prot["name"])
    affinities = concat_predData(pred_train, affinity_columns)
    affinities = pd.concat([df_chem["SMILES"], affinities], axis=1)

    affinities.to_csv(result_file, index=False)

    return affinities

def predict_clearance(config):
    pl.seed_everything(seed=config.num_seed)

    chem_tokenizer = AutoTokenizer.from_pretrained(config.chem_model)
    prot_tokenizer = AutoTokenizer.from_pretrained(config.prot_model)

    prot_path = config.prot_file
    df_protData = load_protdata(prot_path, config.extend_protType)

    if config.input_file != None:
        predict_path = config.input_file
        df_predict = pd.read_csv(predict_path)
    else:
        smiles_list = ["O=C(C1=CC=C(C(N(CC=C)C(N2CC3=CC=CC=C3F)=O)=O)C2=C1)NCC4=CC=CC=C4", "O=C(N1CC2=CC=C(C(NC)=O)C=C2)C3=CC(OC)=C(OC)C=C3N(CC4=CC(F)=CC=C4)C1=O"]
        df_predict = pd.DataFrame(smiles_list, columns=["SMILES"])

    checkpoint_name = str(config.checkpoint_name).lower()
    affinity_path = os.path.join(config.affinity_folder, checkpoint_name, config.train_dataType)
    result_file = f"{affinity_path}/{config.pred_result}"

    if not os.path.exists(result_file):
        if not os.path.exists(affinity_path):
            os.makedirs(affinity_path)

        ckpt_file = os.path.join("./checkpoint", f"{checkpoint_name}.ckpt")
        if checkpoint_name == "regression_bindingDB":
            model = BindingAffinityModel.load_from_checkpoint(ckpt_file)
        else:
            model = BiomarkerModel.load_from_checkpoint(ckpt_file)
        
    predict_affinity = pd.read_csv(result_file) if os.path.exists(result_file) else predict_BindingAffinity(model, chem_tokenizer, prot_tokenizer, 
                                                                                                                df_predict, df_protData,
                                                                                                                result_file, config)

    df_sampledpredict = copy.deepcopy(df_predict)
    ## -- data preprocessing normalization, rdkit range setting -- ##
    # df_predictData, predict_scaler, df_sampledpredict = norm_dataset(df_predict, config.feature_type, config.scale, config.augmentation)
    df_predictAff = get_affinitydata(df_sampledpredict, predict_affinity, config.augmentation)
    df_predictfeatures = pd.concat([df_predictAff.iloc[:, 1:], df_predict.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else df_predictAff.iloc[:, 1:]

    # predict_rdkit_label = get_rdkitlabel(df_sampledpredict)
    
    feature_length = len(df_predictfeatures.columns)
    # feature_length = 0

    if config.model_type.lower() == "decoder":
        clearance_model = clearanceDecoderModel(config.lr, config.dropout, config.chem_model, feature_length,
                                                df_predictData, df_predictData)
    elif config.model_type.lower() == "encoder":
        clearance_model = clearanceEncoderModel(config.lr, config.dropout, config.chem_model, feature_length,
                                                df_predictData, df_predictData)
    else:
        # clearance_model = clearanceMLPModel(config.lr, config.dropout, config.chem_model, feature_length,
        #                                     config.act_func)
        # clearance_model = clearanceMLPModel.load_from_checkpoint('./checkpoint/mlp_all_120.ckpt')

        # .pt 파일에서 state dict 로드
        state_dict = torch.load('./checkpoint/mlp_all_120.pt')
        
        # hparams.json 파일에서 하이퍼파라미터 로드
        with open('./checkpoint/hparams.json', 'r') as f:
            hparams = json.load(f)
            
        clearance_model = clearanceMLPModel(
            lr=hparams['lr'],
            dropout=hparams['dropout'],
            model_name=hparams['model_name'], 
            feature_length=hparams['feature_length'],
            act_func=hparams['act_func'],
        )
        
        # 모델에 state dict 적용
        clearance_model.load_state_dict(state_dict)
        
    
    datamodule = clearanceDatamodule(chem_tokenizer, config, 
                                     df_predictData=df_predict,
                                     df_predictfeatures=df_predictfeatures,
                                     )
    
    trainer = pl.Trainer(accelerator='gpu', devices=config.gpus)
    # clearance_model.eval()
    predictions = trainer.predict(clearance_model, datamodule)
    
    pred_1d = [item for sublist in predictions for item in sublist]
    print(f"Prediction length: {len(pred_1d)}")

    dict_result = {"SMILES": df_predict["SMILES"], "predict": pred_1d}
    df_result = pd.DataFrame.from_dict(dict_result)
    df_result = pd.concat([df_result], axis=1)

    return df_result

def main(config):
    try:
        config = DictX(config)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        predicted_clearance = predict_clearance(config)
        print(f"Clearance Prediction Results :{predicted_clearance}")
    
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--name", help="project name", type=str, default="ClintPreds_Mod")
    parser.add_argument("--chem_model", help="chemistry model", type=str, default="seyonec/PubChem10M_SMILES_BPE_450k")
    parser.add_argument("--prot_model", help="protein model", type=str, default="Rostlab/prot_bert_bfd")

    parser.add_argument("--num_seed", help="random seed number", type=int, default=120)
    parser.add_argument("--gpus", help="gpu ids", type=str, default="0,")
    parser.add_argument("--data_folder", help="data folder path", type=str, default="./dataset/fin/")
    parser.add_argument("--affinity_folder", help="affinity folder path", type=str, default="./dataset/affinity_fin/")
    
    parser.add_argument("--input_file", help="prediction data file", type=str, default="./dataset/fin/newdata_sort/test.csv")
    parser.add_argument("--prot_file", help="protein data file path", type=str, default="./dataset/liver_aas.csv")
    
    parser.add_argument("--checkpoint_name", help="checkpoint file name", type=str, default="biomarker")
    parser.add_argument("--train_dataType", help="training data type", type=str, default="newdata_nonvivo_sort")
    parser.add_argument("--model_type", help="model type (decoder, encoder, mlp)", type=str, default="mlp")

    parser.add_argument("--chem_max", help="maximum length for chem tokenizer", type=int, default=510)
    parser.add_argument("--prot_max", help="maximum length for prot tokenizer", type=int, default=545)

    parser.add_argument("--extend_protType", help="extend protein type", default=True)
    parser.add_argument("--scale", help="scale", type=bool, default=False)
    parser.add_argument("--augmentation", help="data augmentation", default=False)
    parser.add_argument("--sampling_type", help="sampling type", type=str, default="all")
    parser.add_argument("--feature_type", help="feature type", type=str, default="all")
    parser.add_argument("--test_rate", help="test data rate", type=float, default=0.2)

    parser.add_argument("--log_path", help="log path", type=str, default="./log/ClintPreds_Mod")
    parser.add_argument("--max_epoch", help="maximum number of epochs", type=int, default=1)
    parser.add_argument("--batch_size", help="batch size", type=int, default=16)
    parser.add_argument("--num_workers", help="number of workers", type=int, default=0)

    parser.add_argument("--act_func", help="activation function", type=str, default="relu")
    parser.add_argument("--lr", help="learning rate", type=float, default=5e-5)
    parser.add_argument("--dropout", help="dropout rate", type=float, default=0.1)

    parser.add_argument("--pred_result", help="file for result data", type=str, default="./pred_result")

    args = parser.parse_args()

    config = vars(args)
    main(config)
