import os, json
import pandas as pd

from itertools import product
from easydict import EasyDict

from utils.draw_plot import draw_plot, draw_boxplot, draw_boundaryplot


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


def load_hparams(file_path):
    hparams = EasyDict()
    with open(file_path, 'r') as f:
        hparams = json.load(f)
    return hparams


def generate_combinations(dic):
    keys = dic.keys()
    values = (dic[key] if isinstance(dic[key], list) else [dic[key]] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    return combinations


def save_result(config, result_dir, df_result:pd.DataFrame, output_devlog:dict, output_testlog:dict, df_train:pd.DataFrame = None, df_test:pd.DataFrame = None, df_duplData:pd.DataFrame = None):
    result_name = f"{config.model_type}_{config.train_dataType}_seed{config.num_seed}_r2_{output_testlog['r2']:.3f}"
    result_path = f"results/{result_dir}/{config.feature_type}_{result_name}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if df_train is not None:
        df_train.to_csv(f'{result_path}/train.csv', index=False) 
    if df_test is not None:
        df_test.to_csv(f'{result_path}/test.csv', index=False)

    result_file = f'{result_path}/results.csv'
    df_result.to_csv(result_file, index=False)
    print('-----------------Save prediction result ----------------------')

    save_file = os.path.join(result_path, "figure.png")
    slop = draw_boundaryplot(df_result, save_file, df_duplData)

    submit_config = {"model_type":config.model_type, "chem_model":config.chem_model,
                     "checkpoint_name": config.checkpoint_name, "feature_type": config.feature_type, "test_rate":config.test_rate,
                     "data_augmentation" : config.augmentation, "protainData_extended" : config.extend_protType,
                     "scale": config.scale, "chem_max" : config.chem_max, "batch_size" : config.batch_size,
                     "lr":config.lr, "dropout":config.dropout, "max_epoch": config.max_epoch, "test_sampling_type": config.sampling_type,
                     "valid_MSE": str(output_devlog['rmse']), "valid_MAE": str(output_devlog['MAE']), "valid_r2": str(output_devlog['r2']), 
                     "valid_rm2": str(output_devlog['rm2']), "valid_CI": str(output_devlog['ci']), 
                     "test_MSE": str(output_testlog['rmse']), "test_MAE": str(output_testlog['MAE']), "test_r2": str(output_testlog['r2']), 
                     "test_rm2": str(output_testlog['rm2']), "test_CI": str(output_testlog['ci']), "slop": str(slop)}

    with open(os.path.join(result_path,"config.json"), 'w', encoding='utf-8') as mf:
        json.dump(submit_config, mf, indent='\t')

    print('Done.')

    # draw_boxplot(result_file, save_path)
    
