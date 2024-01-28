import copy
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

## -- protatin sequence type -- ##
prot_type = ["AAS_CYP9", "UGT_TYPE", "SULTs"]

## --  chemical Compound Feature type -- ##
features_columns = ["logP", "Fup"]
features_mwlogp_columns = ["logP_rdkit", "Fup"]
rdkit_columns = ["MW_rdkit", "HBD_rdkit", "HBA_rdkit", "NRB_rdkit", "RF_rdkit", "PSA_rdkit"]
default_columns = ["SMILES", "Clint"]

rangeLabel_col = ["MW_range", "PSA_range", "NRB_range", "HBA_range", "HBD_range", "LogP_range"]


def load_protdata(file_path:str, extend_protType:bool = False):
    df_protData = pd.read_csv(file_path)

    if not extend_protType:
        df_protData= df_protData[df_protData["type"] == prot_type[0]]
    
    df_protData["aas"]  = [' '.join(list(aas)) for aas in df_protData["aas"]]

    return df_protData


def norm_dataset(df_loadData:pd.DataFrame, feature_type:str = "default", scale = True, augmentation = False):
    df_loadData = df_loadData[(df_loadData['Clint'] <= 500)]
    df_loadData['Clint'] = np.log1p(df_loadData['Clint'])
    
    ## -- Common setting preprecessing -- ##
    df_loadData = df_loadData[(df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
    df_loadData = df_loadData[(df_loadData['logP'] >= -2.0) & (df_loadData['logP'] <= 6.0)]

    df_loadData = df_loadData[(df_loadData['MW_rdkit'] < 1000.0)]
    df_loadData = df_loadData[(df_loadData['HBA_rdkit'] <= 10.0)]
    df_loadData = df_loadData[(df_loadData['HBD_rdkit'] <= 5.0)]
    df_loadData = df_loadData.drop(df_loadData[df_loadData["logP"] == "None"].index).reset_index(drop=True)

    if feature_type.lower() == "default":
        df_loadData = df_loadData[default_columns].dropna(axis=0).reset_index(drop=True)

    elif feature_type.lower() == "features":
        # df_loadData = df_loadData[(df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        # df_loadData = df_loadData[(df_loadData['logP'] >= -2.0) & (df_loadData['logP'] <= 6.0)]
        # df_loadData = df_loadData.drop(df_loadData[df_loadData["logP"] == "None"].index).reset_index(drop=True)
        
        df_loadData = df_loadData[default_columns + features_columns].dropna(axis=0).reset_index(drop=True)
        

    elif feature_type.lower() == "features_mwlogp":
        # df_loadData = df_loadData[(df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        # df_loadData = df_loadData[(df_loadData['logP_rdkit'] >= -2.0) & (df_loadData['logP_rdkit'] <= 6.0)]
        # df_loadData = df_loadData.drop(df_loadData[df_loadData["logP_rdkit"] == "None"].index).reset_index(drop=True)
        
        df_loadData = df_loadData[default_columns + features_mwlogp_columns].dropna(axis=0).reset_index(drop=True)
        
    
    elif feature_type.lower() == "rdkit":
        # df_loadData = df_loadData[(df_loadData['MW_rdkit'] < 1000.0)]
        # df_loadData = df_loadData[(df_loadData['HBA_rdkit'] <= 10.0)]
        # df_loadData = df_loadData[(df_loadData['HBD_rdkit'] <= 5.0)]
        
        df_loadData = df_loadData[default_columns + rdkit_columns].dropna(axis=0).reset_index(drop=True)
    
    elif feature_type.lower() == "all_mwlogp":
        # df_loadData = df_loadData[(df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        # df_loadData = df_loadData[(df_loadData['logP_rdkit'] >= -2.0) & (df_loadData['logP_rdkit'] <= 6.0)]

        # df_loadData = df_loadData[(df_loadData['MW_rdkit'] < 1000.0)]
        # df_loadData = df_loadData[(df_loadData['HBA_rdkit'] <= 10.0)]
        # df_loadData = df_loadData[(df_loadData['HBD_rdkit'] <= 5.0)]
        # df_loadData = df_loadData.drop(df_loadData[df_loadData["logP_rdkit"] == "None"].index).reset_index(drop=True)
        
        df_loadData = df_loadData[default_columns + features_mwlogp_columns + rdkit_columns].dropna(axis=0).reset_index(drop=True)
        

    else:
        # df_loadData = df_loadData[(df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        # df_loadData = df_loadData[(df_loadData['logP'] >= -2.0) & (df_loadData['logP'] <= 6.0)]

        # df_loadData = df_loadData[(df_loadData['MW_rdkit'] < 1000.0)]
        # df_loadData = df_loadData[(df_loadData['HBA_rdkit'] <= 10.0)]
        # df_loadData = df_loadData[(df_loadData['HBD_rdkit'] <= 5.0)]
        # df_loadData = df_loadData.drop(df_loadData[df_loadData["logP"] == "None"].index).reset_index(drop=True)
        
        df_loadData = df_loadData[default_columns + features_columns + rdkit_columns].dropna(axis=0).reset_index(drop=True)
        
        
    ## -- log scaled dataset augmentation--  ##
    df_sampledData = copy.deepcopy(df_loadData)
    if augmentation is True:
        df_augmentedData = df_loadData[(df_loadData['Clint'] >= 0.5)]
        df_loadData = pd.concat([df_loadData, df_augmentedData], axis=0).reset_index(drop = True)
    
    if scale:
        datacols = list(df_loadData.columns[1:])
        data_scaler = MinMaxScaler()
    
        df_loadData[datacols] = df_loadData[datacols].astype('float')
        scaled_data = data_scaler.fit_transform(df_loadData[datacols])
        df_loadData[datacols] = scaled_data
        
    return df_loadData, data_scaler, df_sampledData

def norm_func(df_dataset:pd.DataFrame, scale:bool):
    # Select features (columns) to be involved intro training and predictions
    column_Length = df_dataset.shape[1]
    cols = list(df_dataset)[1:column_Length]

    # To Numpy and Delete TimeStep
    features = df_dataset[cols]
    # features = features.astype(float)
    data_mean, data_std = 0, 1

    # To Scaling
    if scale:
        data_mean = features.mean(axis=0)
        data_std = features.std(axis=0)
        features = (features-data_mean)/data_std
        # dataset_train = features.values
        df_dataset[cols] = features

    return df_dataset, data_mean, data_std


def get_affinitydata(df_loadData:pd.DataFrame, train_affinity:pd.DataFrame, augmentation = False):
    df_affinityData = train_affinity[train_affinity["SMILES"].isin(list(df_loadData["SMILES"]))].reset_index(drop = True)
    
    ## -- log scaled dataset augmentation--  ##
    if augmentation is True:
        df_augmentedData = df_loadData[(df_loadData['Clint'] >= 0.5)]
        df_augmentedFeature = df_affinityData[df_affinityData["SMILES"].isin(list(df_augmentedData["SMILES"]))]
        df_affinityData = pd.concat([df_affinityData, df_augmentedFeature], axis=0).reset_index(drop = True)
        
    return df_affinityData


def get_rdkitlabel(df_loadData:pd.DataFrame, augmentation = False):
    df_rdkitLabel = rdkit_rangeLabel(df_loadData)
    
    if augmentation is True:
        df_augmentedData = df_loadData[(df_loadData['Clint'] >= 0.5)]
        df_augmentedrdkitLabel = df_rdkitLabel[df_rdkitLabel["SMILES"].isin(list(df_augmentedData["SMILES"]))]
        df_rdkitLabel = pd.concat([df_rdkitLabel, df_augmentedrdkitLabel], axis=0).reset_index(drop = True)
        
    return df_rdkitLabel


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

    elif feature_type.lower() == "features_mwlogp":
        df_loadData = df_loadData[(df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        df_loadData = df_loadData[(df_loadData['logP_rdkit'] >= -2.0) & (df_loadData['logP_rdkit'] <= 6.0)]
        df_loadData = df_loadData[default_columns + features_mwlogp_columns].dropna(axis=0)
        df_loadData = df_loadData.drop(df_loadData[df_loadData["logP_rdkit"] == "None"].index).reset_index(drop=True)
    
    elif feature_type.lower() == "rdkit":
        df_loadData = df_loadData[(df_loadData['MW_rdkit'] < 1000.0)]
        df_loadData = df_loadData[(df_loadData['HBA_rdkit'] <= 10.0)]
        df_loadData = df_loadData[(df_loadData['HBD_rdkit'] <= 5.0)]
        df_loadData = df_loadData[default_columns + rdkit_columns].dropna(axis=0).reset_index(drop=True)
    
    elif feature_type.lower() == "all_mwlogp":
        df_loadData = df_loadData[(df_loadData['Fup'] >= 0.01) & (df_loadData['Fup'] <= 0.99)]
        df_loadData = df_loadData[(df_loadData['logP_rdkit'] >= -2.0) & (df_loadData['logP_rdkit'] <= 6.0)]

        df_loadData = df_loadData[(df_loadData['MW_rdkit'] < 1000.0)]
        df_loadData = df_loadData[(df_loadData['HBA_rdkit'] <= 10.0)]
        df_loadData = df_loadData[(df_loadData['HBD_rdkit'] <= 5.0)]

        df_loadData = df_loadData[default_columns + features_mwlogp_columns + rdkit_columns].dropna(axis=0)
        df_loadData = df_loadData.drop(df_loadData[df_loadData["logP_rdkit"] == "None"].index).reset_index(drop=True)

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
        df_loadData = pd.concat([df_loadData, df_augmentedData], axis=0).reset_index(drop = True)
        
        df_augmentedrdkitLabel = df_rdkitLabel[df_rdkitLabel["SMILES"].isin(list(df_augmentedData["SMILES"]))]
        df_rdkitLabel = pd.concat([df_rdkitLabel, df_augmentedrdkitLabel], axis=0).reset_index(drop = True)
        
        df_augmentedFeature = df_affinityData[df_affinityData["SMILES"].isin(list(df_augmentedData["SMILES"]))]
        df_affinityData = pd.concat([df_affinityData, df_augmentedFeature], axis=0).reset_index(drop = True)
        
    
    if scale:
        datacols = list(df_loadData.columns[1:])
        data_scaler = MinMaxScaler()
    
        df_loadData[datacols] = df_loadData[datacols].astype('float')
        scaled_data = data_scaler.fit_transform(df_loadData[datacols])
        df_loadData[datacols] = scaled_data

    return df_loadData, data_scaler, df_affinityData, df_rdkitLabel


def rdkit_rangeLabel(df_data:pd.DataFrame):
    df_feature = pd.DataFrame(data = df_data["SMILES"], columns=["SMILES"])
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
            df_feature["MW_range"][df_data[df_data["MW_rdkit"] < MW_range[idx]].index] = MW_label[idx]
        else:
            df_feature["MW_range"][df_data[(df_data["MW_rdkit"] >= MW_range[idx-1]) & (df_data["MW_rdkit"] < MW_range[idx])].index] = MW_label[idx]

            if idx == (len(MW_range)-1):
                df_feature["MW_range"][df_data[df_data["MW_rdkit"] >= MW_range[idx]].index] = MW_label[idx+1]


    for idx, _ in enumerate(PSA_range):
        if idx == 0:
            df_feature["PSA_range"][df_data[df_data["PSA_rdkit"] < PSA_range[idx]].index] = PSA_label[idx]
        else:
            df_feature["PSA_range"][df_data[(df_data["PSA_rdkit"] >= PSA_range[idx-1]) & (df_data["PSA_rdkit"] < PSA_range[idx])].index] = PSA_label[idx]

            if idx == (len(PSA_range)-1):
                df_feature["PSA_range"][df_data[df_data["PSA_rdkit"] >= PSA_range[idx]].index] = PSA_label[idx+1]


    for idx, _ in enumerate(NRB_range):
        if idx == 0:
            df_feature["NRB_range"][df_data[df_data["NRB_rdkit"] < NRB_range[idx]].index] = NRB_label[idx]
        else:
            df_feature["NRB_range"][df_data[(df_data["NRB_rdkit"] >= NRB_range[idx-1]) & (df_data["NRB_rdkit"] < NRB_range[idx])].index] = NRB_label[idx]

            if idx == (len(NRB_range)-1):
                df_feature["NRB_range"][df_data[df_data["NRB_rdkit"] >= NRB_range[idx]].index] = NRB_label[idx+1]


    for idx, _ in enumerate(HBA_range):
        if idx == 0:
            df_feature["HBA_range"][df_data[df_data["HBA_rdkit"] < HBA_range[idx]].index] = HBA_label[idx]
        else:
            df_feature["HBA_range"][df_data[(df_data["HBA_rdkit"] >= HBA_range[idx-1]) & (df_data["HBA_rdkit"] < HBA_range[idx])].index] = HBA_label[idx]

            if idx == (len(HBA_range)-1):
                df_feature["HBA_range"][df_data[df_data["HBA_rdkit"] >= HBA_range[idx]].index] = HBA_label[idx+1]


    for idx, _ in enumerate(HBD_range):
        if idx == 0:
            df_feature["HBD_range"][df_data[df_data["HBD_rdkit"] < HBD_range[idx]].index] = HBD_label[idx]
        else:
            df_feature["HBD_range"][df_data[(df_data["HBD_rdkit"] >= HBD_range[idx-1]) & (df_data["HBD_rdkit"] < HBD_range[idx])].index] = HBD_label[idx]

            if idx == (len(HBD_range)-1):
                df_feature["HBD_range"][df_data[df_data["HBD_rdkit"] >= HBD_range[idx]].index] = HBD_label[idx+1]


    for idx, _ in enumerate(LogP_range):
        if idx == 0:
            df_feature["LogP_range"][df_data[df_data["logP"] < LogP_range[idx]].index] = LogP_label[idx]
        else:
            df_feature["LogP_range"][df_data[(df_data["logP"] >= LogP_range[idx-1]) & (df_data["logP"] < LogP_range[idx])].index] = LogP_label[idx]

            if idx == (len(LogP_range)-1):
                df_feature["LogP_range"][df_data[df_data["logP"] >= LogP_range[idx]].index] = LogP_label[idx+1]

    return df_feature


def inverse_data(df_data:pd.DataFrame, scaler:MinMaxScaler):
    inverted_data = scaler.inverse_transform(df_data)
    df_data[:] = inverted_data
    
    return df_data