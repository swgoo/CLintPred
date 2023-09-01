import copy
import torch
import torch.nn as nn

import pytorch_lightning as pl
from transformers import RobertaModel

from modules.models import transformerDecoder, transformerEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils.utils import inverse_data


class clearanceEncoderModel(pl.LightningModule):
    def __init__(self, lr, dropout, model_name, feature_dim, 
                 df_trainColData, df_testColData,
                 TransformerHeads = None, TransformerLayers = None, TransformerActFunc = None):
        super().__init__()
        
        self.lr = lr
        self.dropout = dropout

        self.df_trainColData = df_trainColData
        self.df_testColData = df_testColData

        self.model = RobertaModel.from_pretrained(model_name)
        # self.transformerdecoder = transformerDecoder(self.model.hidden_size, 
        #                                             num_heads=TransformerHeads, num_layers=TransformerLayers, act_function=TransformerActFunc)
        self.transformerEncoder = transformerEncoder(self.model.config.hidden_size)
        self.feature_layer = nn.Linear(feature_dim, self.model.config.hidden_size)
        self.criterior = torch.nn.SmoothL1Loss()
        
        self.save_hyperparameters()

    def forward(self, smiles_input, features):
        features = features.unsqueeze(dim=1)
        features_output = self.feature_layer(features)

        outputs = self.model(smiles_input['input_ids'], smiles_input['attention_mask'])
        _, outs = self.transformerEncoder(outputs.last_hidden_state)
        
        outs = outs.squeeze(dim=1)    

        return outs

    def training_step(self, batch):
        smiles_input, features, labels = batch
        features = features.type(self.feature_layer.weight.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        # fup_loss = self.criterior(outputs[1], fup_labels)

        self.log("train_loss", loss)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        smiles_input, features, labels = batch
        features = features.type(self.feature_layer.weight.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)

        return {"outputs": outputs, "labels": labels}

    def validation_step_end(self, outputs):
        return {"outputs": outputs["outputs"], "labels": outputs["labels"]}

    def validation_epoch_end(self, outputs):
        preds = torch.as_tensor(torch.cat([output['outputs'] for output in outputs], dim=0))
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0))

        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()

        MSE_score = mean_squared_error(y_label, y_pred)
        MAE_score = mean_absolute_error(y_label, y_pred)
        R2_score = r2_score(y_label, y_pred)

        self.log("valid_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_R2", R2_score, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        smiles_input, features, labels = batch
        features = features.type(self.feature_layer.weight.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"outputs": outputs,  "labels": labels}

    def test_step_end(self, outputs):
        return {"outputs": outputs["outputs"], "labels": outputs["labels"]}

    def test_epoch_end(self, outputs):
        preds = torch.as_tensor(torch.cat([output['outputs'] for output in outputs], dim=0))
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0))

        y_pred = preds.detach().cpu().numpy()
        # y_label = labels.detach().cpu().numpy()
        
        if self.trainer.datamodule.sep_test:
            test_pos = self.trainer.datamodule.valid_pos
            test_scaler = self.trainer.datamodule.train_scaler

            df_predData = copy.deepcopy(self.df_trainColData.iloc[test_pos:])
            df_predData["Clint"] = y_pred

            df_pred = inverse_data(df_predData, test_scaler)
            df_label = inverse_data(self.df_trainColData.iloc[test_pos:], test_scaler)

        else:
            test_scaler = self.trainer.datamodule.test_scaler

            df_predData = copy.deepcopy(self.df_testColData)
            df_predData["Clint"] = y_pred

            df_pred = inverse_data(df_predData, test_scaler)
            df_label = inverse_data(self.df_testColData, test_scaler)

        pred, labels = df_pred['Clint'], df_label['Clint']

        MSE_score = mean_squared_error(labels, pred)
        MAE_score = mean_absolute_error(labels, pred)
        R2_score = r2_score(labels, pred)

        self.test_result = {'preds': pred, 'labels': labels}
        # self.test_pred = y_pred
        self.test_log = {'rmse': MSE_score, 'MAE': MAE_score, 'r2': R2_score}

        self.log("test_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_R2", R2_score, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer
    

class clearanceDecoderModel(pl.LightningModule):
    def __init__(self, lr, dropout, model_name, feature_dim, 
                 df_trainColData, df_testColData,
                 TransformerHeads = None, TransformerLayers = None, TransformerActFunc = None):
        super().__init__()
        
        self.lr = lr
        self.dropout = dropout

        self.df_trainColData = df_trainColData
        self.df_testColData = df_testColData

        self.model = RobertaModel.from_pretrained(model_name)
        # self.transformerdecoder = transformerDecoder(self.model.hidden_size, 
        #                                             num_heads=TransformerHeads, num_layers=TransformerLayers, act_function=TransformerActFunc)
        self.transformerdecoder = transformerDecoder(self.model.config.hidden_size)
        self.feature_layer = nn.Linear(feature_dim, self.model.config.hidden_size)
        self.criterior = torch.nn.SmoothL1Loss()
        
        self.save_hyperparameters()

    def forward(self, smiles_input, features):
        features = features.unsqueeze(dim=1)
        features_output = self.feature_layer(features)

        outputs = self.model(smiles_input['input_ids'], smiles_input['attention_mask'])
        _, outs = self.transformerdecoder(outputs.last_hidden_state, features_output)
        
        outs = outs.squeeze(dim=1)    

        return outs

    def training_step(self, batch):
        smiles_input, features, labels = batch
        features = features.type(self.feature_layer.weight.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        # fup_loss = self.criterior(outputs[1], fup_labels)

        self.log("train_loss", loss)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        smiles_input, features, labels = batch
        features = features.type(self.feature_layer.weight.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)

        return {"outputs": outputs, "labels": labels}

    def validation_step_end(self, outputs):
        return {"outputs": outputs["outputs"], "labels": outputs["labels"]}

    def validation_epoch_end(self, outputs):
        preds = torch.as_tensor(torch.cat([output['outputs'] for output in outputs], dim=0))
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0))

        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()
        
        ## -- pred data scaling -- ##
            # # valid_pos = self.trainer.datamodule.train_pos
            # valid_pos, valid_end = self.trainer.datamodule.train_pos, self.trainer.datamodule.valid_pos
            # train_scaler = self.trainer.datamodule.train_scaler

            # # df_predData = copy.deepcopy(self.df_trainColData.iloc[valid_pos:])
            # df_predData = copy.deepcopy(self.df_trainColData.iloc[valid_pos:valid_end])
            # df_predData["Clint"] = y_pred

            # df_pred = inverse_data(df_predData, train_scaler)
            # # df_label = inverse_data(self.df_trainColData.iloc[valid_pos:], train_scaler)
            # df_label = inverse_data(self.df_trainColData.iloc[valid_pos:valid_end], train_scaler)

            # pred, labels = list(df_pred['Clint']), list(df_label['Clint'])

        MSE_score = mean_squared_error(y_label, y_pred)
        MAE_score = mean_absolute_error(y_label, y_pred)
        R2_score = r2_score(y_label, y_pred)

        self.log("valid_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_R2", R2_score, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        smiles_input, features, labels = batch
        features = features.type(self.feature_layer.weight.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"outputs": outputs,  "labels": labels}

    def test_step_end(self, outputs):
        return {"outputs": outputs["outputs"], "labels": outputs["labels"]}

    def test_epoch_end(self, outputs):
        preds = torch.as_tensor(torch.cat([output['outputs'] for output in outputs], dim=0))
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0))

        y_pred = preds.detach().cpu().numpy()
        # y_label = labels.detach().cpu().numpy()
        
        if self.trainer.datamodule.sep_test:
            test_pos = self.trainer.datamodule.valid_pos
            test_scaler = self.trainer.datamodule.train_scaler

            df_predData = copy.deepcopy(self.df_trainColData.iloc[test_pos:])
            df_predData["Clint"] = y_pred

            df_pred = inverse_data(df_predData, test_scaler)
            df_label = inverse_data(self.df_trainColData.iloc[test_pos:], test_scaler)

        else:
            test_scaler = self.trainer.datamodule.test_scaler

            df_predData = copy.deepcopy(self.df_testColData)
            df_predData["Clint"] = y_pred

            df_pred = inverse_data(df_predData, test_scaler)
            df_label = inverse_data(self.df_testColData, test_scaler)

        pred, labels = df_pred['Clint'], df_label['Clint']

        MSE_score = mean_squared_error(labels, pred)
        MAE_score = mean_absolute_error(labels, pred)
        R2_score = r2_score(labels, pred)

        self.test_result = {'preds': pred, 'labels': labels}
        self.test_log = {'rmse': MSE_score, 'MAE': MAE_score, 'r2': R2_score}

        self.log("test_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_R2", R2_score, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer