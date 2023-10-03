import copy
import torch
import torch.nn as nn

import pytorch_lightning as pl
from transformers import RobertaModel, BertModel, BertConfig

from modules.models import transformerDecoder, transformerEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils.utils import inverse_data, make_sequential



class clearanceMLPModel(pl.LightningModule):
    def __init__(self, lr, dropout, model_name, feature_length,
                 df_trainColData, df_testColData, act_func):
        super().__init__()
        self.lr = lr
        self.dropout = dropout

        self.df_trainColData = df_trainColData
        self.df_testColData = df_testColData

        self.model = RobertaModel.from_pretrained(model_name)
        # self.feature_layer = nn.Linear(1, self.model.config.hidden_size)
        config = BertConfig.from_pretrained("bert-base-cased")
        config.max_length = feature_length
        self.feature_model = BertModel(config)

        smiles_dim  = self.model.config.hidden_size + config.hidden_size

        self.output_decoder1 = make_sequential(smiles_dim, 512, act_func[0])
        self.output_decoder2 = make_sequential(512, 256, act_func[1])
        self.output_decoder3 = make_sequential(256, 32, act_func[2])
        self.out_layers = nn.Linear(32, 1)
        
        self.criterior = torch.nn.SmoothL1Loss()
        self.save_hyperparameters()

    def forward(self, smiles_input, features):
        features = features.unsqueeze(dim=2)
        features_output = self.feature_model(inputs_embeds=features)

        outputs = self.model(smiles_input['input_ids'], smiles_input['attention_mask'])

        outs = torch.cat((outputs.last_hidden_state[:,0,:], features_output.last_hidden_state[:,0,:]), dim=1)

        outs = self.output_decoder1(outs)
        outs = self.output_decoder2(outs)
        outs = self.output_decoder3(outs)
        outs = self.out_layers(outs)
        
        outs = outs.squeeze(dim=1)    

        return outs

    def training_step(self, batch):
        smiles_input, labels, features = batch
        features = features.type(self.feature_model.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        self.log("train_loss", loss)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        smiles_input, labels, features = batch
        features = features.type(self.feature_model.dtype)

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
        smiles_input, labels, features = batch
        features = features.type(self.feature_model.dtype)

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


class clearanceEncoderModel(pl.LightningModule):
    def __init__(self, lr, dropout, model_name, feature_length,
                 df_trainColData, df_testColData, 
                 TransformerHeads = None, TransformerLayers = None, TransformerActFunc = None):
        super().__init__()
        
        self.lr = lr
        self.dropout = dropout

        self.df_trainColData = df_trainColData
        self.df_testColData = df_testColData

        self.model = RobertaModel.from_pretrained(model_name)

        config = BertConfig.from_pretrained("bert-base-cased")
        config.max_length = feature_length
        self.feature_model = BertModel(config)

        # smiles_dim  = self.model.config.hidden_size + config.hidden_size

        self.transformerEncoder = transformerEncoder(self.model.config.hidden_size)
        
        self.output_layer = nn.Sequential(nn.Linear(config.hidden_size, 256),
                                          nn.Linear(256, 1))
        self.criterior = torch.nn.SmoothL1Loss()
        
        self.save_hyperparameters()

    def forward(self, smiles_input, features):
        features = features.unsqueeze(dim=2)
        features_output = self.feature_model(inputs_embeds=features)

        outputs = self.model(smiles_input['input_ids'], smiles_input['attention_mask'])
        outputs = outputs.last_hidden_state[:,0,:]
        
        encoder_src = torch.cat((outputs.unsqueeze(dim=1), features_output.last_hidden_state), dim=1)
        smiles_sequence, outs = self.transformerEncoder(encoder_src)

        outs = self.output_layer(smiles_sequence)
        outs = outs.squeeze(dim=1)    

        return outs

    def training_step(self, batch):
        smiles_input, labels, features = batch
        features = features.type(self.feature_model.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        self.log("train_loss", loss)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        smiles_input, labels, features = batch
        features = features.type(self.feature_model.dtype)

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
        smiles_input, labels, features = batch
        features = features.type(self.feature_model.dtype)

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
    


class clearanceDecoderModel(pl.LightningModule):
    def __init__(self, lr, dropout, model_name, feature_length,
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
        self.feature_layer = nn.Linear(1, self.model.config.hidden_size)
        self.criterior = torch.nn.SmoothL1Loss()
        
        self.save_hyperparameters()

    def forward(self, smiles_input, features):
        features = features.unsqueeze(dim=2)
        features_output = self.feature_layer(features)

        outputs = self.model(smiles_input['input_ids'], smiles_input['attention_mask'])
        _, outs = self.transformerdecoder(features_output, outputs.last_hidden_state[:,0,:])
        
        outs = outs.squeeze(dim=1)    

        return outs

    def training_step(self, batch):
        smiles_input, labels, features = batch
        features = features.type(self.feature_layer.weight.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        # fup_loss = self.criterior(outputs[1], fup_labels)

        self.log("train_loss", loss)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        smiles_input, labels, features = batch
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
        
        # valid_scaler = self.trainer.datamodule.train_scaler

        # df_predData = copy.deepcopy(self.df_trainColData[self.trainer.datamodule.train_pos:])
        # df_predData["Clint"] = y_pred

        # df_pred = inverse_data(df_predData, valid_scaler)
        # df_label = inverse_data(self.df_trainColData[self.trainer.datamodule.train_pos:], valid_scaler)
            
        # pred, labels = df_pred['Clint'], df_label['Clint']

        MSE_score = mean_squared_error(y_label, y_pred)
        MAE_score = mean_absolute_error(y_label, y_pred)
        R2_score = r2_score(y_label, y_pred)

        self.log("valid_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_R2", R2_score, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        smiles_input, labels, features = batch
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