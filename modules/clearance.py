import copy
import torch
import torch.nn as nn

from transformers import RobertaModel, BertModel, BertConfig
import pytorch_lightning as pl

from modules.models import transformerDecoder, transformerEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.emetric import get_cindex, get_rm2

from modules.LayerModule import make_sequential
from utils.preprocess import inverse_data


class clearanceMLPModel(pl.LightningModule):
    def __init__(self, lr, dropout, model_name, feature_length,
                 act_func, df_testColData=None):
        super().__init__()
        ## -- learning rate setting -- ##
        self.lr = lr
        self.df_testColData = df_testColData

        ## -- SMILES Attention model -- ##
        self.model = RobertaModel.from_pretrained(model_name)
        
        ## -- SMILES Feature Attention model -- ##
        if feature_length != 0:
            config = BertConfig.from_pretrained("bert-base-cased")
            config.max_length = feature_length
            self.feature_model = BertModel(config)
            feature_hidden_size = config.hidden_size
        else:
            feature_hidden_size = feature_length

        ## -- clearance prediction model -- ##
        smiles_dim  = self.model.config.hidden_size + feature_hidden_size

        self.output_decoder1 = make_sequential(smiles_dim, 512, act_func, dropout)
        self.output_decoder2 = make_sequential(512, 256, act_func, dropout)
        self.output_decoder3 = make_sequential(256, 32, act_func, dropout)
        self.out_layers = nn.Linear(32, 1)
        
        ## -- optimizer function -- ##
        self.criterior = torch.nn.SmoothL1Loss()
        self.save_hyperparameters()

    def forward(self, smiles_input, features):
        outputs = self.model(smiles_input['input_ids'], smiles_input['attention_mask'])
        
        if len(features) != 0:         
            features[0] = features[0].unsqueeze(dim=2)
            features_output = self.feature_model(inputs_embeds=features[0])
            sequence_embedding = torch.mean(features_output.last_hidden_state, dim=1)
            outs = torch.cat((outputs.last_hidden_state[:,0,:], sequence_embedding), dim=1)
        else:
            outs = outputs.last_hidden_state[:,0,:]

        outs = self.output_decoder1(outs)
        outs = self.output_decoder2(outs)
        outs = self.output_decoder3(outs)
        outs = self.out_layers(outs)
        
        outs = outs.squeeze(dim=1)    

        return outs

    def training_step(self, batch):
        smiles_input, labels, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        self.log("train_loss", loss)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        smiles_input, labels, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

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
        
        rm2_score = get_rm2(y_label, y_pred)
        ci_score = get_cindex(y_label, y_pred)

        self.valid_log = {'rmse': MSE_score, 'MAE': MAE_score, 'r2': R2_score, 'rm2': rm2_score, 'ci':ci_score}

        self.log("valid_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_R2", R2_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_rm2", rm2_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_CI", ci_score, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        smiles_input, labels, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

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
        
        rm2_score = get_rm2(labels, pred)
        ci_score = get_cindex(labels, pred)

        self.test_result = {'preds': pred, 'labels': labels}
        self.test_log = {'rmse': MSE_score, 'MAE': MAE_score, 'r2': R2_score, 'rm2': rm2_score, 'ci':ci_score}

        self.log("test_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_R2", R2_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_rm2", rm2_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_CI", ci_score, on_step=False, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx):
        smiles_input, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

        outputs = self(smiles_input, features)
        return outputs.detach().cpu().numpy().tolist()

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer


class clearanceEncoderModel(pl.LightningModule):
    def __init__(self, lr, dropout, model_name, feature_length,
                 df_trainColData, df_testColData, 
                 TransformerHeads = None, TransformerLayers = None, TransformerActFunc = None):
        super().__init__()
        ## -- learning rate setting -- ##
        self.lr = lr
        self.df_trainColData = df_trainColData
        self.df_testColData = df_testColData

        ## -- SMILES Attention model -- ##
        self.model = RobertaModel.from_pretrained(model_name)
        
        ## -- SMILES Feature Attention model -- ##
        config = BertConfig.from_pretrained("bert-base-cased")
        config.max_length = feature_length
        config.hidden_size = self.model.config.hidden_size
        self.feature_model = BertModel(config)
        
        ## -- clearance prediction model -- ##
        self.transformerEncoder = transformerEncoder(self.model.config.hidden_size)
        self.output_layer = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                          nn.Linear(256, 1))
        self.criterior = torch.nn.SmoothL1Loss()
        self.save_hyperparameters()

    def forward(self, smiles_input, features):
        outputs = self.model(smiles_input['input_ids'], smiles_input['attention_mask'])
        
        if len(features) != 0:         
            features[0] = features[0].unsqueeze(dim=2)
            features_output = self.feature_model(inputs_embeds=features[0])
            outs = torch.cat((outputs.last_hidden_state[:,0,:].unsqueeze(dim=1), features_output.last_hidden_state), dim=1)
        else:
            outs = outputs.last_hidden_state[:,0,:]

        # features = features.unsqueeze(dim=2)
        # features_output = self.feature_model(inputs_embeds=features)

        # outputs = self.model(smiles_input['input_ids'], smiles_input['attention_mask'])
        # outputs = outputs.last_hidden_state[:,0,:]
        
        # encoder_src = torch.cat((outputs.unsqueeze(dim=1), features_output.last_hidden_state), dim=1)
        smiles_sequence, _ = self.transformerEncoder(outs)

        outs = self.output_layer(smiles_sequence)
        outs = outs.squeeze(dim=1)    

        return outs

    def training_step(self, batch):
        smiles_input, labels, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        self.log("train_loss", loss)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        smiles_input, labels, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

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

        rm2_score = get_rm2(y_label, y_pred)
        ci_score = get_cindex(y_label, y_pred)

        self.valid_log = {'rmse': MSE_score, 'MAE': MAE_score, 'r2': R2_score, 'rm2': rm2_score, 'ci':ci_score}

        self.log("valid_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_R2", R2_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_rm2", rm2_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_CI", ci_score, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        smiles_input, labels, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

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

        rm2_score = get_rm2(labels, pred)
        ci_score = get_cindex(labels, pred)

        self.test_result = {'preds': pred, 'labels': labels}
        self.test_log = {'rmse': MSE_score, 'MAE': MAE_score, 'r2': R2_score, 'rm2': rm2_score, 'ci':ci_score}

        self.log("test_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_R2", R2_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_rm2", rm2_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_CI", ci_score, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer
    


class clearanceDecoderModel(pl.LightningModule):
    def __init__(self, lr, dropout, model_name, feature_length,
                 df_trainColData, df_testColData, 
                 TransformerHeads = None, TransformerLayers = None, TransformerActFunc = None):
        super().__init__()
        ## -- learning rate setting -- ##
        self.lr = lr
        self.df_trainColData = df_trainColData
        self.df_testColData = df_testColData

        ## -- SMILES Attention model -- ##
        self.model = RobertaModel.from_pretrained(model_name)
        
        ## -- SMILES Feature Attention model -- ##
        if feature_length != 0:
            config = BertConfig.from_pretrained("bert-base-cased")
            config.max_length = feature_length
            config.hidden_size = self.model.config.hidden_size
            self.feature_model = BertModel(config)

        self.transformerdecoder = transformerDecoder(self.model.config.hidden_size)
        self.criterior = torch.nn.SmoothL1Loss()
        self.save_hyperparameters()

    def forward(self, smiles_input, features):
        outputs = self.model(smiles_input['input_ids'], smiles_input['attention_mask'])
        
        features[0] = features[0].unsqueeze(dim=2)
        features_output = self.feature_model(inputs_embeds=features[0])

        _, outs = self.transformerdecoder(outputs.last_hidden_state,  features_output.last_hidden_state)
        outs = outs.squeeze(dim=1)    

        return outs

    def training_step(self, batch):
        smiles_input, labels, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

        outputs = self(smiles_input, features)

        loss = self.criterior(outputs, labels)
        # fup_loss = self.criterior(outputs[1], fup_labels)

        self.log("train_loss", loss)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        smiles_input, labels, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

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

        rm2_score = get_rm2(y_label, y_pred)
        ci_score = get_cindex(y_label, y_pred)

        self.valid_log = {'rmse': MSE_score, 'MAE': MAE_score, 'r2': R2_score, 'rm2': rm2_score, 'ci':ci_score}

        self.log("valid_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_R2", R2_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_rm2", rm2_score, on_step=False, on_epoch=True, logger=True)
        self.log("valid_CI", ci_score, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        smiles_input, labels, *features = batch
        if len(features) != 0:
            features[0] = features[0].type(self.feature_model.dtype)

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
        rm2_score = get_rm2(labels, pred)
        ci_score = get_cindex(labels, pred)

        self.test_result = {'preds': pred, 'labels': labels}
        self.test_log = {'rmse': MSE_score, 'MAE': MAE_score, 'r2': R2_score, 'rm2': rm2_score, 'ci':ci_score}

        self.log("test_MSE", MSE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_MAE", MAE_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_R2", R2_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_rm2", rm2_score, on_step=False, on_epoch=True, logger=True)
        self.log("test_CI", ci_score, on_step=False, on_epoch=True, logger=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer