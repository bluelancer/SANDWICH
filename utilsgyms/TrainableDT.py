import torch    
from transformers import DecisionTransformerModel

import wandb
import os
import pickle

class TrainableDT(DecisionTransformerModel):
    type_loss_func = torch.nn.CrossEntropyLoss()
    surface_loss_func = torch.nn.CrossEntropyLoss()
    def __init__(self, config, len_ds = None, surface_id = False, type_loss = True):    
        self.training_count = 0
        self.best_kwargs = {}
        if len_ds:
            self.len_ds = len_ds
        super().__init__(config)
        self.surface_id = surface_id
        self.type_loss = type_loss
        if self.surface_id:
            print ("Surface id is added")
        if self.type_loss:
            print ("Type loss is added")
        
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        action_preds = output[1] # Shape torch.Size([1024, 20, 8])
        action_targets = kwargs["actions"] # Shape
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # loss = torch.mean(abs(action_preds[:,1:] - action_targets[:,1:]))

        type_loss = torch.mean(TrainableDT.type_loss_func(action_preds[:,:6], action_targets[:,:6])) # first 6 columns are type, one hotted
        Angular_loss = torch.mean(abs(action_preds[:,6:8] - action_targets[:,6:8])) # last 2 columns are azimuth and radian
        # loss = Angular_loss + type_loss
        # TODO: think about this: if angular loss is too low, say < 0.1 thus type loss will be dominant
        # but our test reward is measured by angular loss, actually log(angular_loss)
        # while take log(angular_loss) without clipping, it will be too large
        # thus we need to clip the angular loss
        log_ang_loss = torch.log(Angular_loss +1e-5) # if Angular_loss < 1e-5, we consider type loss shoud be dominant
        if self.type_loss:
            loss = type_loss + log_ang_loss
        else:
            loss = log_ang_loss
        if self.surface_id:    
            surface_loss = torch.mean(TrainableDT.surface_loss_func(action_preds[:,8:], action_targets[:,8:])) 
            loss = loss + surface_loss
            wandb.log({"Angular_loss": Angular_loss, "type_loss": type_loss, "log_ang_loss": log_ang_loss, "surface_loss": surface_loss, "loss": loss})
        else:
            wandb.log({"Angular_loss": Angular_loss, "type_loss": type_loss, "log_ang_loss": log_ang_loss, "loss": loss})
        return {"loss": loss,
                "Angular_loss": Angular_loss,
                "type_loss": type_loss,
                "log_ang_loss": log_ang_loss,
                "original_loss": Angular_loss + type_loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)