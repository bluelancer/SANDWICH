import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ray import train
from ray.train import Checkpoint
import ray.cloudpickle as pickle
from utils_downstream.Model import SimpleModel
from utils_downstream.Baseline_Model import BaselineMLP
from torch.utils.data import TensorDataset
from utils_downstream.data_utils import *
from utilsgyms.utils_decisiontrans import *
import gc

def train_baseline_mlp(config, env_index, config_file, current_base_path, data_postfix, test_data):
    seed_everything(config['seed'])
    train_feat_tensor, test_feat_tensor, _, train_label_tensor, test_label_tensor, _= get_data_dict(env_index, config_file, current_base_path, data_postfix, test_data, return_path = "result_path", prep_baseline = True, cluster= "tetralith")
    assert train_feat_tensor.size(-1) == test_feat_tensor.size(-1), "Feature dimension mismatch"
    assert train_label_tensor.size(-1) == test_label_tensor.size(-1), "Label dimension mismatch"
    # # Initialize the model, loss function and optimizer
    input_dim = train_feat_tensor.size(1) - 1 # last column is path length
    output_dim = train_label_tensor.size(1)
    
    train_dataset = TensorDataset(train_feat_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_feat_tensor, test_label_tensor)
    
    model = BaselineMLP(input_dim, output_dim)
    if torch.cuda.is_available():
        device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)   
    # 2, prepare the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3, prepare the dataloader
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(config['batch_size']), shuffle=False)
    # 4, train the model:
    num_epochs = 100
    iterator = range(num_epochs)
    for epoch in iterator:
        model.train()
        running_loss = 0.0
        running_RSSI_loss = 0.0
        running_angular_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data[...,:-1])
            if config['loss'] == "log_angular":
                train_angular_loss = torch.log(criterion(outputs[...,1:], batch_labels[...,1:]))
            elif config['loss'] == "angular":
                train_angular_loss = criterion(outputs[...,1:], batch_labels[...,1:])
            elif config['loss'] == "non_angular":
                train_angular_loss = torch.tensor(0.0)
            else:
                raise ValueError("Unknown loss type")
            train_RSSI_loss = criterion(outputs[...,0], batch_labels[...,0])
            loss = train_RSSI_loss + train_angular_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_RSSI_loss += train_RSSI_loss.item()
            running_angular_loss += train_angular_loss.item()
        training_loss = running_loss / len(train_loader)
        training_RSSI_loss = running_RSSI_loss / len(train_loader)
        training_angular_loss= running_angular_loss / len(train_loader)
        
        
        
        # Test the model
        test_RSSI_loss = 0.0
        test_angular_reward = 0.0
        with torch.no_grad():
            for test_sample in test_loader :
                batch_test_data, batch_test_labels = test_sample
                # load the data to the device
                batch_test_data, batch_test_labels = batch_test_data.to(device), batch_test_labels.to(device)
                # the model performanc on the test data
                outputs = model(batch_test_data[...,:-1])
                angular_reward = -torch.log(criterion(outputs[...,1:], batch_test_labels[...,1:]))
                RSSI_loss = criterion(outputs[...,0], batch_test_labels[...,0])
                test_angular_reward += angular_reward.item()
                test_RSSI_loss += RSSI_loss.item() # 
                # Collect the data for plotting
        test_RSSI_loss /= len(test_loader)
        test_angular_reward /= len(test_loader)
        test_loss = test_RSSI_loss - test_angular_reward
        checkpoint_data = {
                    "epoch": epoch,
                    "net_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            with open(path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"training_loss":(training_loss),
                 "training_angular_loss": (training_angular_loss),
                 "training_RSSI_loss": (training_RSSI_loss),
                 "test_RSSI_loss": (test_RSSI_loss),
                 "test_angular_reward": (test_angular_reward),
                 "test_loss": (test_loss)},
                checkpoint=checkpoint,
            )
            gc.collect()
    return

def test_baseline_mlp_process(model, test_dataset,batch_size, test_criterion, return_data = False, loss_type = "RSSI"):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    test_loss = 0.0
    list_MLP_RSSI_loss = []
    list_MLP_all_angle_reward_array  = []
    list_MLP_all_path_length = []
    list_MLP_RSSI_all_gts = []
    model.to("cpu")
    with torch.no_grad():
        for batch_test_data, batch_test_labels  in test_loader :
            batch_test_data, batch_test_labels = batch_test_data.to("cpu"), batch_test_labels.to("cpu")
            # the model performanc on the test data
            outputs = model(batch_test_data[...,:-1])
            RSSI_loss_array = test_criterion(outputs[...,0], batch_test_labels[...,0])
            Angle_reward_array = -torch.log(test_criterion(outputs[...,1:], batch_test_labels[...,1:]))
            RSSI_loss = torch.mean(RSSI_loss_array)
            Angle_reward = torch.mean(Angle_reward_array)
            # if loss_type == "RSSI":
            #     test_loss += RSSI_loss.item()
            # elif loss_type == "angular":
            #     test_loss -= Angle_reward.item()
            # else:
            test_loss += RSSI_loss.item() - Angle_reward.item()

            # Collect the data for plotting
            list_MLP_RSSI_loss.append(RSSI_loss_array.squeeze().numpy())
            list_MLP_all_path_length.append(batch_test_data[...,-1].int().numpy()) # last column is path length
            list_MLP_RSSI_all_gts.append(batch_test_labels.squeeze().numpy())
            list_MLP_all_angle_reward_array.append(Angle_reward_array.squeeze().numpy())
            
        test_loss /= len(test_loader)
        
        if return_data:
            return test_loss,RSSI_loss,Angle_reward, list_MLP_RSSI_loss, list_MLP_all_path_length, list_MLP_RSSI_all_gts, list_MLP_all_angle_reward_array
        else:
            return test_loss,RSSI_loss, Angle_reward