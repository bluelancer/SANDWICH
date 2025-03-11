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

def train_rayds(config, env_index, config_file, current_base_path, data_postfix, test_data):
    seed_everything(config['seed'])
    train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor, repo_path= get_data_dict(env_index, config_file, current_base_path, data_postfix, test_data)
    assert train_feat_tensor.size(-1) == test_feat_tensor.size(-1) == pred_feat_tensor.size(-1), "Feature dimension mismatch"
    assert train_label_tensor.size(-1) == test_label_tensor.size(-1), "Label dimension mismatch"
    # # Initialize the model, loss function and optimizer
    input_dim = train_feat_tensor.size(1)
    output_dim = train_label_tensor.size(1)
    
    train_dataset = TensorDataset(train_feat_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_feat_tensor, test_label_tensor)
    pred_dataset = TensorDataset(pred_feat_tensor, test_label_tensor)
    # 1, prepare the model
    model = SimpleModel(input_dim, output_dim, config["n_layers"])
    if torch.cuda.is_available():
        device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)   
    # 2, prepare the loss function and optimizer
    criterion = nn.L1Loss()
    test_criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=float(config['lr']))
    
    # 3, prepare the dataloader
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=int(config['batch_size']), shuffle=False)
    pred_loader = DataLoader(pred_dataset, batch_size=int(config['batch_size']), shuffle=False)
    # 4, train the model:
    num_epochs = int(config['epochs'])
    iterator = range(num_epochs)
    for epoch in iterator:
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_loss = running_loss / len(train_loader)
        # Test the model
        test_loss = 0.0
        pred_loss_sum = 0.0
        with torch.no_grad():
            for test_sample, pred_sample in zip(test_loader,pred_loader) :
                batch_test_data, batch_test_labels = test_sample
                batch_pred_data = pred_sample[0]
                # load the data to the device
                batch_test_data, batch_test_labels = batch_test_data.to(device), batch_test_labels.to(device)
                batch_pred_data = batch_pred_data.to(device)
                # the model performanc on the test data
                outputs = model(batch_test_data)
                loss = test_criterion(outputs, batch_test_labels)
                test_loss += loss.item()
                # the model performance on the pred (generated) data
                pred_outputs = model(batch_pred_data)
                pred_loss = test_criterion(pred_outputs, batch_test_labels)
                pred_loss_sum += pred_loss.item()
                # Collect the data for plotting
        test_loss /= len(test_loader)
        pred_loss_sum /= len(pred_loader)
        
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
            # ValueError: 'checkpoint_at_end' cannot be used with a function trainable. You should include 
            # one last call to `ray.train.report(metrics=..., checkpoint=...)` at the end of your training 
            # loop to get this behavior.
            train.report(
                {"training_loss":(training_loss), "test_loss": (test_loss), "pred_loss_sum": (pred_loss_sum)},
                checkpoint=checkpoint,
            )
    return

def test_process(model, test_dataset, pred_dataset,batch_size,  test_criterion, return_data = False):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    test_loss = 0.0
    pred_test_diff_loss = 0.0
    pred_loss_sum = 0.0
    list_all_test_loss = []
    list_all_pred_test_diff_loss = []
    list_all_pred_loss_sum = []
    list_all_path_length = []
    list_all_gts = []
    model.to("cpu")
    with torch.no_grad():
        for test_sample, pred_sample in zip(test_loader,pred_loader) :
            batch_test_data, batch_test_labels = test_sample
            batch_pred_data = pred_sample[0]
            batch_test_data, batch_test_labels = batch_test_data.to("cpu"), batch_test_labels.to("cpu")
            batch_pred_data = batch_pred_data.to("cpu")
            # the model performanc on the test data
            outputs = model(batch_test_data)
            loss_array = test_criterion(outputs, batch_test_labels)
            loss = torch.mean(loss_array)
            test_loss += loss.item()
            # the model performance on the pred (generated) data
            pred_outputs = model(batch_pred_data)
            pred_loss_array = test_criterion(pred_outputs, batch_test_labels)
            pred_loss = torch.mean(pred_loss_array)
            pred_loss_sum += pred_loss.item()
            # the performance diff between the pred and the test data
            pred_diff_array = test_criterion(pred_outputs, outputs)
            pred_diff = torch.mean(pred_diff_array)            
            pred_test_diff_loss += pred_diff.item()
            # Collect the data for plotting
            list_all_test_loss.append(loss_array.squeeze().numpy())
            list_all_pred_loss_sum.append(pred_loss_array.squeeze().numpy())
            list_all_pred_test_diff_loss.append(pred_diff_array.squeeze().numpy())
            list_all_path_length.append(batch_test_data[...,-1].int().numpy()) # last column is path length
            list_all_gts.append(batch_test_labels.squeeze().numpy())
        
        test_loss /= len(test_loader)
        pred_test_diff_loss /= len(pred_loader)
        pred_loss_sum /= len(pred_loader)
        
        if return_data:
            return test_loss, pred_test_diff_loss, pred_loss_sum, list_all_test_loss, list_all_pred_loss_sum, list_all_pred_test_diff_loss, list_all_path_length, list_all_gts
        else:
            return test_loss, pred_test_diff_loss, pred_loss_sum
    