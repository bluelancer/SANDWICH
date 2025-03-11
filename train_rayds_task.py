import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils_downstream.Model import SimpleModel
from utils_downstream.data_utils import *
from utils_downstream.train_utils import *
from utilsgyms.utils_decisiontrans import *
from datetime import datetime



from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from filelock import FileLock
from torch.utils.data import random_split

from typing import Dict
import ray

USE_WANDB = False
def __main__():
    env_index, _, test_seed, _, _, test_data = parse_input(ds_task = True)
    seed_everything(test_seed) 
    if USE_WANDB:
        import wandb
        # Initialize wandb
        wandb.init(project="simple-model-training")
    data_postfix = "_noise_noise_5"
    config_file, current_base_path = get_config_basepath(allTx=True)
    running_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor, repo_path= get_data_dict(env_index, config_file, current_base_path, data_postfix, test_data)
    assert train_feat_tensor.size(-1) == test_feat_tensor.size(-1) == pred_feat_tensor.size(-1), "Feature dimension mismatch"
    assert train_label_tensor.size(-1) == test_label_tensor.size(-1), "Label dimension mismatch"
    # Initialize the model, loss function and optimizer
    input_dim = train_feat_tensor.size(1)
    output_dim = train_label_tensor.size(1)
    
    train_dataset = TensorDataset(train_feat_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_feat_tensor, test_label_tensor)
    pred_dataset = TensorDataset(pred_feat_tensor, test_label_tensor)

    train_loader = DataLoader(train_dataset, batch_size=int(config_file['DOWNSTREAM']['batch_size']), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(config_file['DOWNSTREAM']['batch_size']), shuffle=False)
    pred_loader = DataLoader(pred_dataset, batch_size=int(config_file['DOWNSTREAM']['batch_size']), shuffle=False)


    model = SimpleModel(input_dim, output_dim, 1)
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    model.to(device)
    criterion = nn.L1Loss()
    test_criterion = nn.L1Loss(reduction = 'none')
    
    optimizer = optim.SGD(model.parameters(), lr=float(config_file['DOWNSTREAM']['learning_rate']))

    # Watch the model
    if USE_WANDB:
        wandb.watch(model, log="all")
    # Training loop
    num_epochs = int(config_file['DOWNSTREAM']['epochs'])
    if USE_WANDB:
        iterator = range(num_epochs)
    else:
        from tqdm import tqdm
        iterator = tqdm(range(num_epochs))
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
        
        # Log training loss to wandb
        if USE_WANDB:
            wandb.log({"epoch": epoch + 1, "train_loss": running_loss / len(train_loader)})
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Test the model
    model.eval()
    model.to("cpu")
    test_loss = 0.0
    pred_test_diff_loss = 0.0
    pred_loss_sum = 0.0
    list_all_test_loss = []
    list_all_pred_test_diff_loss = []
    list_all_pred_loss_sum = []
    list_all_path_length = []
    list_all_gts = []
    list_all_test_outputs = []
    list_all_pred_outputs = []
    with torch.no_grad():
        # for batch_data, batch_labels in zip(test_loader,pred_loader) :
        for test_sample, pred_sample in zip(test_loader,pred_loader) :
            batch_test_data, batch_test_labels = test_sample
            batch_pred_data = pred_sample[0]
            batch_data, batch_labels = batch_test_data.to("cpu"), batch_test_labels.to("cpu")
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
            list_all_test_outputs.append(outputs.squeeze().numpy())
            list_all_pred_outputs.append(pred_outputs.squeeze().numpy())
        
        test_loss /= len(test_loader)
        pred_test_diff_loss /= len(pred_loader)
        pred_loss_sum /= len(pred_loader)
        
        if USE_WANDB:
            wandb.log({"test_loss": test_loss, "pred_test_diff_loss": pred_test_diff_loss, "pred_loss_sum": pred_loss_sum})
        else:
            print(f"Test Loss: {test_loss:.4f}, Pred Test diff: {pred_test_diff_loss:.4f}, Pred Loss: {pred_loss_sum:.4f}")
        result_dict = {"test_loss": list_all_test_loss, 
                       "pred_loss": list_all_pred_loss_sum, 
                       "pred_test_diff": list_all_pred_test_diff_loss, 
                       "path_length": list_all_path_length,
                       "RSSI": list_all_gts,
                       "test_RSSI": list_all_test_outputs,
                       "pred_RSSI": list_all_pred_outputs}
    
    draw_multiple_bars_same_plot_ray_ds(result_dict,
                            ["test_loss", "pred_loss","pred_test_diff"],
                            ["path_length", "RSSI"],
                            f"{repo_path}rayds_{test_data}_{running_datetime_str}",#".",# f"{current_base_path}/../../../../outputs/HF_test_result{data_postfix}/HF_test_result{data_postfix}_{datetime_str}",
                            running_datetime_str,
                            env_id=env_index)
    # # Finish the wandb run
    if USE_WANDB:
        wandb.finish()
        
if __name__ == "__main__":
    __main__()