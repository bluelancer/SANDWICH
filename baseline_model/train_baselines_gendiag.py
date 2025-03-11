import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils_downstream.Baseline_Model import BaselineMLP
from utils_downstream.data_utils import *
from utils_downstream.train_utils import *
from utilsgyms.utils_decisiontrans import *
from datetime import datetime
from tqdm import tqdm

def __main__():
    env_index, _, test_seed, _, _, _ = parse_input(ds_task = True)
    seed_everything(test_seed) 
    data_postfix = "_noise_noise_5"
    config_file, current_base_path = get_config_basepath(allTx=True, cluster= "tetralith")
    running_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_feat_tensor, gendiag_feat_tensor, _, train_label_tensor, gendiag_label_tensor, gendiag_result_path= get_data_dict(env_index, config_file, current_base_path, data_postfix, "gendiag", return_path = "result_path", prep_baseline = True, cluster= "tetralith")
    assert train_feat_tensor.size(-1) == gendiag_feat_tensor.size(-1), "Feature dimension mismatch"
    assert train_label_tensor.size(-1) == gendiag_label_tensor.size(-1), "Label dimension mismatch"
    
    # Initialize the model, loss function and optimizer
    input_dim = train_feat_tensor.size(1) - 1 # last column is path length
    output_dim = train_label_tensor.size(1)

    train_dataset = TensorDataset(train_feat_tensor, train_label_tensor)
    gendiag_dataset = TensorDataset(gendiag_feat_tensor, gendiag_label_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    gendiag_loader = DataLoader(gendiag_dataset, batch_size=16, shuffle=False)
    
    model = BaselineMLP(input_dim, output_dim)
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    model.to(device)
    criterion = nn.L1Loss()
    test_criterion = nn.L1Loss(reduction = 'none')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 80 
    iterator = tqdm(range(num_epochs))

    for epoch in iterator:
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data[...,:-1])
            train_angular_loss = criterion(outputs[...,1:],batch_labels[...,1:])
            train_RSSI_loss = criterion(outputs[...,0], batch_labels[...,0])
            loss = train_RSSI_loss + train_angular_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Log training loss to wandb
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Test the model
    model.eval()
    loader_dict = {"gendiag": gendiag_loader}
    path_dict = {"gendiag": gendiag_result_path}

    for test_data_name, test_loader in loader_dict.items():
        RSSI_test_loss = 0.0
        angle_test_loss = 0.0
        list_MLP_RSSI_loss = []
        list_MLP_all_angle_loss_array  = []
        list_MLP_all_path_length = []
        list_MLP_RSSI_all_gts = []
        with torch.no_grad():
            for batch_test_data, batch_test_labels  in test_loader :
                batch_test_data, batch_test_labels = batch_test_data.to(device), batch_test_labels.to(device)
                # the model performance on the test data
                outputs = model(batch_test_data[...,:-1])
                RSSI_loss_array = test_criterion(outputs[...,0], batch_test_labels[...,0])
                RSSI_loss = torch.mean(RSSI_loss_array)
                RSSI_test_loss += RSSI_loss.item()
                
                angle_loss_array = test_criterion(outputs[...,1:], batch_test_labels[...,1:])
                angle_loss = torch.mean(angle_loss_array)
                angle_test_loss += angle_loss.item()
                # Collect the data for plotting
                list_MLP_RSSI_loss.append(RSSI_loss_array.squeeze().cpu().numpy())
                list_MLP_all_path_length.append(batch_test_data[...,-1].int().cpu().numpy())

            RSSI_test_loss /= len(test_loader)
            angle_test_loss /= len(test_loader)

        print(f"RSSI_test_loss: {RSSI_test_loss:.4f}, angle_test_loss: {angle_test_loss:.4f} for {test_data_name}, in env_index: {env_index}")
        
        result_dict = {"MLP_RSSI_loss": list_MLP_RSSI_loss, 
                       "MLP_angle_loss": list_MLP_all_angle_loss_array, 
                       "path_length": list_MLP_all_path_length,
                       "RSSI": list_MLP_RSSI_all_gts} 
        pickle.dump(result_dict, open(f"{path_dict[test_data_name]}/MLP_result_dict_{test_data_name}_{running_datetime_str}.pkl", "wb"))
        
if __name__ == "__main__":
    __main__()