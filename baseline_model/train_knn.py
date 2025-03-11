import torch.nn as nn
from utils_downstream.data_utils import *
from utils_downstream.train_utils import *
from utilsgyms.utils_decisiontrans import *
from datetime import datetime
from torch_cluster import knn as knn_torch_cluster
import pandas as pd
DEBUG = False
def __main__():
    NO_ENV_INDEX = [9,10,13,18,29,31,32,33,43,45,47,48,49,59,64,67,71,72,77,86,88,90,92,93,94,99]
    if DEBUG:
        env_index_iter = [1]
    else:
        env_index_iter = range(1, 100)
    df = pd.DataFrame()
    for env_index in env_index_iter:    
        if env_index in NO_ENV_INDEX:
            continue
        else:
            seed_everything(42) 
            
            data_postfix = "_noise_noise_5"
            config_file, current_base_path = get_config_basepath(allTx=True, cluster= "tetralith")
            running_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            criterion = nn.L1Loss()
            criterion_no_reduction = nn.L1Loss(reduction="none")
            train_feat_tensor, test_feat_tensor, _, train_label_tensor, test_label_tensor, test_result_path= get_data_dict(env_index, config_file, current_base_path, data_postfix, "test", return_path = "result_path", prep_baseline = True, cluster= "tetralith", baseline_type="KNN")
            _, genz_feat_tensor, _, _, genz_label_tensor, genz_result_path= get_data_dict(env_index, config_file, current_base_path, data_postfix, "genz", return_path = "result_path", prep_baseline = True, cluster= "tetralith", baseline_type="KNN")

            train_Tx, train_Rx, train_path, train_coord, train_edge_attr_tensor, train_reduced_mask, train_AoD, train_AoA = train_feat_tensor
            test_Tx, test_Rx, test_path, test_coord, test_edge_attr_tensor, test_reduced_mask, test_AoD, test_AoA = test_feat_tensor
            genz_Tx, genz_Rx, genz_path, genz_coord, genz_edge_attr_tensor, genz_reduced_mask, genz_AoD, genz_AoA = genz_feat_tensor
            train_Rx, test_Rx, genz_Rx = train_Rx.squeeze()[:,1:], test_Rx.squeeze()[:,1:], genz_Rx.squeeze()[:,1:]
            test_data_dict = {"test": [test_Tx, test_Rx, test_label_tensor, test_edge_attr_tensor,test_reduced_mask, test_path, test_result_path, test_coord, test_result_path, test_AoD, test_AoA],
                            "genz": [genz_Tx, genz_Rx, genz_label_tensor,genz_edge_attr_tensor ,genz_reduced_mask,genz_path, genz_result_path, genz_coord, genz_result_path, genz_AoD, genz_AoA]}
            if env_index < 49:
                _, gendiag_feat_tensor, _, _, gendiag_label_tensor, gendiag_result_path= get_data_dict(env_index, config_file, current_base_path, data_postfix, "gendiag", return_path = "result_path", prep_baseline = True, cluster= "tetralith", baseline_type="KNN")
                gendiag_Tx, gendiag_Rx, gendiag_path, gendiag_coord, gendiag_edge_attr_tensor, gendiag_reduced_mask, gendiag_AoD, gendiag_AoA = gendiag_feat_tensor
                gendiag_Rx = gendiag_Rx.squeeze()[:,1:]
                test_data_dict["gendiag"] = [gendiag_Tx, gendiag_Rx, gendiag_label_tensor, gendiag_edge_attr_tensor,gendiag_reduced_mask, gendiag_path, gendiag_result_path, gendiag_coord, gendiag_result_path, gendiag_AoD, gendiag_AoA]
                # "gendiag": [gendiag_Tx, gendiag_Rx, gendiag_label_tensor, gendiag_edge_attr_tensor, gendiag_path, gendiag_result_path, gendiag_coord, gendiag_result_path]}

            # https://discuss.pytorch.org/t/usage-of-torch-scatter-for-multi-dimensional-value/171770
            # KNN of |Tx_train - Tx_test|^2 + |Rx_train - Rx_test|^2
            for key, value in test_data_dict.items():
                test_data_name = key
                test_Tx, test_Rx, test_label_tensor, test_edge_attr_tensor,test_reduced_mask,test_path, test_result_path, test_coord, test_result_path, test_AoD, test_AoA = value
                tx_assign_index = knn_torch_cluster(train_Tx, test_Tx, 1)
                # torch way of doing KNN
                rx_assign_index = knn_torch_cluster(train_Rx.squeeze(), test_Rx.squeeze(), 1)
                
                # knn = NearestNeighbors(n_neighbors=3, algorithm='auto')
                # train_Rx_np = train_Rx.squeeze().cpu().numpy() 
                # test_Rx_np = test_Rx.squeeze().cpu().numpy()
                # knn.fit(train_Rx_np)
                # distances, indices = knn.kneighbors(test_Rx_np)
                # second_nearest_index = indices[:,0]
                # rx_assign_index = torch.tensor(second_nearest_index)
                # predicted_RSSI = torch.scatter_add(train_label_tensor[tx_assign_index[1], rx_assign_index[1],...], 0, tx_assign_index[0].unsqueeze(-1).expand(-1, train_label_tensor.shape[-1]), 1)
                
                predicted_RSSI = train_label_tensor[tx_assign_index[1], :, :, :][:, rx_assign_index[1], :, :]
                predicted_edge_attr = train_edge_attr_tensor[tx_assign_index[1], :, :][:, rx_assign_index[1], :]
                pred_AoD = train_AoD[tx_assign_index[1], :, :][:, rx_assign_index[1], :]
                pred_AoA = train_AoA[tx_assign_index[1], :, :][:, rx_assign_index[1], :]

                # AoD_pred = predicted_edge_attr[...,0,:]
                # AoA_pred = torch.zeros_like(AoD_pred)
                # test_path_index = test_path.unsqueeze(-1).expand(-1, -1, -1, predicted_edge_attr.size(-2), predicted_edge_attr.size(-1)) - 1
                # AoA_pred = AoA_pred.scatter(-2, test_path_index, predicted_edge_attr)
                # AoA_GT = torch.zeros_like(AoD_pred)
                # AoA_GT = AoA_GT.scatter(-2, test_path_index, test_edge_attr_tensor)

                test_reduced_mask_expanded_edge = test_reduced_mask.unsqueeze(-1).expand(-1, -1, -1, -1, predicted_edge_attr.size(-1))
                train_reduced_mask_expanded_edge = train_reduced_mask.unsqueeze(-1).expand(-1, -1, -1, -1, predicted_edge_attr.size(-1))
                # mask the valid path length
                predicted_edge_attr[~test_reduced_mask_expanded_edge[...,1:,:]] = 0
                test_edge_attr_tensor[~test_reduced_mask_expanded_edge[...,1:,:]] = 0

                RSSI_test_loss_mean = criterion(predicted_RSSI, test_label_tensor)
                angle_test_loss_mean = criterion(predicted_edge_attr, test_edge_attr_tensor)
                AoA_loss_mean = criterion(pred_AoA, test_AoA)
                AoD_loss_mean = criterion(pred_AoD, test_AoD)
                RSSI_test_loss = criterion_no_reduction(predicted_RSSI, test_label_tensor)
                angle_test_loss = criterion_no_reduction(predicted_edge_attr, test_edge_attr_tensor)
                print (f"RSSI_test_loss: {RSSI_test_loss_mean.item()}, angle_test_loss: {angle_test_loss_mean.item()}, in {test_data_name}, env_index: {env_index}, AoA_loss: {AoA_loss_mean.item()}, AoD_loss: {AoD_loss_mean.item()}")
                df = df._append({"RSSI_test_loss": RSSI_test_loss_mean.item(), "angle_test_loss": angle_test_loss_mean.item(), "test_data_name": test_data_name, "env_index": env_index, "AoA_loss": AoA_loss_mean.item(), "AoD_loss": AoD_loss_mean.item()}, ignore_index=True)
                result_dict = {"RSSI_loss": RSSI_test_loss.numpy(), 
                                "KNN_angle_loss": angle_test_loss.numpy(), 
                                "RSSI": test_label_tensor.numpy(),
                                "pred_RSSI": predicted_RSSI.numpy()}
                pickle.dump(result_dict, open(f"{test_result_path}/KNN_result_dict_{test_data_name}_{running_datetime_str}.pkl", "wb"))
            
    df.to_csv(f"KNN_result_df_{running_datetime_str}.csv", index=False)
if __name__ == "__main__":
    __main__()