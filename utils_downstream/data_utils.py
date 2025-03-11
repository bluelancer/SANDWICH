import torch
device = torch.device('cpu')
import numpy as np
import pickle
from utilsgyms.utils_preprocess import *
import sys
import os
import re

def distance_torch(point_1, point_2):
    point_1_tensor = torch.tensor(point_1).clone()
    point_2_tensor = torch.tensor(point_2).clone()
    return torch.sqrt(torch.sum((point_1_tensor - point_2_tensor)**2))

def get_tensor(storage_dir, surface_index_path, env_index):
    # importlib.reload(sys.modules['WINeRTEnv'])
    del sys.modules["WINeRTEnv"]
    from WINeRTEnv import WINeRTEnv
    # delete and reload the module
    env = WINeRTEnv(storage_dir = storage_dir,
                    surface_index_path =surface_index_path,
                    env_index = env_index,
                    debug= True)
    node_attr_tensor = env.node_attr_tensor
    edge_attr_tensor = env.edge_attr_tensor
    gt_tensor = env.gt_tensor
    all_max_incremental_index = env.all_max_incremental_index
    all_max_incremental_index_for_filter_path = env.all_max_incremental_index_for_filter_path
    reduced_mask = env.reduced_mask
    return node_attr_tensor, edge_attr_tensor, gt_tensor, all_max_incremental_index, all_max_incremental_index_for_filter_path, reduced_mask

def get_data_dict(env_index,
                  config_file,
                  current_base_path,
                  data_postfix,
                  test_name,
                  return_path = "repo_path",
                  debug_print = False,
                  prep_baseline = False, 
                  cluster = "berzelius",
                  baseline_type = "MLP"): 
    repo_path, train_datetime_str = find_latest_train_str(current_base_path, data_postfix, env_index, baseline = prep_baseline)
    result_path = f"{repo_path}HF_{test_name}_Result{data_postfix}_{train_datetime_str}"
    if cluster == "berzelius":
        train_storage_dir = config_file["DIRECTORIES"]["berzelius_train_storage_dir"]
        train_surface_index_path = config_file["DIRECTORIES"]["berzelius_train_surface_index_path"]
        if test_name == "test":
            test_storage_dir = config_file["DIRECTORIES"]["berzelius_test_storage_dir"]
            test_surface_index_path = config_file["DIRECTORIES"]["berzelius_test_surface_index_path"]
        elif test_name == "genz":
            test_storage_dir = config_file["DIRECTORIES"]["berzelius_genz_storage_dir"]
            test_surface_index_path = config_file["DIRECTORIES"]["berzelius_genz_surface_index_path"]
        elif test_name == "gendiag":
            test_storage_dir = config_file["DIRECTORIES"]["berzelius_gendiag_storage_dir"]
            test_surface_index_path = config_file["DIRECTORIES"]["berzelius_gendiag_surface_index_path"]
        else:
            raise ValueError("test_name should be either 'test' or 'genz'")
    elif cluster == "tetralith":
        train_storage_dir = config_file["DIRECTORIES"]["tetralith_train_storage_dir"]
        train_surface_index_path = config_file["DIRECTORIES"]["tetralith_train_surface_index_path"]
        if test_name == "test":
            test_storage_dir = config_file["DIRECTORIES"]["tetralith_test_storage_dir"]
            test_surface_index_path = config_file["DIRECTORIES"]["tetralith_test_surface_index_path"]
        elif test_name == "genz":
            test_storage_dir = config_file["DIRECTORIES"]["tetralith_genz_storage_dir"]
            test_surface_index_path = config_file["DIRECTORIES"]["tetralith_genz_surface_index_path"]
        elif test_name == "gendiag":
            test_storage_dir = config_file["DIRECTORIES"]["tetralith_gendiag_storage_dir"]
            test_surface_index_path = config_file["DIRECTORIES"]["tetralith_gendiag_surface_index_path"]
        else: 
            raise ValueError("test_name should be either 'test' or 'genz'")
    else:
        raise ValueError("cluster should be either 'berzelius' or 'tetralith'")
    if debug_print:
        print (f"==== loading from result_path = {result_path} ====")
        print (f"==== loading from train_storage_dir = {train_storage_dir} ====")
        print (f"==== loading from train_surface_index_path = {train_surface_index_path} ====")
        print (f"==== loading from test_storage_dir = {test_storage_dir} ====")
        print (f"==== loading from test_surface_index_path = {test_surface_index_path} ====")
    ### acquire dt prediction RT nodes####
    pred_node_attr_path = f"{result_path}/state_tensor_pred_env_{env_index}.pkl"
    pred_edge_attr_path = f"{result_path}/edge_tensor_pred_env_{env_index}.pkl"
    pred_node_attr_tensor = pickle.load(open(pred_node_attr_path, 'rb'))
    pred_edge_attr_tensor = pickle.load(open(pred_edge_attr_path, 'rb'))

    ### acquire train/test data ####
    train_node_attr_tensor, train_edge_attr_tensor, train_gt_tensor, train_all_max_incremental_index, train_all_max_incremental_index_for_filter_path, train_reduced_mask =  get_tensor(train_storage_dir, train_surface_index_path, env_index)
    test_node_attr_tensor, test_edge_attr_tensor, test_gt_tensor, test_all_max_incremental_index, test_all_max_incremental_index_for_filter_path, test_reduced_mask =  get_tensor(test_storage_dir, test_surface_index_path, env_index)
    if debug_print:
        print ("==== Train_data, Test_data, Pred_data loaded ====")
        print (f"loading Pred_data type = {test_name}, env_index = {env_index}, train_datetime_str = {train_datetime_str}")
    assert train_node_attr_tensor.shape[:2] == train_gt_tensor.shape[:2], "train_node_attr_tensor.shape[:2] != train_gt_tensor.shape[:2]"
    assert test_node_attr_tensor.shape[:2] == test_gt_tensor.shape[:2], "test_node_attr_tensor.shape[:2] != test_gt_tensor.shape[:2]"
    assert train_datetime_str is not None, "train_datetime_str is None"
    assert data_postfix is not None, "data_postfix is None"
    assert env_index is not None, "env_index is None"

    ########################## ####§
    ##### Prep Train/Test data #####
    ########################## ####§
    # Prepare the train/test data dist tensor
    train_edge_dist_tensor = train_edge_attr_tensor[:,:,:,:,0]
    train_edge_dist_tensor = torch.mul(train_reduced_mask[...,1:],train_edge_dist_tensor).unsqueeze(-1)
    test_edge_dist_tensor = test_edge_attr_tensor[:,:,:,:,0]
    test_edge_dist_tensor = torch.mul(test_reduced_mask[...,1:],test_edge_dist_tensor).unsqueeze(-1)

    expanded_index_train = train_all_max_incremental_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 4)
    train_node_attr_tensor_clean = train_node_attr_tensor[...,:4]
    train_rx_coord = torch.gather(train_node_attr_tensor_clean, dim=3, index=expanded_index_train[...,:4])
    ##########################
    ##### Prep pred data #####
    ##########################
    pred_edge_attr_tensor_clip = pred_edge_attr_tensor[:,:,:,:4,:3]
    # 1st action is missed in pred_edge_attr_tensor, so we need to add it back
    first_action = test_edge_attr_tensor[:,:,:,0,:3].unsqueeze(-2)
    # Extract the first action from test_edge_attr_tensor
    pred_edge_attr_tensor_clean = torch.cat([first_action, pred_edge_attr_tensor_clip], dim = -2)

    expanded_index_test = test_all_max_incremental_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 7)    
    test_node_attr_tensor_clean = test_node_attr_tensor[...,:4]
    rx_coord = torch.gather(test_node_attr_tensor_clean, dim=3, index=expanded_index_test[...,:4])
    test_rx_coord = rx_coord # just a grammer sugar
    
    pred_node_attr_tensor_clean = pred_node_attr_tensor[...,:4]
    pred_node_attr_tensor_clean = pred_node_attr_tensor_clean.scatter_(3, expanded_index_test[...,:4], rx_coord)
    # TODO: compute the distance between the Tx and Rx
    pred_node_pair = torch.cat([pred_node_attr_tensor_clean[:,:,:,:,1:4],
                        torch.roll(pred_node_attr_tensor_clean[:,:,:,:,1:4],
                                    1, dims=-2)], dim=-1)[:,:,:,1:,:]  
    pred_edge_dist = torch.norm(pred_node_pair[:,:,:,:,0:3] - pred_node_pair[:,:,:,:,3:6], dim=4)
    # test_edge_dist_tensor = torch.mul(test_reduced_mask[...,1:],test_edge_dist_tensor)
    pred_edge_dist_tensor = torch.mul(test_reduced_mask[...,1:], pred_edge_dist).unsqueeze(-1)
    
    train_ds_dict = {"train_coord": train_node_attr_tensor[:,:,:,:,1:4],
                    "train_interType": train_node_attr_tensor[:,:,:,:,0].unsqueeze(-1),
                    "train_edge_attr_tensor": train_edge_attr_tensor[:,:,:,:,1:3],
                    "train_RSSI": train_gt_tensor[...,0,:].unsqueeze(-2),
                    "train_Phase": train_gt_tensor[...,1,:].unsqueeze(-2),
                    "train_ToA": train_gt_tensor[...,2,:].unsqueeze(-2),
                    "train_AoD": train_gt_tensor[...,3:5,:],
                    "train_AoA": train_gt_tensor[...,5:7,:],
                    "train_valid_mask": train_gt_tensor[...,7,:].unsqueeze(-2),
                    "train_dist": train_edge_dist_tensor,
                    "train_path_length": train_all_max_incremental_index_for_filter_path.unsqueeze(-1),
                    "train_Tx_coord": train_node_attr_tensor[:,:,:,0,1:4],
                    "train_Rx_coord": train_rx_coord,
                    }
    
    pred_ds_dict = {"pred_coord": pred_node_attr_tensor_clean[:,:,:,:,1:4],
                    "pred_interType": pred_node_attr_tensor_clean[:,:,:,:,0].unsqueeze(-1),
                    "pred_edge_attr_tensor": pred_edge_attr_tensor_clean,
                    "pred_dist": pred_edge_dist_tensor,
                    "pred_path_length": test_all_max_incremental_index_for_filter_path.unsqueeze(-1),
                    "pred_Tx_coord": test_node_attr_tensor_clean[:,:,:,0,1:4],
                    "pred_Rx_coord": test_rx_coord, # same as test_rx_coord
                    }
    
    test_ds_dict = {"test_coord": test_node_attr_tensor[:,:,:,:,1:4],
                    "test_interType": test_node_attr_tensor[:,:,:,:,0].unsqueeze(-1),
                    "test_edge_attr_tensor": test_edge_attr_tensor[:,:,:,:,1:3],
                    "test_RSSI": test_gt_tensor[...,0,:].unsqueeze(-2),
                    "test_Phase": test_gt_tensor[...,1,:].unsqueeze(-2),
                    "test_ToA": test_gt_tensor[...,2,:].unsqueeze(-2),
                    "test_AoD": test_gt_tensor[...,3:5,:],
                    "test_AoA": test_gt_tensor[...,5:7,:],
                    "test_valid_mask": test_gt_tensor[...,7,:].unsqueeze(-2),
                    "test_dist": test_edge_dist_tensor, #test_edge_dist
                    "test_path_length": test_all_max_incremental_index_for_filter_path.unsqueeze(-1),
                    "test_Tx_coord": test_node_attr_tensor[:,:,:,0,1:4],
                    "test_Rx_coord": test_rx_coord,
                    }
    if prep_baseline and baseline_type == "MLP":
        train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor = prep_dataset(["Tx_coord", "Rx_coord","path_length"], ["RSSI", "edge_attr_tensor"], train_ds_dict, test_ds_dict, pred_ds_dict, prep_baseline = True)
    elif prep_baseline and baseline_type == "KNN":
        train_Tx_coord_path_len = torch.cat([train_ds_dict["train_Tx_coord"], train_ds_dict["train_path_length"]], dim = -1)
        test_Tx_coord_path_len = torch.cat([test_ds_dict["test_Tx_coord"], test_ds_dict["test_path_length"]], dim = -1)
        train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor = train_Tx_coord_path_len, test_Tx_coord_path_len, None, train_ds_dict["train_RSSI"], test_ds_dict["test_RSSI"]
    else:
        train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor = prep_dataset(["coord", "interType", "dist", "path_length"], ["RSSI"], train_ds_dict, test_ds_dict, pred_ds_dict,prep_baseline = False)
    del train_ds_dict, test_ds_dict, pred_ds_dict
    
    if return_path == "repo_path":
        return train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor, repo_path
    elif return_path == "result_path":
        return train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor, result_path
    else:
        raise ValueError("return_path should be either 'repo_path' or 'result_path'")

def prep_dataset(feature_name_list, label_name_list, train_ds_dict, test_ds_dict, pred_ds_dict, type = "RT", prep_baseline = False):
    for feature_name in feature_name_list:
        assert feature_name in ["coord", "interType", "edge_attr_tensor", "valid_mask", "dist", "path_length","Tx_coord", "Rx_coord"], f"feature_name = {feature_name} is not in ['coord', 'interType', 'edge_attr_tensor', 'valid_mask', 'dist', 'path_length']"
    for label_name in label_name_list:
        assert label_name in ["RSSI", "Phase", "ToA", "AoD", "AoA", "edge_attr_tensor"], f"label_name = {label_name} is not in ['RSSI', 'Phase', 'ToA', 'AoD', 'AoA', 'edge_attr_tensor']"
    train_feature_name_list = []
    test_feature_name_list = []
    pred_feature_name_list = []
    for feature_name in feature_name_list:
        train_feature_name_list.append(f"train_{feature_name}")
        test_feature_name_list.append(f"test_{feature_name}")
        pred_feature_name_list.append(f"pred_{feature_name}")
    train_label_name_list = []
    test_label_name_list = []
    for label_name in label_name_list:
        train_label_name_list.append(f"train_{label_name}")
        test_label_name_list.append(f"test_{label_name}")
        
    train_ds_tensors = []
    test_ds_tensors = []
    pred_ds_tensors = []

    train_ds_tensors = from_name_to_tensor(train_feature_name_list, train_ds_dict)
    test_ds_tensors = from_name_to_tensor(test_feature_name_list, test_ds_dict)
    pred_ds_tensors = from_name_to_tensor(pred_feature_name_list, pred_ds_dict)
    train_feat_tensor = torch.cat(train_ds_tensors, dim = -1)
    test_feat_tensor = torch.cat(test_ds_tensors, dim = -1)
    pred_feat_tensor = torch.cat(pred_ds_tensors, dim = -1)
    
    assert len(train_label_name_list) == len(test_label_name_list), "len(train_label_name_list) != len(test_label_name_list)"
    if not prep_baseline:
        assert len(train_label_name_list)== 1, "len(train_label_name_list) != 1"
        train_label_tensor = train_ds_dict[train_label_name_list[0]]
        test_label_tensor = test_ds_dict[test_label_name_list[0]]
        train_label_tensor = train_label_tensor.reshape(-1, len(train_label_name_list)).float()
        test_label_tensor = test_label_tensor.reshape(-1, len(train_label_name_list)).float()
    else:
        list_train_label_tensor = []
        list_test_label_tensor = []
        for label_index in range(len(train_label_name_list)):
            train_label_name = train_label_name_list[label_index]
            test_label_name = test_label_name_list[label_index]
            if train_label_name.endswith("edge_attr_tensor"):
                list_train_label_tensor.append(train_ds_dict[train_label_name].reshape(-1, train_ds_dict[train_label_name].shape[-1] * train_ds_dict[train_label_name].shape[-2]))
                list_test_label_tensor.append(test_ds_dict[test_label_name].reshape(-1, test_ds_dict[test_label_name].shape[-1] * test_ds_dict[test_label_name].shape[-2]))
            else:
                list_train_label_tensor.append(train_ds_dict[train_label_name].reshape(-1, 1))
                list_test_label_tensor.append(test_ds_dict[test_label_name].reshape(-1, 1))
                
        train_label_tensor = torch.cat(list_train_label_tensor, dim = -1)
        test_label_tensor = torch.cat(list_test_label_tensor, dim = -1)

    # reshape the tensor
    train_feat_tensor = train_feat_tensor.reshape(-1, train_feat_tensor.shape[-1])
    test_feat_tensor = test_feat_tensor.reshape(-1, test_feat_tensor.shape[-1])
    pred_feat_tensor = pred_feat_tensor.reshape(-1, pred_feat_tensor.shape[-1])

    return train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor
    
def from_name_to_tensor(name_list, ds_dict):
    tensor_list = []
    for name in name_list:
        if name == "train_interType" or name == "test_interType" or name == "pred_interType":
            data_tensor = torch.abs(ds_dict[name])
            one_hot_tensor = torch.nn.functional.one_hot(data_tensor.to(torch.int64), num_classes = 6).squeeze(-2)
            tensor_list.append(one_hot_tensor.reshape(-1, one_hot_tensor.shape[-1]* one_hot_tensor.shape[-2]))
        elif name.endswith("path_length"):
            tensor_list.append(ds_dict[name].reshape(-1, 1))
        elif name.endswith("x_coord"):
            tensor_list.append(ds_dict[name].reshape(-1, ds_dict[name].shape[-1]))
        else:
            tensor_list.append(ds_dict[name].reshape(-1, ds_dict[name].shape[-1] * ds_dict[name].shape[-2]))
    return tensor_list
        
def find_latest_train_str(current_base_path, data_postfix, env_index, baseline = False):
    if baseline:
        print ("baseline is True")
        repo_path = f"/proj/raygnn_storage/outputs/huggingface_test_result{data_postfix}/models/env_id_index/env_id_{env_index}/"
    else:
        # repo_path = f"{current_base_path}/../../../../outputs/huggingface_trained_dt{data_postfix}/"
        repo_path = f"{current_base_path}/../../../../outputs/huggingface_test_result{data_postfix}/models/env_id_index/env_id_{env_index}/"
    repo_pattern = f"huggingface_trained_dt{data_postfix}_\d{{8}}-\d{{6}}_{env_index}"
    train_dt_str_matches = []
    for repo in os.listdir(repo_path):
        if re.match(repo_pattern, repo):
            train_dt_str_match = repo.split("_")[-2]
            train_dt_str_matches.append(train_dt_str_match)
    if len(train_dt_str_matches) == 0:
        raise ValueError("No model is found")
    elif len(train_dt_str_matches) == 1:
        train_dt_str = train_dt_str_matches[0]
    else:
        print (f"multiple models are found, using the latest one, {max(train_dt_str_matches)}")
        train_dt_str = max(train_dt_str_matches)
    return repo_path, train_dt_str
    