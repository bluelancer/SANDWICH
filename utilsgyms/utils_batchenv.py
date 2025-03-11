import torch
device = torch.device('cpu')
from datasets import Dataset
import json
import numpy as np
import multiprocessing as mp
import os
import pickle
import argparse
from tqdm import tqdm

from WINeRTEnv import WINeRTEnv
from utilsgyms.utils_decisiontrans import *

                
def gen_data_sample_batch_rx(Tx_id,
                          Rx_id = None,
                          debug_path = False,
                          path = None,
                          add_noise = False,
                          local_node = False,
                          append_receipt_loss = False,
                          do_assertion = False,
                          debug = False,
                          optim_load = False,
                          recurrent_gen_times = 1,
                          multi_processing = False,
                          split_Rx = False,
                          batch_Rx = True):
    
    if type(debug_path) == list: # 
        assert len(debug_path) == 1, "debug_path should be a single value"
        debug_path = debug_path[0]
        debug = debug[0]
        add_noise = add_noise[0]
        do_assertion = do_assertion[0]
        append_receipt_loss = append_receipt_loss[0]
        local_node = local_node[0]

    if split_Rx and Rx_id is not None and Rx_id[0] is not None:
        assert Rx_id is not None, "Rx_id should not be None"
        assert type(Rx_id) == list, f"Rx_id should be a list, Rx_id = {Rx_id}, type(Rx_id) = {type(Rx_id)}, Tx_id = {Tx_id}, split_Rx = {split_Rx}"
        rx_iterator = Rx_id[0]
        assert len(list(rx_iterator)) == 1800/2, f"len(list(rx_iterator)) = {len(list(rx_iterator))}"
        assert len(Tx_id) == len(Rx_id) == 1, f"len(Tx_id) = {len(Tx_id)}, len(Rx_id) = {len(Rx_id)}"
        assert not debug_path, "split_Rx is enabled, debug_path should be False"
        assert not debug, "split_Rx is enabled, debug should be False"
        
    elif not split_Rx and type(Rx_id) == int:
        rx_iterator = [Rx_id]
        
    elif not split_Rx and Rx_id is None:
        assert Rx_id is None, f"Rx_id should be None, enumerate all Rx, got Rx_id = {Rx_id}, split_Rx = {split_Rx}"
        rx_iterator = tqdm(range(1800))
        if type(debug_path) == list:
            assert not debug_path[0], "Rx_id is not provided, debug_path should be False"
            add_noise = add_noise[0]
            debug_path = debug_path[0]
        else:
            assert not debug_path, "Rx_id is not provided, debug_path should be False"
    else:
        raise ValueError(f"Rx_id should be either int or list of int or None, Rx_id = {Rx_id}, split_Rx = {split_Rx}")
    if debug_path and not split_Rx:
        assert not split_Rx, "split_Rx is enabled, debug_path should be False"
        assert path is not None, "path is None"
        assert Rx_id is not None, "Rx_id is None"
        assert not split_Rx, "split_Rx is enabled, debug_path should be False"
        assert not add_noise, "add_noise should be False"
        path_iterator = [path] 
    else:
        path_iterator = range(30)
    skip_path = []
    unreachable_path_count = 0
    for rx in rx_iterator: # generate the same path multiple times to reduce the epoch time
        test_env = WINeRTEnv(Tx_id = Tx_id,Rx_id=rx, debug = local_node)
        for path_id in path_iterator:
            if add_noise:
                inner_iter = range(20)
            else:
                inner_iter = [path_id]
            for noisy_path_id in inner_iter:
                test_env.reset_init_path(init_path=path_id, after_stepping = False)
                test_env.reset() # 1st reset the 
                
                assert test_env.Tx_id == Tx_id, f"test_env.Tx_id = {test_env.Tx_id} != Tx_id = {Tx_id}"
                assert test_env.Rx_id == rx, f"test_env.Rx_id = {test_env.Rx_id} != Rx_id = {rx}"
                assert test_env.init_path == path_id, f"test_env.init_path = {test_env.init_path} != path_id = {path_id}"
                
                if test_env.max_incremental_index_for_filter_path[test_env.init_path] < 2:
                    if test_env.init_path in skip_path:
                        continue
                    else:
                        skip_path.append(test_env.init_path)
                        # print (f"Path {test_env.init_path} is not valid, since test_env.max_incremental_index_for_filter_path[test_env.init_path] = {test_env.max_incremental_index_for_filter_path[test_env.init_path]}")
                    continue
                else:
                    data = {}
                    action_angle_traj = test_env.edge_attr_tensor[test_env.Tx_id,test_env.Rx_id,test_env.init_path,:,1:3]
                    # apply noise on the action_angle_traj
                    if noisy_path_id == inner_iter[0]:
                        noise_level_azimuth = 0
                        noise_level_radian = 0
                    else:
                        noise_level_azimuth = np.random.uniform(0,1e-2,5).reshape(5,1).astype(np.float16)
                        noise_level_radian = np.random.uniform(0,1e-2, 5).reshape(5,1).astype(np.float16)
                        noise_level_azimuth[0] = 0
                        noise_level_radian[0] = 0
                        action_angle_traj = action_angle_traj + np.concatenate([noise_level_azimuth, noise_level_radian], axis = 1)
                    
                    action_type_traj = abs(np.expand_dims(test_env.node_attr_tensor[test_env.Tx_id,test_env.Rx_id,test_env.init_path,1:,0], axis=-1))
                    gt_action_traj = np.concatenate((action_type_traj,action_angle_traj),axis=-1)
    
                    assert gt_action_traj.shape[-2:] == torch.Size([5, 3])

                    if len(action_angle_traj.shape) == 3:
                        init_state_action = np.squeeze(action_angle_traj[:,0,:])
                    else:
                        init_state_action = action_angle_traj[0]
                    init_state = np.concatenate([test_env.init_value,init_state_action], axis = -1)
                    assert init_state.shape[-1] == 6, f"init_state.shape = {init_state.shape}"
                    try:
                        rewards_list, states_list, info_list = get_rew_by_action_traj(gt_action_traj, test_env, init_state, append_receipt_loss, path_id)
                    except:
                        raise ValueError(f"Error at path_id = {path_id}, Rx_id = {rx}")
                    try_again = True # This is still needed, as pyvista still can not be 100% accurate
                    tried_times = 0
                    pertubation_switch = False
                    reachability = "None"
                    if noisy_path_id == inner_iter[0]:
                        reachability = "True"
                        while do_assertion and try_again:
                            # Let's do an assertion for the first path -- path without noise
                            gt_action_from_env = test_env.gt_action
                            gt_state_coord_from_env = test_env.node_attr_tensor[test_env.Tx_id,test_env.Rx_id,test_env.init_path,1:,1:4]
                            state_pred = np.array(states_list)
                            action_pred = np.array(gt_action_traj)
                            assert_test_result = assertion_test(test_env.max_incremental_index_for_filter_path[test_env.init_path],
                                        gt_action_from_env,
                                        gt_state_coord_from_env,
                                        action_pred, state_pred[:,1:4],
                                        print_out = debug)  
                            if debug:
                                print (f"assert_test_result = {assert_test_result}")
                                print (f"gt_action_from_env = {gt_action_from_env}")
                                print (f"gt_state_coord_from_env = {gt_state_coord_from_env}")
                                print (f"action_pred = {action_pred}")
                                print (f"state_pred = {state_pred}")
                            if not assert_test_result:
                                reachability = "Adjusted"
                                if pertubation_switch: # if the 1st try failed, we try with coordinate pertubation
                                    pertubation = np.ones_like(test_env.init_value[1:4]) * (8-tried_times)*1e-3
                                    rewards_list, states_list, info_list = get_rew_by_action_traj(gt_action_traj , 
                                                                                                test_env, 
                                                                                                init_state, 
                                                                                                append_receipt_loss,
                                                                                                path_id,
                                                                                                pertubation = pertubation)
                                else: # 1st try with angular pertubation, this covers 90% of the cases
                                    rewards_list, states_list, info_list = get_rew_by_action_traj(gt_action_traj + (8-tried_times)*1e-4, 
                                                                                                test_env, 
                                                                                                init_state, 
                                                                                                append_receipt_loss,
                                                                                                path_id)
                                tried_times += 1
                            else:
                                try_again = False
                            if tried_times > 15 and not pertubation_switch: # if the 1st try failed, we try with coordinate pertubation
                                pertubation_switch = True
                                tried_times = 0
                            if tried_times > 15 and pertubation_switch:
                                # import ipdb; ipdb.set_trace()
                                reachability = "False"
                                unreachable_path_count += 1
                                try_again = False
                                                                
                    data["rews"] = np.array(rewards_list)
                    assert len(rewards_list) == 4 # TODO: adapt this to the length of the max_step
                    assert len(states_list) == 5 # TODO: adapt this to the length of the max_step
                    data["obs"] = np.array(states_list)
                    data["path_length"] = test_env.max_incremental_index_for_filter_path[test_env.init_path]
                    # TODO: try with init_step = 0
                    if test_env.set_init_path_flag:# if the path is set to the 1st hop
                        init_step = 1
                    else:
                        init_step = 0
                    data["terminal"] = [True if test_env.max_incremental_index_for_filter_path[test_env.init_path] - 1  < i else False for i in range(init_step,test_env.max_step)]
                    data["infos"] = info_list
                    # data["acts"] = np.array(gt_action_traj[1:])
                    action_angular = np.array(gt_action_traj[...,1:, 1:])
                    action_type_np = gt_action_traj[...,1:, 0]
                    action_type = np.array(
                        torch.nn.functional.one_hot(torch.tensor(action_type_np).long(), num_classes=6))
                    data["acts"] = np.concatenate([action_type, action_angular], axis = -1)
                    data["tried_times"] = tried_times  
                    data["path_id"] = path_id
                    data["reachability"] = reachability
                    if unreachable_path_count > 100:
                        raise ValueError(f"Unreachable path count is more than 100")
                yield data    