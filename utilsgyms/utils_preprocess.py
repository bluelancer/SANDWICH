import torch
device = torch.device('cpu')
from WINeRTEnv import WINeRTEnv
import json
import numpy as np
from visualization.get_scene import *
from visualization.vis_winert import *
from utilsgyms.utils_winert_dataset import *
from utilsgyms.utils_decisiontrans import *
from utilsgyms.utils_preprocess import *

def get_rew_by_action_traj(gt_action_traj , env, init_state, append_receipt_loss, path_id, pertubation = None, return_ndarray = True):
    env.reset_init_path(init_path=path_id, after_stepping = False)
    env.reset() # 1st reset the 
    if pertubation is not None:
        env.state[1:4] =  env.state[1:4] + pertubation
    rewards_list = []
    states_list = []
    info_list = []            
    states_list.append(init_state)

    #TODO: for batched action, we need to iterate over the batch
    if len(gt_action_traj.shape) == 3:
        action_iter = gt_action_traj[...,1:,:].swapaxes(0,1) # action_iter.shape = [path_length, batch_size, 3]
    else:
        assert len(gt_action_traj.shape) == 2, f"gt_action_traj.shape = {gt_action_traj.shape}"
        action_iter = gt_action_traj[1:,:]
    # import ipdb; ipdb.set_trace()
    assert action_iter.shape[-1] == 3, f"action_iter.shape = {action_iter.shape}"
    assert action_iter.shape[0] == 4, f"action_iter.shape = {action_iter.shape}"
    for action in action_iter:
        state, rew, _, _ ,info  = env.step(action, without_assert = True)
        info["init_path"] = env.init_path
        info["Tx_id"] = env.Tx_id
        info["Rx_id"] = env.Rx_id
        if append_receipt_loss and info["done"]:
            rew += env.receipt_los
        info["receipt_los"] = env.receipt_los
        rewards_list.append(rew)
        states_list.append(state)
        info_list.append(json.dumps(info))
    if return_ndarray:
        return np.array(rewards_list), np.array(states_list), info_list
    else:
        return rewards_list, states_list, info_list

def assertion_test(max_incremental_index_for_filter_path, gt_action_traj, gt_state_traj, pred_action_traj, pred_state_traj, print_out = False):
    gt_action_traj = np.array(gt_action_traj)
    gt_state_traj = np.array(gt_state_traj)
    pred_action_traj = np.array(pred_action_traj)
    pred_state_traj = np.array(pred_state_traj)
    assert gt_action_traj.shape[-2:] == pred_action_traj.shape[-2:], f"gt_action_traj.shape = {gt_action_traj.shape}, pred_action_traj.shape = {pred_action_traj.shape}"
    assert gt_state_traj.shape[-2:] == pred_state_traj.shape[-2:], f"gt_state_traj.shape = {gt_state_traj.shape}, pred_state_traj.shape = {pred_state_traj.shape}"
    if gt_state_traj.shape != pred_state_traj.shape:
        pred_state_traj = pred_state_traj.reshape(gt_state_traj.shape)
    for i in range(max_incremental_index_for_filter_path - 1): # the Rx location should not be included in the assertion
        action_assertion = np.allclose(gt_action_traj[...,i,1:],pred_action_traj[...,i,1:],rtol = 5e-3 )
        state_assertion = np.allclose(gt_state_traj[...,i,:],pred_state_traj[...,i,:],atol = 1.5e-2)
        assertion_success = action_assertion and state_assertion
        if not assertion_success:
            if print_out:
                print ("i = ", i)
                print ("gt_action_traj[i,1:] = ", gt_action_traj[...,i,1:])
                print ("pred_action_traj[i,1:] = ", pred_action_traj[...,i,1:])
                print ("gt_state_traj[i] = ", gt_state_traj[...,i,:])
                print ("pred_state_traj[i] = ", pred_state_traj[...,i,:])
                print (f"gt_action_traj.shape = {gt_action_traj.shape}, pred_action_traj.shape = {pred_action_traj.shape}")
                print (f"gt_state_traj.shape = {gt_state_traj.shape}, pred_state_traj.shape = {pred_state_traj.shape}")
            return False
    return True                              

def gen_data_sample_across_rx(Tx_id,
                          env_index = 1,
                          Rx_id = None,
                          rx_range = None,
                          debug_path = False,
                          path = None,
                          add_noise = False,
                          local_node = False,
                          append_receipt_loss = False,
                          do_assertion = False,
                          debug = False,
                          test_env_list = None,
                          noise_sample = 20, 
                          test_pp = False):
    
    Tx_id, Rx_id, rx_range, debug_path, path, add_noise, local_node, append_receipt_loss, do_assertion, debug, rx_iterator, path_iterator = parse_gen_func_inputs (Tx_id,
                                                                                                                                                                                Rx_id,
                                                                                                                                                                                rx_range,
                                                                                                                                                                                debug_path,
                                                                                                                                                                                path,
                                                                                                                                                                                add_noise,
                                                                                                                                                                                local_node,
                                                                                                                                                                                append_receipt_loss,
                                                                                                                                                                                do_assertion,
                                                                                                                                                                                debug,
                                                                                                                                                                                test_pp = test_pp)

    skip_path = []
    unreachable_path_count = 0
    all_path_count = 0   
    if type(test_pp) == list:
        test_pp = test_pp[0]
    if test_pp:
        max_trying_times = 3
        mid_way = 2
    else:
        max_trying_times = 15
        mid_way = 8 
    for tx_id in Tx_id:
        for rx in rx_iterator: # generate the same path multiple times to reduce the epoch time
            test_env = WINeRTEnv(env_index = env_index, Tx_id = tx_id,Rx_id=rx, debug = local_node, test_pp = test_pp)
            #### Handing Test Set ####
            if test_pp: # if we are preprocessing the test data
                node_attr_tensor_coord = test_env.node_attr_tensor[tx_id,rx,:,:,]
                max_incremental_index_for_filter_path = test_env.all_max_incremental_index_for_filter_path[tx_id,rx]
                edge_attr_tensor_angle = test_env.edge_attr_tensor[tx_id,rx,:,:,:]
                path_range_index = torch.arange(node_attr_tensor_coord.size(0))
                hop_before_rx_loc = node_attr_tensor_coord[path_range_index, max_incremental_index_for_filter_path - 1, :4]
                last_act_index = max_incremental_index_for_filter_path - 1
                last_act = edge_attr_tensor_angle[path_range_index, last_act_index,:3]
                last_angle = last_act[...,1:3]  
            #### Handing Test Set End ####
            for path_id in path_iterator:
                all_path_count += 1
                if add_noise:
                    inner_iter = range(noise_sample[0])
                else:
                    inner_iter = [path_id]
                for noisy_path_id in inner_iter:
                    test_env.reset_init_path(init_path=path_id, after_stepping = False)
                    test_env.reset() # 1st reset the 
                    
                    assert test_env.Tx_id == tx_id, f"test_env.Tx_id = {test_env.Tx_id} != Tx_id = {tx_id}"
                    assert test_env.Rx_id == rx, f"test_env.Rx_id = {test_env.Rx_id} != Rx_id = {rx}"
                    assert test_env.init_path == path_id, f"test_env.init_path = {test_env.init_path} != path_id = {path_id}"
                    
                    if test_env.max_incremental_index_for_filter_path[test_env.init_path] < 2:
                        if test_env.init_path in skip_path:
                            continue
                        else:
                            skip_path.append(test_env.init_path)
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
                                    # print (f"assert_test_result = {assert_test_result}")
                                    # print (f"gt_action_from_env = {gt_action_from_env}")
                                    # print (f"action_pred = {action_pred}")
                                    print (f"tried_times = {tried_times}, pertubation_switch = {pertubation_switch}")
                                    print (f"gt_state_coord_from_env = {gt_state_coord_from_env}")
                                    print (f"state_pred[:,1:4] = {state_pred[:,1:4]}")
                                    # import ipdb; ipdb.set_trace()
                                    
                                if not assert_test_result:
                                    reachability = "Adjusted"
                                    if pertubation_switch: # if the 1st try failed, we try with coordinate pertubation
                                        pertubation = np.ones_like(test_env.init_value[1:4]) * (mid_way-tried_times)*1e-3
                                        rewards_list, states_list, info_list = get_rew_by_action_traj(gt_action_traj , 
                                                                                                    test_env, 
                                                                                                    init_state, 
                                                                                                    append_receipt_loss,
                                                                                                    path_id,
                                                                                                    pertubation = pertubation)
                                        
                                    else: # 1st try with angular pertubation, this covers 90% of the cases
                                        rewards_list, states_list, info_list = get_rew_by_action_traj(gt_action_traj + (mid_way-tried_times)*1e-4, 
                                                                                                    test_env, 
                                                                                                    init_state, 
                                                                                                    append_receipt_loss,
                                                                                                    path_id)
                                    tried_times += 1
                                else:
                                    try_again = False
                                if tried_times > max_trying_times and not pertubation_switch: # if the 1st try failed, we try with coordinate pertubation
                                    pertubation_switch = True
                                    tried_times = 0
                                if tried_times > max_trying_times and pertubation_switch:
                                    reachability = "False"
                                    unreachable_path_count += 1
                                    try_again = False
                                                                    
                        data["rews"] = np.array(rewards_list)
                        assert len(rewards_list) == 4 # TODO: adapt this to the length of the max_step
                        assert len(states_list) == 5 # TODO: adapt this to the length of the max_step
                        #### Handing Test Set ####
                        if test_pp:
                            action_type = gt_action_from_env[...,0].reshape(-1,1)
                            action_angular = gt_action_from_env[...,1:]
                            obs = np.concatenate([action_type, gt_state_coord_from_env, action_angular], axis = -1)
                            old_x = torch.concatenate((hop_before_rx_loc[path_id], last_angle[path_id]), dim = 0) # take the last action and the last coordinate
                            new_x = winert_step_np(old_x,
                                                test_env,
                                                last_act[path_id][1:3],
                                                last_act[path_id][0],
                                                torch.tensor(3), # not 3 as 3 means penetration 
                                                'cpu').to(test_env.device) # this step have to be on CPU
                            terminal_coord = new_x[1:4]
                            obs[max_incremental_index_for_filter_path[test_env.init_path] - 1:, 1:4] = terminal_coord
                            data["obs"] = obs
                            #### Handing Test Set End ####
                        else:
                            data["obs"] = np.array(states_list)
                        data["path_length"] = test_env.max_incremental_index_for_filter_path[test_env.init_path]
                        # TODO: try with init_step = 0
                        if test_env.set_init_path_flag:# if the path is set to the 1st hop
                            init_step = 1
                        else:
                            init_step = 0
                        data["terminal"] = [True if test_env.max_incremental_index_for_filter_path[test_env.init_path] - 1  < i else False for i in range(init_step,test_env.max_step)]
                        data["infos"] = info_list
                        action_angular = np.array(gt_action_traj[...,1:, 1:])
                        action_type_np = gt_action_traj[...,1:, 0]
                        action_type = np.array(
                            torch.nn.functional.one_hot(torch.tensor(action_type_np).long(), num_classes=6))
                        data["acts"] = np.concatenate([action_type, action_angular], axis = -1)
                        data["tried_times"] = tried_times  
                        data["path_id"] = path_id
                        data["reachability"] = reachability
                        if unreachable_path_count/all_path_count > 5e-2 and unreachable_path_count > 100 and not test_pp: 
                            # we dont want to have too many unreachable paths,
                            # but we keep the scheme in test_pp to have a better understanding of the unreachable paths
                            raise ValueError(f"all_path_count = {all_path_count}, unreachable_path_count = {unreachable_path_count}, unreachable_path_ratio = {unreachable_path_count/all_path_count}")
                        if all_path_count % 1000 == 0:
                            print (f"all_path_count = {all_path_count}, unreachable_path_count = {unreachable_path_count}, unreachable_path_ratio = {unreachable_path_count/all_path_count}")
                    yield data           
                         
# Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
def get_action(model, states, actions, rewards, returns_to_go, timesteps, return_dict = False, action_type_enabled = False):
    # This implementation does not condition on past rewards
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)
    
    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)

    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()

    action_padding = -10 * torch.ones((1, padding, model.config.act_dim))

    # This is last pad
    actions = torch.cat([action_padding, actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)
    output = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=return_dict,
    )
    if not return_dict:
        action_preds = output[1]
        action = action_preds[0, -1]
        if action_type_enabled:
            action_angular = action[-2:]
            action_onehot_type = action[:-2]
            action_prob = torch.nn.functional.softmax(action_onehot_type, dim = 0)
            action_type = torch.argmax(action_prob)
            action = torch.cat([action_type, action_angular])
        return action
    else:
        if action_type_enabled:
            raise NotImplementedError
        return output

def parse_gen_func_inputs(Tx_id, Rx_id, rx_range, debug_path, path, add_noise, local_node, append_receipt_loss, do_assertion, debug, test_pp = False):
    non_list_var = {"Rx_id": Rx_id,
                    "rx_range": rx_range,
                    "debug_path": debug_path,
                    "path": path, # if debug_path is True, path should be provided
                    "add_noise": add_noise,
                    "local_node": local_node,
                    "append_receipt_loss": append_receipt_loss,
                    "do_assertion": do_assertion,
                    "debug": debug}
    if type(Tx_id) == int:
        Tx_id = [Tx_id]
    for key in non_list_var.keys():
        non_list_var[key] = non_list_var[key][0]        
    if non_list_var["Rx_id"] is not None:
        rx_iter = [Rx_id]
    elif non_list_var["rx_range"] is not None:
        rx_iter = non_list_var["rx_range"]
    elif test_pp:
        rx_iter = range(1711)
    else:
        rx_iter = range(1800)
    if not non_list_var["debug_path"]:
        path_iter = range(30)
    else:
        path_iter = [path]
    return Tx_id, non_list_var["Rx_id"], non_list_var["rx_range"], non_list_var["debug_path"], non_list_var["path"], non_list_var["add_noise"], non_list_var["local_node"], non_list_var["append_receipt_loss"], non_list_var["do_assertion"], non_list_var["debug"], rx_iter, path_iter