import os
os.environ["WANDB_PROJECT"]="wirelessRT"
os.environ["WANDB_CACHE_DIR"]="../wandb_cache"
import torch
from WINeRTEnv import WINeRTEnv
import numpy as np
# for plotting
from visualization.get_scene import *
from visualization.vis_winert import *
from utilsgyms.utils_winert_dataset import *
from utilsgyms.utils_decisiontrans import *
from utilsgyms.utils_preprocess import *
from utilsgyms.TrainableDT import *
from utilsgyms.DecisionTransformerGymDataCollatorTensor import *
from utilsgyms.draw_figure import *
from utils_downstream.data_utils import *

from transformers import DecisionTransformerConfig
import pickle
from datetime import datetime
from datasets import load_from_disk
import pickle
from tqdm import tqdm
TEST_SET = ["test", "genz",'gendiag']
# TODO: use Hyperparameter tuning with Ray Tune https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
def __main__():
    env_index, _, _, Tx_id, allTx, _, _,  add_noise, _, _, _, _, _, _, _,test_seed, eval_from, data_prefix, data_postfix, _, _, noise_sample, test_data, add_type_loss = parse_input()
    seed_everything(test_seed)
    config_file, current_base_path = get_config_basepath(allTx=allTx)
    data_postfix = "_noise"
    if noise_sample is not None and noise_sample != 20:
        data_postfix = f"{data_postfix}_noise_{noise_sample}"
    else :
        data_postfix = f"{data_postfix}_{noise_sample}"
    if not add_noise:
        current_base_path = current_base_path.replace("train", "trainwoAug")
        type_loss = add_type_loss
    else:
        type_loss = True
    add_surface_index = False

    train_ds_path = find_trainpp_repo(current_base_path,data_prefix, data_postfix, env_index)

    if not add_noise and noise_sample == 20 and add_type_loss:
        data_postfix = f"{data_postfix}_wType"
    train_ds = load_from_disk(train_ds_path)
    train_ds_withRx = train_ds.map(attach_Rxloc)
    print ("========== Training Data is loaded ==========")
    # Load the test data
    assert test_data in TEST_SET, f"test_data should be in {TEST_SET}"
    if test_data == "test":
        test_node_repo = config_file["DIRECTORIES"]["berzelius_test_storage_dir"]
        test_surface_index_path = config_file["DIRECTORIES"]["berzelius_test_surface_index_path"]
    elif test_data == "genz":
        test_node_repo = config_file["DIRECTORIES"]["berzelius_genz_storage_dir"]
        test_surface_index_path = config_file["DIRECTORIES"]["berzelius_genz_surface_index_path"]
    elif test_data == "gendiag":
        test_node_repo = config_file["DIRECTORIES"]["berzelius_gendiag_storage_dir"]
        test_surface_index_path = config_file["DIRECTORIES"]["berzelius_gendiag_surface_index_path"]
    else:
        raise ValueError(f"test_data should be in {TEST_SET}")
    if bool(add_surface_index):
        print ("========== Loading surface index ==========")
        load_node_attr_tensor(env_index,
                              test_node_repo,
                              test_surface_index_path)
        prepare_surface_index_tensor()
        train_ds_wRx_wSurface = train_ds_withRx.map(attach_SurfaceIndex)
        train_ds_withRx = train_ds_wRx_wSurface
        
    collator = DecisionTransformerGymDataCollatorTensor(train_ds_withRx, batch_operations = True, debug = False)
    config = DecisionTransformerConfig(state_dim=collator.state_dim,
                                       act_dim=collator.act_dim,
                                       n_layer = int(config_file['Hyperparameters']['n_layer']),
                                       max_ep_len= int(config_file["DATAParam"]['max_ep_len']),
                                       resid_pdrop= float(config_file['Hyperparameters']['resid_pdrop']),
                                       embd_pdrop = float(config_file['Hyperparameters']['embd_pdrop']),
                                       action_tanh = False,
                                       use_cache=True)
    print ("========== Model is initialized ==========")
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    model = TrainableDT(config, len_ds = len(train_ds_withRx), surface_id = add_surface_index, type_loss = type_loss)
    # ======= Load the model for testing =======
    if eval_from is None:
        train_dt_str = "20240611-202934" # this shall be modified, DEFAULT MODEL
        print("==== Warning: eval_from is None, using the default model ====")
        loading_model_path = os.path.join(current_base_path, f"../../../../outputs/huggingface_test_result{data_postfix}/models/huggingface_trained_dt{data_postfix}_{train_dt_str}_{env_index}/")
        print (f"loading the model for replotting, from: {loading_model_path}")
        model = model.from_pretrained(loading_model_path)
    elif eval_from == "latest":
        print ("loading the latest model for replotting")
        model_path, train_dt_str = find_latest_train_str(current_base_path, data_postfix, env_index)
        model = model.from_pretrained(f"{current_base_path}/../../../../outputs/huggingface_test_result{data_postfix}/models/env_id_index/env_id_{env_index}/huggingface_trained_dt{data_postfix}_{train_dt_str}_{env_index}/")
        print (f"loading the model for replotting, from: f{current_base_path}/../../../../outputs/huggingface_test_result{data_postfix}/models/env_id_index/env_id_{env_index}/huggingface_trained_dt{data_postfix}_{train_dt_str}_{env_index}/")
    else:
        print (f"loading the model for replotting, from: {eval_from}")
        train_dt_str = eval_from
        loading_model_path = os.path.join(current_base_path, f"../../../../outputs/huggingface_test_result{data_postfix}/models/huggingface_trained_dt{data_postfix}_{train_dt_str}_{env_index}/")
        print (f"loading the model for replotting, from: {loading_model_path}")
        model = model.from_pretrained(loading_model_path)
    model.eval()
    output_datetime_str = train_dt_str
    print ("========== Model is loaded ==========")
    result_path = f"{current_base_path}/../../../../outputs/huggingface_test_result{data_postfix}/models/env_id_index/env_id_{env_index}/HF_{test_data}_Result{data_postfix}_{output_datetime_str}"
    os.makedirs(result_path, exist_ok=True)
    pickle.dump(model, open(f"{result_path}/model.pkl", "wb"))
    print (f"========== Result path is created: {result_path} ==========")
    max_ep_len, device, scale, TARGET_RETURN, state_mean, state_std, state_dim, act_dim = prepare_HP_for_validation(collator)

    all_rewards_dict = {}
    all_log_ave_ang_loss_dict = {}
    all_path_length_dict = {}
    if test_data == "test" or test_data == "genz":
        Tx_num = int(config_file["DATAParam"]['Tx_num_test']) # train set has 10 Tx, test set has 15 Tx
        Rx_num = int(config_file["DATAParam"]['Rx_num_test']) # train set has 1800 Rx, test set has 1711 Rx
    elif test_data == "gendiag":
        Tx_num = int(config_file["DATAParam"]['Tx_num_gendiag'])
        Rx_num = int(config_file["DATAParam"]['Rx_num_gendiag'])
    else:
        raise ValueError(f"test_data should be in {TEST_SET}") 
    if allTx:
        Tx_range = range(Tx_num)
    else:
        Tx_range = [Tx_id]
    # CHECKED: test env is identical to the one used in training, env mesh should be same
    if add_surface_index:
        state_tensor_pred = -torch.ones((Tx_num,Rx_num, 30, 6, config.state_dim - 18), device=device)
        edge_tensor_pred = -torch.ones((Tx_num,Rx_num, 30, 6, config.act_dim - 18), device=device)
    else:
        state_tensor_pred = -torch.ones((Tx_num,Rx_num, 30, 6, config.state_dim), device=device)
        edge_tensor_pred = -torch.ones((Tx_num,Rx_num, 30, 6, config.act_dim), device=device)
        
    state_tensor_pred = -torch.ones((Tx_num,Rx_num, 30, 6, config.state_dim), device=device)
    edge_tensor_pred = -torch.ones((Tx_num,Rx_num, 30, 6, config.act_dim), device=device)
    for tx_id in tqdm(Tx_range):
        for test_rx in range(Rx_num):
            env = WINeRTEnv(
                storage_dir =test_node_repo,
                surface_index_path =test_surface_index_path,
                env_index = env_index,
                Tx_id= tx_id, 
                Rx_id=test_rx, 
                debug= True)
            assert env.env_index == env_index, f"env_index is not correct: {env.env_index} != {env_index}"
            # get Rx coordinates, conduct the last ray direction on the rx location and find the intersection point
            # attach the intersection point to the init state of each path
            node_attr_tensor_coord = env.node_attr_tensor[tx_id,test_rx,:,:,]
            max_incremental_index_for_filter_path = env.all_max_incremental_index_for_filter_path[tx_id,test_rx]
            edge_attr_tensor_angle = env.edge_attr_tensor[tx_id,test_rx,:,:,:]

            path_range_index = torch.arange(node_attr_tensor_coord.size(0))
            hop_before_rx_loc = node_attr_tensor_coord[path_range_index, max_incremental_index_for_filter_path - 1, :4]

            last_act_index = max_incremental_index_for_filter_path - 1
            last_act = edge_attr_tensor_angle[path_range_index, last_act_index,:3]
            last_angle = last_act[...,1:3]
            if type(state_mean) == np.ndarray:
                state_mean = torch.from_numpy(state_mean).to(device=device)
                state_std = torch.from_numpy(state_std).to(device=device)
            else:
                pass

            p_iterator = range(30)
            rewards_dict = {}
            log_ave_ang_loss_dict = {} 
            path_length_dict = {}
            for p_id in p_iterator:
                if add_surface_index:
                    surface_id_seq_onehot = get_surface_index_tensor(tx_id, test_rx, p_id) # shape == (6, 18)
                if env.max_incremental_index_for_filter_path[p_id] > 1:
                    old_x = torch.concatenate((hop_before_rx_loc[p_id], last_angle[p_id]), dim = 0)
                    new_x = winert_step_np(old_x,
                                        env,
                                        last_act[p_id][1:3],
                                        last_act[p_id][0],
                                        torch.tensor(3), # not 3 as 3 means penetration
                                        'cpu').to(env.device) # this step have to be on CPU
                    
                    terminal_coord = new_x[1:4]
                    episode_return, episode_length = 0, 0
                    env.reset_init_path(after_stepping = False,init_path=p_id)
                    init_state, _ = env.reset()
                    init_state_pad_terminal = np.concatenate((init_state, terminal_coord.cpu().numpy()), axis=0)
                    if add_surface_index:
                        init_state_pad_terminal_pad_surface = np.concatenate((init_state_pad_terminal, surface_id_seq_onehot[1]), axis=0)
                        state = init_state_pad_terminal_pad_surface
                    else:
                        state = init_state_pad_terminal
                    
                    state_list = []
                    action_list = []
                    target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
                    state_list.append(state)
                    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
                    
                    actions = torch.zeros((0, act_dim),
                                        device=device,
                                        dtype=torch.float32)
                    rewards = torch.zeros(0,
                                        device=device,
                                        dtype=torch.float32)
                    timesteps = torch.tensor(0,
                                            device=device,
                                            dtype=torch.long).reshape(1, 1)
                    per_step_inputs = []
                        
                    for t in range(max_ep_len):
                        actions = torch.cat([actions, -10 * torch.ones((1, act_dim), device=device)], dim=0)
                        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
                        if add_surface_index and t > 0: # add surface index to the state, from 2nd step
                            action_onehot_surface = action_onehot_surface.reshape(1, -1)
                            states = np.concatenate((states, action_onehot_surface), axis=0)
                        normed_states = (states - state_mean) / state_std
                        step_inputs = {
                            "states": states,
                            "actions": actions,
                            "rewards": rewards,
                            "returns_to_go": target_return,
                            "timesteps": timesteps,
                        }
                        per_step_inputs.append(step_inputs)
                        action_onehot_type_anglar = get_action(
                            model, 
                            normed_states, 
                            actions, 
                            rewards,
                            target_return,
                            timesteps,
                        )
                        actions[-1] = action_onehot_type_anglar
                        action_onehot_type_anglar = action_onehot_type_anglar.detach().cpu().numpy()
                        
                        if add_surface_index:
                            action_onehot_type = action_onehot_type_anglar[:6]
                            action_anglar = action_onehot_type_anglar[6:8]
                            action_onehot_surface = action_onehot_type_anglar[8:]
                            action_type = np.argmax(action_onehot_type).reshape(1)
                            # action surface is not used in the reward calculation, but we need to keep it for the next step
                            action_surface = np.argmax(action_onehot_surface).reshape(1)
                            assert action_onehot_surface.shape[0] == 18, f"action_onehot_surface.shape[0] != 18: {action_onehot_surface.shape[0]}"
                        else:
                            action_onehot_type = action_onehot_type_anglar[:-2]
                            action_anglar = action_onehot_type_anglar[-2:]
                            action_type = np.argmax(action_onehot_type).reshape(1)

                        action = np.concatenate((action_type, action_anglar), axis=0)
                        state, reward, done, _, info = env.step(action, without_assert=True)
                        
                        # add terminal coordinate to the state
                        state_pad_terminal = np.concatenate((state, terminal_coord.cpu().numpy()), axis=0)
                        if add_surface_index:
                            state_pad_terminal_pad_surf = np.concatenate((state_pad_terminal, surface_id_seq_onehot[1]), axis=0)
                            state_list.append(state_pad_terminal_pad_surf)
                            action_list.append(action)
                            cur_state = torch.from_numpy(state_pad_terminal_pad_surf).to(device=device).reshape(1, state_dim)
                        else:
                            state_list.append(state_pad_terminal)
                            action_list.append(action)            
                            cur_state = torch.from_numpy(state_pad_terminal).to(device=device).reshape(1, state_dim)
                        states = torch.cat([states, cur_state], dim=0)
                        
                        rewards[-1] = reward
                        pred_return = target_return[0, -1] - (reward / scale)
                        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
                        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
                        episode_return += reward
                        episode_length += 1
                        if done:
                            log_ave_ang_loss_dict[env.init_path] = - np.log(info["mean_angular_loss"])
                            rewards_dict[env.init_path] = reward
                            path_length_dict[env.init_path] = env.max_incremental_index_for_filter_path[p_id]
                            per_Tx_Rx_path_state_tensor = torch.tensor(np.array(state_list))
                            per_Tx_Rx_path_edge_tensor = torch.tensor(np.array(action_list))
                            # prepare the state tensor and edge tensor, for later training
                            state_tensor_pred[tx_id,test_rx,p_id, 1:per_Tx_Rx_path_state_tensor.shape[0]+1, :6] = per_Tx_Rx_path_state_tensor[:,:6]
                            state_tensor_pred[tx_id,test_rx,p_id, 0, :6] = torch.tensor(env.state_traj[0]) # initial Tx location
                            state_tensor_pred[tx_id,test_rx,p_id, :per_Tx_Rx_path_state_tensor.shape[0], 6:9] = env.rx_coord # terminal Rx location
                            edge_tensor_pred[tx_id, test_rx, p_id, :per_Tx_Rx_path_edge_tensor.shape[0], :per_Tx_Rx_path_edge_tensor.shape[1]] = per_Tx_Rx_path_edge_tensor
                            break
                else:
                    pass

            all_rewards_dict[(tx_id, test_rx)] = rewards_dict
            all_log_ave_ang_loss_dict[(tx_id, test_rx)] = log_ave_ang_loss_dict
            all_path_length_dict[(tx_id, test_rx)] = path_length_dict
        print ("========== Validation is done ==========")
            
    pickle.dump(state_tensor_pred, open(f"{result_path}/state_tensor_pred_env_{env_index}.pkl", "wb"))
    pickle.dump(edge_tensor_pred, open(f"{result_path}/edge_tensor_pred_env_{env_index}.pkl", "wb"))
    pickle.dump(all_log_ave_ang_loss_dict, open(f"{result_path}/all_log_ave_ang_loss_dict_{datetime_str}.pkl", "wb"))
    # Let's plot the rewards on a bar chart, show axis labels, and save the plot
    pickle.dump(all_rewards_dict, open(f"{result_path}/allTx_all_rewards_dict_{datetime_str}.pkl", "wb"))
    pickle.dump(all_path_length_dict, open(f"{result_path}/allTx_all_path_length_dict_{datetime_str}.pkl", "wb"))
    print ("========== All rewards are saved ==========")
    draw_multiple_bars_same_plot([all_log_ave_ang_loss_dict],
                                [all_path_length_dict],
                                datetime_str,
                                result_path,
                                trival_sample = True,
                                test_data = test_data)
if __name__ == "__main__":
    __main__()