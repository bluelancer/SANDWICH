import torch
device = torch.device('cpu')


import numpy as np
from stable_baselines3 import PPO

from utilsgyms.utils_gym import *
from utilsgyms.utils_gym import *
from copy import deepcopy

# for plotting
from visualization.get_scene import *
from utilsgyms.utils_winert_dataset import *

import plotly.graph_objects as go

def inference_winert(Tx_id, Rx_id, path_id = None, trained_policy_repo = None, env=None, gt_only = False, debug = False):
    from WINeRTEnv import WINeRTEnv
    predict_node_seq = []
    predict_act_seq = []
    # Prepare the environment
    assert path_id is not None, "path_id is None"
    
    if env is None:
        env = WINeRTEnv(Tx_id=Tx_id, Rx_id=Rx_id, init_path=path_id)
    else:
        env.reset_init_path(False, path_id) 
        env.reset()
    # prepare gt_act and gt_node seq for reference 
    gt_act_seq = env.gt_action
    path_length = env.max_incremental_index_for_filter_path[path_id]
    gt_node_seq = [x[1:4] for x in env.node_attr_tensor[Tx_id, Rx_id, path_id,0: path_length+1]]
    assert len(gt_node_seq) == path_length+1, "len(gt_node_seq) != path_length+1"
    if not gt_only:
        if trained_policy_repo is None:
            trained_policy_repo = "/proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/scripts/TrainedPolicy/ac_winert_Tx_{}_Rx_{}_path_{}_wpenalty.zip".format(Tx_id, Rx_id, path_id)
        if path_id is None:
            path_id = env.init_path
            print ("path_id is None, using env.init_path: ", path_id)
        # init the model
        model = PPO.load(trained_policy_repo)
        env.reset_init_path(path_id)
        obs, _ = env.reset()
        predict_node_seq.append(obs[1:4])
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs_temp, _, _, _, info = env.step(action)
            if debug:
                print ("env.distance(obs[1:4], obs_temp[1:4])", env.distance(obs[1:4], obs_temp[1:4]))
            if env.distance(obs[1:4], obs_temp[1:4]) < 0.1:
                action = torch.tensor(action).to(device)
                delta = spherical_to_ray_direction_np (action[2], action[1])
                if debug:            
                    print ("No movement detected")
                    print ("delta: ", delta)
                    print ("obs[1:4]: ", obs[1:4])
                obs_fix = deepcopy(obs)
                i = 1
                sign = 1
                while True:
                    obs_fix[1] = obs[1] - delta[0] * 0.1 * i * sign
                    obs_fix[2] = obs[2] - delta[1] * 0.1 * i * sign
                    obs_fix[3] = obs[3] - delta[2] * 0.1 * i * sign
                    env.current_step = env.current_step - 1
                    action, _states = model.predict(obs_fix, deterministic=True)
                    obs_calibrate_temp, reward, done, _, info = env.step(action)
                    if i * sign > 10: # if no movement detected after 10 steps, then change the sign
                        sign = -1
                        i = 0
                    if debug:
                        print ("env.distance(obs_calibrate_temp[1:4], obs_fix[1:4])", env.distance(obs_calibrate_temp[1:4], obs_fix[1:4]))
                        print ("0.101 * i", 0.101 * i)
                    if env.distance(obs_calibrate_temp[1:4], obs_fix[1:4]) < 0.101 * i:
                        if debug:
                            print("No movement detected, obs_temp: {} obs_calibrate_temp: {}".format(obs_temp[1:4], obs_calibrate_temp[1:4]))
                            print ("tried delta: ", delta * 0.1 * i * sign)
                        i += 1
                    else:
                        if debug:
                            print ("obs of first hop callibrated from: ", obs[1:4], " to: ", obs_fix[1:4])
                            print ('obs of next hop callibrated from: ', obs_temp[1:4], ' to: ', obs_calibrate_temp[1:4])
                        obs = obs_calibrate_temp
                        break
                    if i > 11:
                        raise ValueError("No movement detected, obs: {} obs_: {}".format(obs, obs_temp))
            else:
                obs = obs_temp
            predict_act_seq.append(action)
            predict_node_seq.append(obs[1:4])
            print("info: ", info)
            if info['done']:
                break
        return predict_act_seq, predict_node_seq, gt_act_seq, gt_node_seq
    else:
        print ("No inferece is done, only gt is returned")
        return None, None, gt_act_seq, gt_node_seq
    
def draw_node(figure, node_xyz, path_index, ith_node, color_code, debug = False, hovertext = False):
    assert figure is not None, "figure is None"
    assert len(node_xyz) == 3, "node_xyz: {}".format(node_xyz)
    if hovertext:
        hovertext = "Path: {}<br>Node: {}".format(path_index, ith_node)
        if debug:
            print ("hovertext: ", hovertext)
        figure.add_trace(go.Scatter3d(x=[node_xyz[0]],
                                y=[node_xyz[1]],
                                z=[node_xyz[2]],
                                mode='markers',
                                marker=dict(size=5, color=color_code),
                                hovertext = hovertext,
                                hoverinfo='text'))  
    else: 
        figure.add_trace(go.Scatter3d(x=[node_xyz[0]],
                        y=[node_xyz[1]],
                        z=[node_xyz[2]],
                        mode='markers',
                        marker=dict(size=5, color=color_code),
                        hoverinfo='text'))
    return figure

def draw_link(figure, node1_xyz, node2_xyz,path_index, ith_node, color_code, debug = False):
    assert figure is not None, "figure is None"
    assert len(node1_xyz) == 3, "node1_xyz: {}".format(node1_xyz)
    assert len(node2_xyz) == 3, "node2_xyz: {}".format(node2_xyz)
    hovertext = "Path: {}<br>Node: {}<br>Node: {}".format(path_index, ith_node, ith_node+1)
    figure.add_trace(go.Scatter3d(x=[node1_xyz[0], node2_xyz[0]],
                                y=[node1_xyz[1],node2_xyz[1]],
                                z=[node1_xyz[2], node2_xyz[2]],
                                mode='lines',
                                line=dict(width=2, color=color_code),
                                hovertext = hovertext,
                                hoverinfo='text'))
    return figure

def render_winert(path_dict, fig, gt_only = False, selected_path = None, debug = False):
    if gt_only:
        print ("drawing gt only")
    if selected_path is not None:
        print ("drawing selected path: ", selected_path)
        iterator = [selected_path]
    else: 
        iterator = list(path_dict.keys())
        assert len(iterator) > 0, "path_dict is empty"
        
    for path_id in iterator:
        print ("path_id: ", path_id)
        if path_dict[path_id] is not None:
            gt_node_seq = path_dict[path_id]['gt_node_seq']
            gt_act_seq = path_dict[path_id]['gt_act_seq']
            tx_coord = path_dict[path_id]['tx_coord']
            rx_coord = path_dict[path_id]['rx_coord']
            path_length = path_dict[path_id]['path_length']
            # TODO: should be path_length  and path_length +1, but we feed the 1st hop by appending 
            # if 1st hop is not given, then we should use path_length  and path_length +1

            predict_node_seq = path_dict[path_id]['predict_node_seq']
            predict_act_seq = path_dict[path_id]['predict_act_seq']
        
            if not gt_only:
                for i in range(len(predict_node_seq) - 1):
                    fig = draw_node(fig, predict_node_seq[i], path_id, i, 'purple')
                    fig = draw_link(fig, predict_node_seq[i], predict_node_seq[i+1], path_id, i, 'purple')
                if len(predict_node_seq) == len(gt_node_seq):
                    pass
                else:
                    print ("detected a shorter path")
                    gt_node_seq = [predict_node_seq[0]] + gt_node_seq
                    # assert len(predict_node_seq) == len(gt_node_seq), "path_id: {} len(predict_node_seq): {} len(gt_node_seq): {}".format(path_id, len(predict_node_seq), len(gt_node_seq))
            for i in range(len(gt_node_seq) - 1):
                fig = draw_node(fig, gt_node_seq[i], path_id, i, 'green')
                fig = draw_link(fig, gt_node_seq[i], gt_node_seq[i+1], path_id, i, 'green')
            fig = draw_node(fig, gt_node_seq[len(gt_node_seq) - 1], path_id, len(gt_node_seq) - 1, 'blue')
            fig = draw_node(fig, tx_coord, path_id, "Tx", 'blue')
            fig = draw_node(fig, rx_coord, path_id, "Rx", 'red')
        else:
            continue
        # Adjusting layout if needed
        if selected_path is not None:
            fig.update_layout(scene=dict(
                                xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z'),
                            title="Tx to Rx: {} to {}".format(tx_coord, rx_coord))
        else:
            fig.update_layout(scene=dict(
                                xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z'))
    return fig

def draw_path_tidy(Tx_id, Rx_id, path = None, gt_mode = True, debug = False, run_on_local = False, tidy = True, compute_l1_loss = False, A2C_model = True):
    dir_path1 = "/proj/raygnn/workspace/raytracingdata/testenv/objs"
    _, _, _, _, fig= get_scene_plotly_lay_objs(dir_path1, show=False, return_3D_list=False)
    winert_env = WINeRTEnv(debug= run_on_local,Tx_id = Tx_id, Rx_id = Rx_id)
    env_path_init_length = winert_env.max_incremental_index_for_filter_path[winert_env.init_path]
    if env_path_init_length >1 :
        tx_coord =  winert_env.node_attr_tensor[winert_env.Tx_id, winert_env.Rx_id,winert_env.init_path,0, 1:4].astype(float)
        rx_coord = winert_env.node_attr_tensor[winert_env.Tx_id, winert_env.Rx_id, winert_env.init_path, env_path_init_length, 1:4].astype(float)
        if debug:
            print ("rx_coord: ", rx_coord)
            print ("winert_env.init_path: ", winert_env.init_path)
            print ("winert_env.node_attr_tensor[winert_env.Tx_id, winert_env.Rx_id, winert_env.init_path, path_init_length, 1:4].astype(float)",  winert_env.node_attr_tensor[winert_env.Tx_id, winert_env.Rx_id, winert_env.init_path, :, 1:4].astype(float))
            print ("env_path_init_length: ", env_path_init_length)
    else:
        tx_coord = None
        rx_coord = None
    obs,_ = winert_env.reset()
    if path is not None:
        iterator = [path]
    else: 
        iterator = range(30)
    
    path_dict={}
    for path_id in iterator:
        path_length = winert_env.max_incremental_index_for_filter_path[path_id]
        if path_length <2:
            path_dict[path_id] = None
            continue
        elif path_id is not None:
            if A2C_model:
                predict_act_seq, predict_node_seq, gt_act_seq, gt_node_seq = inference_winert(Tx_id, Rx_id, path_id = path_id, trained_policy_repo = None, env=winert_env, debug = debug, gt_only = gt_mode)
            else:
                raise ValueError("Not implemented yet")
            if tidy:
                predict_node_seq = predict_node_seq[0:path_length]
                predict_act_seq = predict_act_seq[0:path_length-1]
            if compute_l1_loss:
                assert len(predict_act_seq[0:path_length-1]) == len(gt_act_seq[:path_length]) - 1 , "len(predict_act_seq) != len(gt_act_seq) , path_id: {} len(predict_act_seq): {} len(gt_act_seq): {}".format(path_id, len(predict_act_seq), len(gt_act_seq))
                pred = np.array(predict_act_seq)
                gt = np.array(gt_act_seq)[1:path_length]
                l1_loss = np.sum(np.abs(pred[:,1:] - gt[:,1:]))
            else:
                l1_loss = None
            path_dict[path_id] = {"predict_act_seq": predict_act_seq,"predict_node_seq": predict_node_seq,"gt_act_seq":gt_act_seq, "gt_node_seq":gt_node_seq , "tx_coord": tx_coord, "rx_coord": rx_coord, "path_length": path_length, "l1_loss": l1_loss}
            assert np.all(gt_node_seq[0] == tx_coord), "path_id: {} path_length: {} node_seq[0]: {} tx_coord: {}".format(path_id, path_length, gt_node_seq[0], tx_coord)
            assert np.all(gt_node_seq[-1] == rx_coord), "path_id: {} path_length: {} node_seq[-1]: {} rx_coord: {}".format(path_id, path_length, gt_node_seq[-1], rx_coord)
     
    fig = render_winert(path_dict, fig, gt_only = gt_mode, selected_path = None, debug = False)
    # TODO: add a loss function to compare the predicted path with the ground truth path
    return fig, path_dict
            
def draw_path_single(act_seq, node_seq, debug = False):
    dir_path1 = "/proj/raygnn/workspace/raytracingdata/testenv/objs"
    _, _, _, _, fig= get_scene_plotly_lay_objs(dir_path1, show=False, return_3D_list=False)
    if type(node_seq) == list:
        raise NotImplementedError("Not implemented yet") # this is for the case of node_seq is a list of nodes
    # above should be same as draw_path_tidy
    # draw a path according to a series of actions or a series of nodes
    tx_coord =  node_seq[0]
    rx_coord = node_seq[-1]
    assert tx_coord.shape[0] == 3, "tx_coord: {}".format(tx_coord)
    assert rx_coord.shape[0] == 3, "rx_coord: {}".format(rx_coord)
    path_length = act_seq.shape[0]
    path_dict={}
    
    path_dict[0] = {"predict_act_seq":np.ones_like(act_seq) ,"predict_node_seq":np.ones_like(node_seq) ,
                          "gt_act_seq":act_seq, "gt_node_seq":node_seq,
                          "tx_coord": tx_coord, "rx_coord": rx_coord,
                          "path_length": path_length, "l1_loss": 0}
    assert len(path_dict) == 1, "len(path_dict) != 1"
 
    fig = render_winert(path_dict, fig, gt_only = True, selected_path = 0, debug = False)
    # TODO: add a loss function to compare the predicted path with the ground truth path
    return fig, path_dict
            
def draw_path_single_n_simple(node_seqs,env_id, path_len_lim = None, Tx_coord = None, debug = False):
    dir_path1 = "/proj/gaia/RayDT/dataset/raw_data/wi3rooms/objs"
    _, _, _, _, fig= get_scene_plotly_lay_objs(dir_path1, env_id, show=False, return_3D_list=False)
    node_seq1, node_seq2 = node_seqs
    if path_len_lim is not None:
        node_seq1 = node_seq1[:path_len_lim]
        node_seq2 = node_seq2[:path_len_lim]
        print ("path_len_lim: ", path_len_lim)
    if Tx_coord is not None:
        node_seq1 = np.vstack([Tx_coord, node_seq1])
        node_seq2 = np.vstack([Tx_coord, node_seq2])
        print("appending Tx_coord to the node_seq")
    if type(node_seq1) == list:
        raise NotImplementedError("Not implemented yet") # this is for the case of node_seq is a list of nodes
    assert len(node_seq1) == len(node_seq2), "len(node_seq1) != len(node_seq2)"
    for i in range(len(node_seq1) - 1):
        fig = draw_node(fig, node_seq1[i], 0, i, 'purple')
        fig = draw_link(fig, node_seq1[i], node_seq1[i+1], 0, i, 'purple')
        fig = draw_node(fig, node_seq2[i], 0, i, 'green')
        fig = draw_link(fig, node_seq2[i], node_seq2[i+1], 0, i, 'green')
    fig.update_layout(scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z'))
    return fig, None
        
        