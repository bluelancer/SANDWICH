import torch
import os
import pyvista as pv  
import numpy as np
from tqdm import trange
from utilsgyms.utils_gym import *
import re

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.spaces import Box

# use trajecotry to get the reward
from typing import Dict, Optional
from typing import Any

from visualization.get_scene import *
from visualization.vis_winert import *
from utilsgyms.utils_winert_dataset import *

RADIAN_RANGE = (-np.pi/2, np.pi/2) 
AZIMUTH_RANGE = (-np.pi, np.pi)

class WINeRTEnv(gym.Env):
    env_mesh = None
    node_attr_tensor = None
    edge_attr_tensor = None
    gt_tensor = None
    
    def __init__(self,
                 storage_dir = "/proj/raygnn_storage/HFdata/raw_data/train/node_attr_tensor",
                 surface_index_path = "/proj/raygnn_storage/HFdata/raw_data/train/surface_index", #'/proj/raygnn_storage/HFdata/raw_data/train/objs',
                 env_id = None, 
                 env_index = None,
                 debug = False,
                 Tx_id= 0,
                 Rx_id = 0,
                 state_w_ray = True, 
                 set_init_path_flag = True, 
                 init_path = 1,
                 dummy_input =False,  
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 test_pp = False):
        
        assert (env_id is not None or env_index is not None), f"env_id {env_id}, env_index {env_index}"
        if env_index is not None:
            # find the env_id from the env_index from the storage_dir
            env_id = self.find_env_id_or_index(storage_dir, env_index, find_id_or_index_opt = 'env_id')
            assert env_id is not None, f"env_id {env_id}"
        elif env_id is not None:
            env_index = self.find_env_id_or_index(storage_dir, env_id, find_id_or_index_opt = 'env_index')
            assert env_index is not None, f"env_index {env_index}"
        self.env_id = env_id
        self.env_index = env_index
        
        self.debug = debug
        if self.debug:
            self.device = torch.device('cpu')
        else:
            self.device = device

        # Training parameters
        self.Tx_id = Tx_id
        self.Rx_id = Rx_id
        self.set_init_path_flag = set_init_path_flag
        self.state_w_ray = state_w_ray
        
        # Env parameters, Tx, Rx indepent
        if WINeRTEnv.node_attr_tensor is None:
            if test_pp:
                print ("Test Preprocessing")
                storage_dir = storage_dir.replace("train", "test")
                surface_index_path = surface_index_path.replace("train", "test")
                print (f"storage_dir {storage_dir}, surface_index_path {surface_index_path}")
            
            gt_storage_dir = storage_dir.replace("node_attr_tensor", "gt")           #"node_attr_tensor"->"channel_gts/"
            WINeRTEnv.gt_dir  = os.path.join(gt_storage_dir, f'gt_{env_index}.pt')
            WINeRTEnv.node_attr_tensor_path = os.path.join(storage_dir, f'node_attr_tensor_{env_id}_{env_index}.pt')
            # print ("WINeRTEnv.node_attr_tensor_path: ", WINeRTEnv.node_attr_tensor_path)
            if device == torch.device('cpu'):
                WINeRTEnv.node_attr_tensor = torch.load(WINeRTEnv.node_attr_tensor_path, map_location=torch.device('cpu')).to(self.device)
                WINeRTEnv.gt_tensor = torch.load(WINeRTEnv.gt_dir, map_location=torch.device('cpu')).to(self.device)
            else: 
                WINeRTEnv.node_attr_tensor = torch.load(os.path.join(storage_dir, f'node_attr_tensor_{env_id}_{env_index}.pt')).to(self.device)
                WINeRTEnv.gt_tensor = torch.load(WINeRTEnv.gt_dir).to(self.device)
            WINeRTEnv.edge_attr_tensor = self.prepare_edge_attr_tensor().to(self.device)
            
            # path filtering parameters
            WINeRTEnv.all_incremental_index, WINeRTEnv.all_incremental_index_for_filter_path, WINeRTEnv.reduced_mask = self.get_incremental_index(WINeRTEnv.node_attr_tensor)
            # all_max_incremental_index is an index filter of node num
            WINeRTEnv.all_max_incremental_index = torch.max(WINeRTEnv.all_incremental_index, dim=-1).values.to(self.device).squeeze() # This will be useful to compute the set loss
            # all_max_incremental_index_for_filter_path is an index filter of edge number, which is all_max_incremental_index -1 
            WINeRTEnv.all_max_incremental_index_for_filter_path = torch.max(WINeRTEnv.all_incremental_index_for_filter_path, dim=-1).values.to(self.device).squeeze()
            assert WINeRTEnv.all_max_incremental_index_for_filter_path.shape[-1] == 30, f"WINeRTEnv.all_max_incremental_index.shape {WINeRTEnv.all_max_incremental_index.shape}"

            WINeRTEnv.surface_index_path = surface_index_path  + "/"+ str(env_id)+"_surface_index.pt"
            # Generating train split: 5367 examples all_path_count = 7000, unreachable_path_count = 39, unreachable_path_ratio = 0.005571428571428572
            # WINeRTEnv.surface_index_path = os.path.join(surface_index_path, f"{env_index}_surface_index.pt") # try this # no this is wrong
            # ValueError: all_path_count = 375, unreachable_path_count = 101, unreachable_path_ratio = 0.2693333333333333
            WINeRTEnv.faces_rect_path = WINeRTEnv.surface_index_path.replace("surface_index.pt", "faces_rect_path.pt")
            WINeRTEnv.vertices_path = WINeRTEnv.surface_index_path.replace("surface_index.pt", "vertices_path.pt")

            # load the environment for rendering mesh
            WINeRTEnv.surface_index = torch.load(WINeRTEnv.surface_index_path).to(device)
            WINeRTEnv.face_rect = torch.load(WINeRTEnv.faces_rect_path).to(device)
            WINeRTEnv.vertices = torch.load(WINeRTEnv.vertices_path).to(device)
        
        # 7 features {intereaction types [0], x[1],y[2],z[3] coordinates, texture[4], validaty[5] (always 1), surface index[6]}
        if self.set_init_path_flag:
            self.init_value = WINeRTEnv.node_attr_tensor[self.Tx_id,self.Rx_id, 0 ,1,:4]
            # Given Tx_id all 1st coodinates are same, we take the first one
        else:
            self.init_value = WINeRTEnv.node_attr_tensor[self.Tx_id,self.Rx_id, 0 ,0,:4]
            # Given Tx_id all 1st coodinates are same, we take the first one
        
        # get incremental index, each denote temination depth of the path
        self.incremental_index, self.incremental_index_for_filter_path = WINeRTEnv.all_incremental_index[self.Tx_id,self.Rx_id,:,:], WINeRTEnv.all_incremental_index_for_filter_path[self.Tx_id,self.Rx_id,:,:]
        self.max_incremental_index = torch.max(self.incremental_index, dim=-1).values.to(self.device).squeeze() # This will be useful to compute the set loss
        self.max_incremental_index_for_filter_path = torch.max(self.incremental_index_for_filter_path, dim=-1).values.to(self.device).squeeze()
        assert  self.max_incremental_index.shape[-1] == 30, f"self.max_incremental_index.shape {self.max_incremental_index.shape}"

        self.max_step = 5 # maximum number of steps, always 5, if we start from Tx, current_step = 0 else current_step = 1
        if self.set_init_path_flag:
            self.current_step = 1
        else:
            self.current_step = 0
        # 2 construct the environment, including the surface_index, face_rect, vertices
        if WINeRTEnv.env_mesh is None:
             WINeRTEnv.env_mesh = self.render_env(WINeRTEnv.face_rect, WINeRTEnv.surface_index, WINeRTEnv.vertices)
             WINeRTEnv.env_mesh.compute_normals(inplace=True) 
        # RL Gym state space
        self.action_traj = np.zeros(torch.Size([self.max_step, 3]))
        if state_w_ray:
            self.state_traj = np.zeros(torch.Size([self.max_step + 1, 6]))
        else:
            self.state_traj = np.zeros(torch.Size([self.max_step + 1, 4]))
    
        if self.set_init_path_flag:
            self.set_init_path(init_path)
        self.state = winert_initalize_state_np(self, init_angle=True, state_w_ray = self.state_w_ray)
        # RL Gym action space
        self.action_space = gym.spaces.Box(
            low=np.array([0, RADIAN_RANGE[0], AZIMUTH_RANGE[0]]), 
            high=np.array([6, RADIAN_RANGE[1], AZIMUTH_RANGE[1]]),
            dtype=np.float32
        )
        
        # add receipt_loss:
        self.tx_coord = WINeRTEnv.node_attr_tensor[self.Tx_id, self.Rx_id, self.init_path, 0, 1:4]
        self.rx_coord = WINeRTEnv.node_attr_tensor[self.Tx_id, self.Rx_id, self.init_path, self.max_incremental_index[self.init_path], 1:4]
        self.sphere_rx = pv.Sphere(center=self.rx_coord, radius=0.1).triangulate()
        self.receipt_los = -10

        if self.set_init_path_flag:
            self.state_traj[0, :4] = WINeRTEnv.node_attr_tensor[self.Tx_id,self.Rx_id, 0 ,1,:4] # maybe not init value
        # action starts from 0 while state starts from 1
        self.action_traj[self.current_step-1, 1:] = WINeRTEnv.edge_attr_tensor[self.Tx_id,self.Rx_id,self.init_path,0,1:3]
        self.action_traj[self.current_step-1, 0] = abs(WINeRTEnv.node_attr_tensor[self.Tx_id,self.Rx_id,self.init_path,1,0]) # correct the type -1 into 1
        self.state_traj[self.current_step, :] = self.state 

        self.gt_action = np.concatenate([self.gt_action_type, self.gt_action_angle], axis=-1)

        assert self.gt_action.shape [-2] == 5, f"self.gt_action.shape {self.gt_action.shape}"
        assert self.gt_action.shape [-1] == 3, f"self.gt_action.shape {self.gt_action.shape}"
        
        # fill the first row with the initial state
        # Define observation_space based on the nature of your observations
        # For example, if observations are vectors of a fixed size:
        # TODO: change the observation space to a fixed size
        if state_w_ray:
            self.observation_space = gym.spaces.Box(
                low=np.array([0, -0.1, -0.1, -0.1, RADIAN_RANGE[0], AZIMUTH_RANGE[0]]),
                high=np.array([6, 10.1, 5.1, 3.1, RADIAN_RANGE[1], AZIMUTH_RANGE[1]]), # 1st dim is the type, 2nd, 3rd, 4th dim is the x,y,z coord
                shape=([6]), # 4 features per-hop (type, x,y,z) 
                dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array([0, -0.1, -0.1, -0.1]),
                high=np.array([6, 10.1, 5.1, 3.1]), # 1st dim is the type, 2nd, 3rd, 4th dim is the x,y,z coord
                shape=([4]), # 4 features per-hop (type, x,y,z) 
                dtype=np.float32
            )
        self.dummy_input = dummy_input
        if self.dummy_input:
           self.state = np.array([0, 7, 2.5, 1, 0, 0], dtype=np.float32)
    
    def set_init_path(self, init_path, reset_flag = False):
        self.init_path = init_path
    
        self.action_traj[self.current_step-1, 1:] = WINeRTEnv.edge_attr_tensor[self.Tx_id,self.Rx_id,self.init_path,0,1:3]
        self.action_traj[self.current_step-1, 0] =abs(WINeRTEnv.node_attr_tensor[self.Tx_id,self.Rx_id,self.init_path,1,0])
        
        self.gt_action_angle = WINeRTEnv.edge_attr_tensor[self.Tx_id,self.Rx_id,self.init_path,:,1:3]
        self.gt_action_type = abs(np.expand_dims(WINeRTEnv.node_attr_tensor[self.Tx_id,self.Rx_id,self.init_path,1:,0], axis = -1))
        if len(self.gt_action_type.shape) != len(self.gt_action_angle.shape):
            self.gt_action_type = self.gt_action_type.reshape(-1,1)
        self.gt_action = np.concatenate([self.gt_action_type, self.gt_action_angle], axis=-1)
        
        assert self.gt_action.shape [-2] == 5, f"self.gt_action.shape {self.gt_action.shape}"
        assert self.gt_action.shape [-1] == 3, f"self.gt_action.shape {self.gt_action.shape}"
        return
        
    def get_incremental_index(self, node_attr_tensor):
        if type(node_attr_tensor) == np.ndarray:
            node_attr_tensor = torch.tensor(node_attr_tensor)
            print ("node_attr_tensor is converted to torch.tensor, it was np.ndarray")
        # Step 1: Compute the avaliable_node_order_mask for the entire tensor
        # Exclude padding and zeros: Combine conditions for [-1.0, -1.0, -1.0] and [0.0, 0.0, 0.0]
        condition_no_padding = torch.logical_not(torch.all(node_attr_tensor[:,:,:,:,1:4] == torch.tensor([-1.0, -1.0, -1.0], device=self.device), dim=-1))
        condition_no_zeros = torch.logical_not(torch.all(node_attr_tensor[:,:,:,:,1:4] == torch.tensor([0.0, 0.0, 0.0], device=self.device), dim=-1))

        # Apply both conditions
        available_node_order_mask_for_filter_path = torch.logical_and(condition_no_padding, condition_no_zeros)
        avaliable_node_order_mask = torch.not_equal(node_attr_tensor[:,:,:,:,1:4], torch.tensor([-1.0, -1.0, -1.0], device=self.device)).to(self.device)
        # Step 2: Reduce the mask to a single True/False value per row
        reduced_mask = torch.all(avaliable_node_order_mask, dim=-1)
        # reduced_mask = available_node_order_mask
        # Step 3: Compute cumulative indices
        cumulative_indices = torch.cumsum(reduced_mask, dim=-1) - 1
        cumulative_indices_for_filter_path = torch.cumsum(available_node_order_mask_for_filter_path, dim=-1) - 1
        # Step 4: Replace False values with -1
        incremental_index = torch.where(reduced_mask, cumulative_indices, torch.tensor(-1, dtype=torch.int64))
        incremental_index_for_filter_path = torch.where(available_node_order_mask_for_filter_path, cumulative_indices_for_filter_path, torch.tensor(-1, dtype=torch.int64))
        # return incremental_index[self.Tx_id,self.Rx_id,:,:], incremental_index_for_filter_path[self.Tx_id,self.Rx_id,:,:]
        return incremental_index, incremental_index_for_filter_path, reduced_mask
    
    def get_reward(self, trajectory, angular_loss_type='radius', type_loss = "onehot", debug=False):
        # input initial position x, and a set of action trajectory, output the reward, and the final position
        trajectory = torch.tensor(trajectory).to(self.device)
        assert trajectory.shape == torch.Size([5, 3]), f"trajectory.shape {trajectory.shape}"
        # Step 1: Compute the max_path_length for the entire tensor
        max_path_length = self.max_incremental_index_for_filter_path.unsqueeze(-1) #-1
        # Step 2: Compute the loss between the persudo_gt and the edge_attr_tensor, only within the max_path_length
        # Step 2.1: Compute the mask for the entire tensor
        node_mask = torch.logical_and(self.incremental_index <= max_path_length + 1, self.incremental_index != -1)

        mask = node_mask[...,:,1:].to(self.device)
        trajectory = trajectory.unsqueeze(0).to(self.device)
        # Step 2.2: Compute the angular loss
        edge_attr_tensor = torch.tensor(WINeRTEnv.edge_attr_tensor).clone()

        if angular_loss_type == 'set_loss':
            # each trajectory is a set, we compute the set loss
            # compare one trajectory with all the edge_attr_tensor
            # Calculate the L1 loss for each path_id against all path_id_edge
            assert WINeRTEnv.edge_attr_tensor.shape[-2:] == torch.Size([5, 3]), f"WINeRTEnv.edge_attr_tensor.shape {WINeRTEnv.edge_attr_tensor.shape}"
            assert trajectory.shape[-2:] == torch.Size([WINeRTEnv.edge_attr_tensor.shape[3], 3])
            # a single prediction comparing with 30 paths
            prediction = trajectory[...,:,1:3].to(self.device)
            gt = edge_attr_tensor[self.Tx_id,self.Rx_id,:,:,1:3].to(self.device)
            if self.set_init_path_flag:
                assert torch.all(prediction[...,0,:] == gt[...,self.init_path,0,:]), f"prediction[...,0,:] {prediction[...,0,:]}, gt[...,self.init_path,0,:] {gt[...,self.init_path,0,:]}"
            angular_loss_l1 = torch.abs(gt.unsqueeze(0) - prediction.unsqueeze(1))
            assert angular_loss_l1.shape[-3:] == torch.Size([WINeRTEnv.edge_attr_tensor.shape[2], WINeRTEnv.edge_attr_tensor.shape[3], 2]), f"angular_loss_l1.shape {angular_loss_l1.shape}"            
            assert mask.shape[-2:] == torch.Size([WINeRTEnv.edge_attr_tensor.shape[2], WINeRTEnv.edge_attr_tensor.shape[3]]), f"mask.shape {mask.shape}"
            
            unsqueezed_mask = mask.unsqueeze(-3)
            
            assert unsqueezed_mask.shape[-2:] == torch.Size([WINeRTEnv.edge_attr_tensor.shape[2],  WINeRTEnv.edge_attr_tensor.shape[3]])
            extended_unsqueezed_mask = unsqueezed_mask.unsqueeze(-1)  # Shape becomes [batch_size, Rx, 1, 30, some_other_dimension, 2]
            assert extended_unsqueezed_mask.shape[-3:] == torch.Size([WINeRTEnv.edge_attr_tensor.shape[2],  WINeRTEnv.edge_attr_tensor.shape[3], 1])
            ready_mask,_ = torch.broadcast_tensors(extended_unsqueezed_mask, angular_loss_l1)

            masked_angular_loss  = angular_loss_l1 * ready_mask
            # Sum over the azimuth and radian dimension
            summed_angular_azi_rad_loss = torch.sum(masked_angular_loss, dim=-1)
            inverse_ordered_power = torch.arange(summed_angular_azi_rad_loss.shape[-1], 0, -1).to(self.device)
            # TODO:DO ABLATION STUDY
            # Tried: using inverse exp power to strengthen the loss of the first few hops in each path
            # inverse_ordered_power_exp = torch.exp(inverse_ordered_power)
            # Tried: or even using the square of the inverse exp power
            # inverse_ordered_power_square = inverse_ordered_power ** 2
            
            expanded_inverse_power = inverse_ordered_power.repeat(30, 1)
            range_tensor = torch.arange(1, inverse_ordered_power.size(0) + 1).unsqueeze(0).repeat(max_path_length.size(0), 1)
            # Now, create a mask where each element is True if it is less than or equal to the corresponding max_path_length
            weight_mask = (range_tensor <= max_path_length).float()
            # Step 2: Calculate the sum of the first n elements for reweighting
            # Use the mask to zero out invalid elements and sum along the last dimension
            sums_for_reweighting = (inverse_ordered_power * weight_mask[:,:5]).sum(dim=1, keepdim=True)   
            # Step 3: Reweight the tensor
            # Broadcast the sums for division and multiply by the valid_mask to zero out elements beyond max_path_length
            reweighted_full_op = (expanded_inverse_power / sums_for_reweighting) * weight_mask[:,:5]

            order_powered_summed_angular_azi_rad_loss = summed_angular_azi_rad_loss * reweighted_full_op
            assert summed_angular_azi_rad_loss.shape[-3:] == torch.Size([1, WINeRTEnv.edge_attr_tensor.shape[2], WINeRTEnv.edge_attr_tensor.shape[3]])
            # summed_path_loss = torch.sum(summed_angular_azi_rad_loss, dim=-1)
            summed_path_loss = torch.sum(order_powered_summed_angular_azi_rad_loss, dim=-1)
            # length_ave_summed_path_loss = torch.div(summed_path_loss, max_path_length.squeeze())
            assert summed_path_loss.shape[-2:] == torch.Size([1, WINeRTEnv.edge_attr_tensor.shape[2]])
            # mean_angular_loss, min_path_id = torch.min(summed_path_loss, dim=1)
            
            summed_path_loss[..., 0, self.max_incremental_index_for_filter_path < 0] = 1e6
            mean_angular_loss, min_path_id = torch.min(summed_path_loss, dim=-1)
            assert mean_angular_loss.shape[-1] == 1, f"mean_angular_loss.shape {mean_angular_loss.shape}"

        else:
            raise NotImplementedError("angular_loss_type should be 'set_loss'")
        # Step 2.3: Compute the type loss
        gt_type = abs(edge_attr_tensor[self.Tx_id,self.Rx_id,:,:,0]).to(self.device) # get the type from the node_attr_tensor, omit the initial position, change -1 to 1 for one hot

        est_type = trajectory [:,:,0].to(self.device)
        assert gt_type.shape[-1] == est_type.shape[-1], f"gt_type.shape {gt_type.shape}, est_type.shape {est_type.shape}"
        # get one hot encoding
        if type_loss == "onehot":
            gt_type_one_hot = torch.nn.functional.one_hot(gt_type.to(torch.int64), num_classes=6).to(torch.float32).to(self.device)
            est_type_one_hot = torch.nn.functional.one_hot(est_type.to(torch.int64), num_classes=6).to(torch.float32).to(self.device)
            type_loss_each = torch.nn.functional.mse_loss(gt_type_one_hot, est_type_one_hot, reduction='none')
            type_loss = torch.sum(type_loss_each, dim=-1)
        elif type_loss == "TF":
            if angular_loss_type == 'set_loss':
                min_path_id_for_mask = min_path_id.unsqueeze(-1).expand(min_path_id.shape+torch.Size([5]))
                selected_gt_type = torch.take(gt_type, min_path_id_for_mask)
                type_loss = torch.where(selected_gt_type == est_type, torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device))
                selected_mask = torch.take(mask, min_path_id_for_mask)
                masked_type_loss = type_loss * selected_mask # only consider the min_path_id
                type_loss = torch.sum(masked_type_loss, dim=-1)
                assert type_loss.shape[-1] == trajectory.shape[0], f"type_loss.shape {type_loss.shape}"
                mean_type_loss = torch.sum(type_loss) / torch.sum(selected_mask)
            else:
                type_loss = torch.where(gt_type == est_type, torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device))
                mean_type_loss = torch.sum(type_loss * mask) / torch.sum(mask)

        else:
            raise ValueError("type_loss should be either 'onehot' or 'TF'")

        # Step 2.4: Compute the mean loss
        # mean_loss = mean_angular_loss + mean_type_loss + 1e-6 # add a small number to avoid 0 loss
        # This offers an alway negative reward, iif the trajectory is valid, we will get 0 reward
        if debug:
            print (f"shape of trajectory {trajectory.shape}, shape of edge_attr_tensor {WINeRTEnv.edge_attr_tensor.shape}")
            print (f"shape of max_path_length {max_path_length.shape}, node_mask {node_mask.shape}, mask {self.mask.shape}")
            print (f"mean_angular_loss {mean_angular_loss}, mean_type_loss {mean_type_loss}") 

        reward = 1/(mean_angular_loss.to(self.device)+ 1e-6) # + 1/(mean_type_loss.to(self.device)+ 1e-6)
        return reward, mean_angular_loss, mean_type_loss
    
    def get_log_reward(self, trajectory,  angular_loss_type='mse', type_loss = 'TF', debug=False):
        reward, mean_angular_loss, mean_type_loss = self.get_reward(trajectory, angular_loss_type=angular_loss_type, type_loss=type_loss, debug = debug)
        log_reward = torch.log(reward)
        return log_reward, mean_angular_loss, mean_type_loss

    def get_ray_target(self, ray_directions_azimuth, ray_directions_radian, ray_origins, return_rays=False, debug = False, use_perturb=True):
        if WINeRTEnv.env_mesh is None:
            raise ValueError("No environment mesh was provided")
        # Get the ray target
        ray_directions = spherical_to_ray_direction_np(ray_directions_azimuth , ray_directions_radian )
        ray_origins_clamped = torch.clamp(ray_origins - torch.tensor([1e-6,1e-6,1e-6]), min=0)
        if use_perturb:
            points, rays, _ = WINeRTEnv.env_mesh.multi_ray_trace(ray_origins_clamped,
                                                                 ray_directions,
                                                                 first_point=False)
            
            points_adjusted = torch.tensor(points) + torch.tensor([1e-6,1e-6,1e-6])
            points_adjusted_de_src_index = abs(torch.norm(torch.Tensor(points_adjusted) - ray_origins, dim=-1))>= 1e-4
            points_adjusted_true = points_adjusted[torch.roll(torch.logical_not(points_adjusted_de_src_index), -1, dims=-1)] 
        else:            
            points, rays, _ = WINeRTEnv.env_mesh.multi_ray_trace(ray_origins,
                                                    ray_directions,
                                                    first_point=True)
            points_adjusted_true = torch.tensor(points)

        if debug:
            assert points_adjusted_true is not None, "points is None"
            pass
        if return_rays:
            return points_adjusted_true, rays
        else:
            return points_adjusted_true
        
    # helper functions
    def distance(self, point_1, point_2):
        point_1_tensor = torch.tensor(point_1).clone()
        point_2_tensor = torch.tensor(point_2).clone()
        return torch.sqrt(torch.sum((point_1_tensor - point_2_tensor)**2))

    def radian(self, point_1, point_2):
        delta = point_1 - point_2
        point_1_shape = point_1.shape
        point_2_shape = point_2.shape
        assert point_1.shape == point_2.shape, f"point_1_shape {point_1_shape}, point_2_shape {point_2_shape}"
        # parameterize delta to get last dim
        if len(point_1_shape) > 1:
            delta = delta.reshape(-1, point_1_shape[-1])
            xy_dist = torch.sqrt(delta[:,0]**2 + delta[:,1]**2)
            radian_2dim = torch.atan2(delta[:,2], xy_dist)
            radian = radian_2dim.reshape(point_1_shape[:-1])
        else:
            xy_dist = torch.sqrt(delta[0]**2 + delta[1]**2)
            radian = torch.atan2(delta[2], xy_dist)
        return radian

    def azimuth(self, point_1, point_2):
        assert point_1.shape == point_2.shape, f"point_1.shape {point_1.shape}, point_2.shape {point_2.shape}"
        point_1_shape = point_1.shape
        # parameterize delta to get last dim
        if len(point_1_shape) > 1:
            delta = point_1 - point_2
            delta = delta.reshape(-1, point_1_shape[-1])
            azimuth_2dim = torch.atan2(delta[:,1], delta[:,0])
            azimuth = azimuth_2dim.reshape(point_1_shape[:-1])
        else:  
            azimuth = torch.atan2(point_1[1] - point_2[1], point_1[0] - point_2[0])
        return azimuth
    
    def prepare_edge_attr_tensor(self):
        # edge_attr_tensor.shape torch.Size([10, 1800, 30, 5, 4])
        # 10 Tx, 1800 Rx, 30 paths, 5 edges connecting 6 hops,
        # 3 features dist [0], radian[1], azimuth[2]
        tx_num = WINeRTEnv.node_attr_tensor.shape[0]
        rx_num = WINeRTEnv.node_attr_tensor.shape[1]
        path_num = WINeRTEnv.node_attr_tensor.shape[2]
        edge_attr_tensor = torch.zeros(tx_num, rx_num, path_num, 5, 3)
        assert WINeRTEnv.node_attr_tensor is not None, "node_attr_tensor is None"
        node_attr_coord = WINeRTEnv.node_attr_tensor[:,:,:,:,1:4]
        node_pair = torch.cat([node_attr_coord,
                            torch.roll(node_attr_coord,
                                        1, dims=-2)], dim=-1)[:,:,:,1:,:]  
        edge_attr_tensor[:,:,:,:,0] = torch.norm(node_pair[:,:,:,:,0:3] - node_pair[:,:,:,:,3:6], dim=4)
        edge_attr_tensor[:,:,:,:,1] = self.radian(node_pair[:,:,:,:,3:6], node_pair[:,:,:,:,0:3])
        edge_attr_tensor[:,:,:,:,2] = self.azimuth(node_pair[:,:,:,:,3:6], node_pair[:,:,:,:,0:3])
        return edge_attr_tensor
    
    def get_rt_result_single_process_np(self, x, ray_directions, debug = False):
        ray_origins_clamped = torch.clamp(x - torch.tensor([1e-6,1e-6,1e-6]), min=0)
        multi_points, multi_rays, _ = WINeRTEnv.env_mesh.multi_ray_trace([ray_origins_clamped],
                                                                [ray_directions],
                                                                first_point=False,
                                                                retry = False
                                                                )
        if len(multi_rays) ==  0:
            multi_points, multi_rays, _ = WINeRTEnv.env_mesh.multi_ray_trace([ray_origins_clamped],
                                                                [ray_directions],
                                                                first_point=False,
                                                                retry = True
                                                                )
        if len(multi_rays) ==  0:
            multi_points = np.array([ray_origins_clamped])
            multi_rays = np.array([0])
            print (f"multi_rays is empty, x {x}, ray_directions {ray_directions}")
        cleaned_points_stack, cleaned_rays = clean_rt_result_np(multi_points, multi_rays)
        return cleaned_points_stack, cleaned_rays
    def reset_init_path(self, after_stepping, init_path = None):
        if self.set_init_path_flag:
            assert init_path is not None, f"init_path {init_path}"
            if after_stepping: # if some steps have been taken, reset the init_path
                self.set_init_path(init_path, reset_flag = True)
            else: # if no step has been taken, reset the init_path, on condition that the environment has just been initialized
                self.set_init_path(init_path, reset_flag = False)
            assert self.init_path == init_path, f"self.init_path {self.init_path}, init_path {init_path}"
        else:
            raise ValueError("set_init_path_flag is False, cannot reset the init_path")
        return
    # Gym functions, reset, step, This is not verified yet, just copied from the gym tutorial
    def reset(self, seed: int = None, options: Optional[Dict[str, Any]] = None):
        # Optionally seed the environment (for reproducibility)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        # Initialize the state
        # Init use:  winert_initalize_state_np(self, init_angle=True, state_w_ray = self.state_w_ray)
        self.state = winert_initalize_state_np(self, init_angle=True, re_tensor_node_edge = False, state_w_ray = self.state_w_ray, set_init_path_flag = self.set_init_path_flag )
        if self.dummy_input:
            self.state = np.array([0, 7, 2.5, 1, 0, 0], dtype=np.float32)
            print ("dummy_input is used")
        self.state_traj = np.zeros_like(self.state_traj)
        self.action_traj = np.zeros_like(self.action_traj)
        if self.set_init_path_flag:
            self.current_step = 1
            self.state_traj[0, :4] = WINeRTEnv.node_attr_tensor[self.Tx_id,self.Rx_id, 0 ,0,:4]
            assert self.distance(self.state_traj[0, 1:4], self.tx_coord) < 1e-3, f"self.state_traj[0, 1:4] {self.state_traj[0, 1:4]}, self.tx_coord {self.tx_coord}"
        else:
            self.current_step = 0            
        self.action_traj[self.current_step-1, 1:] = WINeRTEnv.edge_attr_tensor[self.Tx_id,self.Rx_id,self.init_path,0,1:3]
        self.action_traj[self.current_step-1, 0] =abs(WINeRTEnv.node_attr_tensor[self.Tx_id,self.Rx_id,self.init_path,1,0])
        self.init_value  = self.state [:4]
        self.state_traj[self.current_step, :] = self.state 
        if not self.dummy_input:
            try:
                assert self.distance(WINeRTEnv.node_attr_tensor[self.Tx_id, self.Rx_id, self.init_path, self.current_step, 1:4], self.state[1:4]) < 1e-3, f"WINeRTEnv.node_attr_tensor[self.Tx_id, self.Rx_id, self.init_path, self.current_step, 1:4] {WINeRTEnv.node_attr_tensor[self.Tx_id, self.Rx_id, self.init_path, self.current_step, 1:4]}, self.state[1:4] {self.state[1:4]}"
            except:
                raise ValueError("try to check if reset init_path is called before reset")
        return self.state, {}

    def step(self, action, action_angle = True, action_type = True, debug_w_gt_action = False, without_assert = False, action_print = False):
        if without_assert:
            action[...,0] = np.clip(action[...,0], 0, 6)
            action[...,1] = np.clip(action[...,1], -np.pi/2, np.pi/2)
            action[...,2] = np.clip(action[...,2], -np.pi, np.pi)
            
        assert action[...,0] < 6 + 1e-3, f"action[...,0] {action[...,0]}"
        assert action[...,0] >= 0 - 1e-3, f"action[...,0] {action[...,0]}"
        assert action[...,1] <= np.pi/2 + 1e-3, f"action[...,1] {action[...,1]}"
        assert action[...,1] >= -np.pi/2 - 1e-3, f"action[...,1] {action[...,1]}" 
        assert action[...,2] <= np.pi + 1e-3, f"action[...,2] {action[...,2]}"
        assert action[...,2] >= -np.pi - 1e-3, f"action[...,2] {action[...,2]}"
        
        # new_observations, rewards, dones, infos = env.step(actions)
        if self.state is None:
            raise ValueError("The environment must be reset before stepping.")
        # 2 steps, first get the reward, then update the state
        # Execute one time step within the environment
        assert action.shape[-1] == 3, f"action.shape {action.shape}"

        # record the action trajectory
        action_type = action[...,0]
        action_angle = action[...,1:3]
        self.current_step += 1
        self.action_traj[self.current_step-1, 0] = action_type
        self.action_traj[self.current_step-1, 1:3] =  action_angle
        info = {}
        
        state, non_valid_update_count = self.update_state(action_angle, action_type, self.state, get_non_update_count = True)
        self.state = state.numpy()
        if action_print:
            print (f"action: {action}, acquired state: {self.state}")
        non_valid_update_count = non_valid_update_count.numpy()
 
        if non_valid_update_count > 0:
            penalty = 1
        else:
            penalty = 0
        self.state_traj[self.current_step, :] = self.state

        if self.current_step < self.max_step:
            done = False
            log_reward = torch.tensor(0.0) - penalty
            info["done"] = False
            info["current_step"] = self.current_step
            info["mean_angular_loss"] = None
            info["receipt_los"]  = self.receipt_los
            
        else: 
            done = True
            log_reward, mean_angular_loss, _ = self.get_log_reward(self.action_traj, angular_loss_type='set_loss', type_loss = 'TF')  # Reward from the final state.
            
            info["done"] = True
            info["current_step"] = self.current_step
            info["mean_angular_loss"] = mean_angular_loss.cpu().numpy().item()
            # add the receipt loss
            max_valid_hop_nr, _ = torch.max(self.incremental_index_for_filter_path[...,self.init_path,:], -1)
            last_ray_start = self.state_traj[max_valid_hop_nr-1, 1:4]
            last_ray_stop = self.state_traj[max_valid_hop_nr, 1:4]
            intersect_1st_point = self.sphere_rx.ray_trace(last_ray_start, last_ray_stop, first_point=True)
            if len (intersect_1st_point) == 0:
                pass
            else:
                distance_ray = self.distance(last_ray_start, last_ray_stop)
                distance_intersect = self.distance(intersect_1st_point[0], last_ray_start)
                if distance_intersect < distance_ray:
                    self.receipt_los = 0
                else:
                    pass
            info["receipt_los"]  = self.receipt_los
            
        return self.state, log_reward.cpu().item(), done, False, info

    def update_state(self, action_angle, action_type, pre_state, get_non_update_count= False):
        prev_action_type = pre_state[0] 
        # prev action type, appened to state and use this to determine if the 2nd nearest rt point should be selected,
        new_x = winert_step_np(torch.tensor(pre_state),
                               self,
                               torch.tensor(action_angle),
                               torch.tensor(action_type),
                               prev_action_type,
                               'cpu').to(self.device) # this step have to be on CPU
        # Clamp each dimension separately and compute masks
        clamped_x = torch.clamp(new_x[1], max=10.01)
        mask_x = clamped_x != 10.01

        clamped_y = torch.clamp(new_x[2], max=5.01)
        mask_y = clamped_y != 5.01

        clamped_z = torch.clamp(new_x[3], max=3.01)
        mask_z = clamped_z != 3.01
        # they should not sum up but a union
        non_updated_count = torch.sum(~mask_x | ~mask_y | ~mask_z)        
        # Update x selectively
        state_update = torch.Tensor(self.state).clone()
        state_update[0] = torch.tensor(action_type) # update the action type
        state_update[1][mask_x] = clamped_x[mask_x].to(torch.float32)
        state_update[2][mask_y] = clamped_y[mask_y].to(torch.float32)
        state_update[3][mask_z] = clamped_z[mask_z].to(torch.float32)
        updated_state = state_update

        if get_non_update_count:
            return updated_state, non_updated_count
        else:
            return updated_state
        
    def render_env(self, face_rect, surface_index, vertices):
        # Assuming your data is in PyTorch tensors, convert them to NumPy arrays
        vertices = vertices.cpu().numpy()  # Convert vertices to NumPy array
        face_rect = face_rect.cpu().numpy()  # Convert face_rect to NumPy array

        # Triangulate each quadrilateral face
        tri_faces = []
        for face in face_rect:
            tri_faces.append([face[0], face[1], face[3]])  # First triangle
            tri_faces.append([face[2], face[3], face[0]])  # Second triangle

        # Convert to the format expected by PyVista
        # It needs the number of points in each face followed by the indices of the points
        faces = np.hstack([[3, *tri_face] for tri_face in tri_faces])
        mesh = pv.PolyData(vertices, faces)
        return mesh
    
    def find_env_id_or_index(self, storage_dir, env_id_or_index, find_id_or_index_opt = "env_id"):
        # Compile a regular expression pattern to match the filenames
        pattern = re.compile(r'node_attr_tensor_(\d+)_(\d+)\.pt')

        # Iterate through the files in the storage directory
        for filename in os.listdir(storage_dir):
            match = pattern.match(filename)
            if match:
                env_id = match.group(1)
                env_index = match.group(2)
                if env_index == str(env_id_or_index) and find_id_or_index_opt == "env_id":
                    return env_id
                elif env_id == str(env_id_or_index) and find_id_or_index_opt == "env_index":
                    return env_index
        # If no matching is found
        return None
        
    # debug API for rendering the path
    def render_path(self, return_dict = False, init_path = None):
        if init_path is not None:
            self.set_init_path(init_path)
        init_state, init_dict = self.reset()
        state_record = []
        dict_record = []       
        state_record.append(init_state)
        dict_record.append(init_dict)
        actual_action = []
        assert self.current_step == 1, f"self.current_step {self.current_step}"
        for i in range(4):
            actual_action.append(self.gt_action[i+1])
            next_obs, _, _, _, next_dict = self.step(self.gt_action[i+1], without_assert = True, action_print = True)
            state_record.append(next_obs)
            dict_record.append(next_dict)
        state_record = np.array(state_record)
        gt_coord = self.node_attr_tensor[self.Tx_id, self.Rx_id, self.init_path][1:,1:4]
        gt_state_coord = state_record[:,1:4]
        fig, path_dict = draw_path_single_n_simple((gt_state_coord, gt_coord),self.env_id,
                                            path_len_lim = self.max_incremental_index_for_filter_path[self.init_path],
                                            Tx_coord = self.tx_coord)
        if return_dict:
            return fig, path_dict
        else:
            return fig
