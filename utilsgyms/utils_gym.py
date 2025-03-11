from torch.distributions import Normal
import math
import numpy as np
import torch
import random
from tqdm import trange

trajectory_length = 5
min_policy_std = 0.1
max_policy_std = 1.0
batch_size = 256
seed = 4444

def spherical_to_ray_direction_np(azimuth, radian):
    x = torch.cos(radian) * torch.cos(azimuth)
    y = torch.cos(radian) * torch.sin(azimuth)
    z = torch.sin(radian)
    x_y_z = torch.stack([-x, -y, -z], dim=-1)
    return x_y_z

def winert_initalize_state_np(env_x, init_angle = True, re_tensor_node_edge = True, state_w_ray = True, set_init_path_flag = True):
    """Trajectory starts at state = (X_0, t=0)."""
    if not re_tensor_node_edge:
        node_attr_tensor = torch.tensor(env_x.node_attr_tensor, dtype=torch.float32, device=env_x.device)
        edge_attr_tensor = torch.tensor(env_x.edge_attr_tensor, dtype=torch.float32, device=env_x.device)
    else: 
        node_attr_tensor = env_x.node_attr_tensor
        edge_attr_tensor = env_x.edge_attr_tensor
    if not init_angle:
        x = np.zeros([5])
        x[:4] = env_x.init_value
        x[4] = 0  # Initialize step counter.
    elif init_angle: # attach in-angle (aziz, radian) to the init state
        if state_w_ray:
            x = np.zeros([6])
            if set_init_path_flag:
                x[:4] = node_attr_tensor[env_x.Tx_id, env_x.Rx_id, env_x.init_path,1,:4]#.cpu().numpy() # Note: here we directly feed the 1st hop's node attribute
            else:
                x[:4] = node_attr_tensor[env_x.Tx_id, env_x.Rx_id, env_x.init_path,0,:4]#.cpu().numpy() # else, we feed the 0th hop's node attribute
            # x[4] = 0  # Initialize step counter.
            x[4] = edge_attr_tensor[env_x.Tx_id, env_x.Rx_id, env_x.init_path,0,1]#.cpu().numpy() # incoming radian 
            x[5] = edge_attr_tensor[env_x.Tx_id, env_x.Rx_id, env_x.init_path,0,2]#.cpu().numpy() # incoming azimuth
        else:
            x = np.zeros([4])
            x[:4] = node_attr_tensor[env_x.Tx_id, env_x.Rx_id, env_x.init_path,0,:4]#.cpu().numpy()
    else:
        raise ValueError("init_angle should be True or False, but it is {}".format(init_angle))
    return x

def winert_step_np(x, env_x, action_location, action_type, prev_action_type, device, state_w_ray = True):
    """same as step but with action type"""
    new_x = torch.zeros_like(x)
    if state_w_ray:
        assert x.shape[0] == 6, "x should have 6 elements, since state_w_ray is True but it has {}".format(x.shape[0])
    else:
        assert x.shape[0] == 4, "x should have 4 elements, since state_w_ray is False but it has {}".format(x.shape[0])
        
    action_location =action_location.to(device) # in case self.device is on GPU
    # import ipdb; ipdb.set_trace()
    # make action_location_radian_1d, action_location_azimuth_1d, x_coord_2dim to CPU, for pyvista
    action_location_radian_1d = action_location[...,0].to(torch.float16)
    action_location_azimuth_1d = action_location[...,1].to(torch.float16)    
    x_coord_2dim = x[1:4].to(device).to(torch.float16)
    
    ray_directions = spherical_to_ray_direction_np(action_location_azimuth_1d, action_location_radian_1d)
    ray_directions = ray_directions.squeeze()

    all_cleaned_points, _ = env_x.get_rt_result_single_process_np(x_coord_2dim, ray_directions)

    # Calculate the condition for rolling, assuming all_cleaned_points and x_coord_2dim are defined
    # roll_condition = abs(torch.norm(all_cleaned_points - x_coord_2dim.unsqueeze(-2), dim=-1)) >= 1e-4
    dst = torch.norm(all_cleaned_points - x_coord_2dim.unsqueeze(-2), dim=-1)
    # roll_condition = abs(dst) >= 1e-4
    not_same_point = abs(dst) >= 1e-4
    if not_same_point.sum() == 0:
        # print("Warning: all points are the same, the agent should be stuck at the same point")
        # get the first point as the not_same_point mask
        not_same_point = torch.ones_like(not_same_point)
        
    # Prepare a mask for points that are not [-1,-1,-1]
    not_negative_ones = ~torch.all(all_cleaned_points == torch.tensor([-1, -1, -1]), dim=-1)

    # Combine conditions: we want to roll points that meet the roll condition and are not [-1,-1,-1]
    # combined_condition = roll_condition & not_negative_ones
    combined_condition = not_same_point & not_negative_ones
    
    # next, we use the combined_condition to select the valid points and get the indices of smallest dst
    chosen_point = all_cleaned_points[combined_condition] # shape: [batch_size, 3]
    chosen_dst = dst[combined_condition] # shape: [batch_size]
    # import ipdb; ipdb.set_trace()
    if prev_action_type == 3: # if penetration
        # get the 2nd closest point as penetration interaction
        k_index = 2
    else:
        k_index = 1
    if len(chosen_point.shape) < 3: # if batch free
        # import ipdb; ipdb.set_trace()
        if chosen_point.shape[0] == 1:
            k_index = 1 # we only find one single point, so the ray already penertrate the wall
        _, indices = torch.topk(-chosen_dst.float(),k_index , dim=0)
        result_points = chosen_point[indices[-1]]
    else: # if batched
        if chosen_point.shape[1] == 1:
            k_index = 1 
        _, indices = torch.topk(-chosen_dst.float(), k_index, dim=1)
        # result_points = chosen_point[torch.arange(chosen_point.shape[0]), indices]
        result_points = chosen_point[indices[:,-1]]
    # # Count valid points (not [-1, -1, -1])
    # valid_points_mask = not_negative_ones
    # # valid_points_count = valid_points_mask.sum(dim=1)
    # valid_points_count = combined_condition.sum(dim=1)

    # # Roll conditionally: only if there's more than one valid point
    # # Otherwise, keep the first valid point
    # # rolled_indices = torch.roll(combined_condition, shifts=-1, dims=1)  # UPDATE: NO ROLLING
    # condition_shape = (valid_points_count > 1).unsqueeze(-1)
    
    # selected_indices = torch.where(condition_shape, rolled_indices, valid_points_mask)
    
    # # Now find the index of the first 'True' condition in selected_indices for each batch item, UPDATE: NO! We should select the last valid point, i.e. the last 'True' condition
    # # indices = (selected_indices.cumsum(dim=1) == 1).max(dim=1).indices # this was prev code, before UPDATE
    # # indices = (selected_indices.cumsum(dim=1) == selected_indices.shape[1]).max(dim=1).indices
    # # Update Again: It seems neither the first nor the last valid point is the best choice,  we should select the closest point to the previous point 
    # # Update Again: We should select the closest point to the previous point, so we should select the point with the smallest distance to the previous point
    # # Update Again: TODO: Consider reuse the 2nd closest point as PENERTRATION intereaction, should be crisp, Update: Fixed 
    # _, indices = torch.min(dst + ~selected_indices * 1e6, dim=1)

    # Using the gathered indices to select points from all_cleaned_points
    # result_points = all_cleaned_points[torch.arange(all_cleaned_points.shape[0]), indices]

    new_x[1:4]  = result_points.to(torch.float16).to(device)
    new_x[0] =  action_type.squeeze()  # add action type
    if state_w_ray:
        new_x[4:] = action_location.to(torch.float16).to(device)
        assert new_x.shape[0] == 6, "new_x should have 6 elements, but it has {}".format(new_x.shape[0])
    # new_x[0] = x[0] + action_type.squeeze()  # add action type
    # new_x[4] = x[4] + 1  # Increment step counter.
    
    return new_x

def clean_rt_result_np(points, rays):
    assert len(points) == len(rays), "points and rays should have the same length, but points {} and rays {}".format(len(points), len(rays))
    assert len(points) > 0, "points should not be empty"
    ray_tensor = torch.tensor(rays).to(torch.float16)
    points_tensor = torch.tensor(points).to(torch.float16)
    unique_rays, inverse_indices, counts = torch.unique(ray_tensor, return_inverse=True, return_counts=True)
    
    # record each element appearance order in ray_tensor
    # like 1 for the first appearance, 2 for the second appearance for each ray
    # Creating a 2D tensor where each row represents an element in ray_tensor and each column represents a unique element
    expanded_ray_tensor = ray_tensor.unsqueeze(1).expand(-1, len(unique_rays))
    expanded_unique_elements = unique_rays.unsqueeze(0).expand(len(ray_tensor), -1)

    # Generating a mask of where elements in ray_tensor match the unique elements
    match_mask = expanded_ray_tensor == expanded_unique_elements

    # Calculating the cumulative sum for each column in the mask, representing cumulative appearances of each unique element
    cumulative_appearances = match_mask.cumsum(dim=0)

    # Using the mask to filter the cumulative appearances back to the shape of ray_tensor, representing each element's appearance order
    cumulative_indices = torch.masked_select(cumulative_appearances, match_mask)

    max_ray_id_len = counts.max()

    # Allocate tensor of the right shape: [number of unique ray_id, max_ray_id_len, 3]
    # Fill it with some placeholder value e.g., -1 or 0
    padded_points = points_tensor.new_full((len(unique_rays), max_ray_id_len, 3), -1)

    # Use a scatter operation to fill in the points
    padded_points[inverse_indices, cumulative_indices- 1] = points_tensor
    assert padded_points.shape == torch.Size([len(unique_rays), max_ray_id_len, 3])
    return padded_points, ray_tensor
