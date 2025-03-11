from torch.distributions import Normal
import math
import numpy as np
import torch
import random
from tqdm import trange
# from torchviz import make_dot
import wandb

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

def winert_initalize_state_np(env_x, init_angle = True, First_time = True):
    """Trajectory starts at state = (X_0, t=0)."""
    if not First_time:
        node_attr_tensor = torch.tensor(env_x.node_attr_tensor, dtype=torch.float32, device=env_x.device)
        edge_attr_tensor = torch.tensor(env_x.edge_attr_tensor, dtype=torch.float32, device=env_x.device)
    else: 
        node_attr_tensor = env_x.node_attr_tensor
        edge_attr_tensor = env_x.edge_attr_tensor
    if not init_angle:
        x = np.zeros([5])
        x[:4] = env_x.init_value
        x[4] = 0  # Initialize step counter.
    elif init_angle:
        # import ipdb; ipdb.set_trace()
        x = np.zeros([6])
        x[:4] = node_attr_tensor[env_x.Tx_id, env_x.Rx_id, env_x.init_path,0,:4].cpu().numpy()
        # x[4] = 0  # Initialize step counter.
        x[4] = edge_attr_tensor[env_x.Tx_id, env_x.Rx_id, env_x.init_path,0,1].cpu().numpy() # incoming radian 
        x[5] = edge_attr_tensor[env_x.Tx_id, env_x.Rx_id, env_x.init_path,0,2].cpu().numpy() # incoming azimuth
    else:
        raise ValueError("init_angle should be True or False, but it is {}".format(init_angle))
    return x

def winert_step_np(x, env_x, action_location, action_type, device):
    """same as step but with action type"""
    new_x = torch.zeros_like(x)
    action_location = torch.tensor(action_location).to(device) # in case self.device is on GPU
    # print ("action_location.shape", action_location.shape)
    # print ("action_location: ", action_location)
    # print ("action_type", action_type)
    # make action_location_radian_1d, action_location_azimuth_1d, x_coord_2dim to CPU, for pyvista
    action_location_radian_1d = action_location[0].to(torch.float16)
    action_location_azimuth_1d = action_location[1].to(torch.float16)    
    x_coord_2dim = x[1:4].to(device).to(torch.float16)
    
    ray_directions = spherical_to_ray_direction_np(action_location_azimuth_1d, action_location_radian_1d)
    ray_directions = ray_directions.squeeze()

    all_cleaned_points, _ = env_x.get_rt_result_single_process_np(x_coord_2dim, ray_directions)
    # print ("all_cleaned_points", all_cleaned_points)
    # print ("x_coord_2dim", x_coord_2dim)
    # Calculate the condition for rolling, assuming all_cleaned_points and x_coord_2dim are defined
    roll_condition = abs(torch.norm(all_cleaned_points - x_coord_2dim.unsqueeze(-2), dim=-1)) >= 1e-4

    # Prepare a mask for points that are not [-1,-1,-1]
    not_negative_ones = ~torch.all(all_cleaned_points == torch.tensor([-1, -1, -1]), dim=-1)

    # Combine conditions: we want to roll points that meet the roll condition and are not [-1,-1,-1]
    combined_condition = roll_condition & not_negative_ones

    # Count valid points (not [-1, -1, -1])
    valid_points_mask = not_negative_ones
    valid_points_count = valid_points_mask.sum(dim=1)

    # Roll conditionally: only if there's more than one valid point
    # Otherwise, keep the first valid point
    rolled_indices = torch.roll(combined_condition, shifts=-1, dims=1)
    condition_shape = (valid_points_count > 1).unsqueeze(-1)
    selected_indices = torch.where(condition_shape, rolled_indices, valid_points_mask)
    # Now find the index of the first 'True' condition in selected_indices for each batch item
    indices = (selected_indices.cumsum(dim=1) == 1).max(dim=1).indices

    # Using the gathered indices to select points from all_cleaned_points
    result_points = all_cleaned_points[torch.arange(all_cleaned_points.shape[0]), indices]

    new_x[1:4]  = result_points.to(torch.float16).to(device)
    new_x[0] = x[0] + action_type.squeeze()  # add action type
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
