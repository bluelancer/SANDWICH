from torch.distributions import Normal
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

def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def spherical_to_ray_direction(azimuth, radian):
    x = torch.cos(radian) * torch.cos(azimuth)
    y = torch.cos(radian) * torch.sin(azimuth)
    z = torch.sin(radian)
    x_y_z = torch.stack([-x, -y, -z], dim=-1)
    return x_y_z

def clean_rt_result(points, rays, debug = False):
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
    # indices = torch.arange(len(ray_tensor))
    padded_points[inverse_indices, cumulative_indices- 1] = points_tensor
    assert padded_points.shape == torch.Size([len(unique_rays), max_ray_id_len, 3])
    if debug:
        y = [points_tensor[ray_tensor == ray_id] for ray_id in unique_rays]
        print("DEBUG: padded_points.shape", padded_points.shape,"y.shape", len(y))
        # Slow execution
        # y_tensor = torch.stack([torch.cat([y[i], torch.zeros(max(unique_counts) - y[i].shape[0], 3)]) for i in range(len(y))]).shape
        # Debug the padded_points
        for i in range(len(y)):
            # filter padding points's padded_points[i] != -1
            pd_point_cleaned = padded_points[i,:,:][torch.not_equal(padded_points[i], torch.tensor([-1,-1,-1]))]    
            yi_tensor = torch.tensor(y[i])    
            if torch.equal(yi_tensor,pd_point_cleaned.reshape(-1,3)):                                                                                                   
                pass
            else:
                print (f"i {i}, y[i] {y[i]}, padded_points[i] {padded_points[i]}")
                raise ValueError("y[i] and padded_points[i] are not equal")
    return padded_points, ray_tensor

def get_winert_policy_dist(model_angle, model_type, input_tensor, init_angle = False):
    """
    winert policy is a joint distribution of the interaction type and the parameters of the policy, interaction type is a categorical distribution, policy is a normal distribution
    Input: x is the state of the environment
    Input: type of interaction
    return: policy distribution
    """
    if init_angle:
        current_point, prev_angular = input_tensor 
        x = torch.cat([current_point, prev_angular], dim=-1)
        assert x.shape[-1] == 7, "x should have 7 dimensions, type x,y,z and count radian azimuth , but x.shape[-1] is {}".format(x.shape[-1])
    else:
        x = input_tensor
        assert x.shape[-1] == 5, "x should have 5 dimensions, type x,y,z and count , but x.shape[-1] is {}".format(x.shape[-1])
    
    pf_params = model_angle(x)    
    
    # TODO: how to combine the two distributions
    # In principle, we should sample from the interaction type first, then sample from the policy
    angle_mean = pf_params[:,:,:,:2]
    angle_std = torch.sigmoid(pf_params[:,:,:,2:]) * (max_policy_std - min_policy_std) + min_policy_std
    angle_dist = torch.distributions.Normal(angle_mean, angle_std)
    
    # This is the interaction type distribution
    inter_type = torch.nn.functional.one_hot(abs(x[:,:,:,0]).to(torch.int64), num_classes=6).to(torch.int16)
    rad_azi = x[:,:,:,5:7]
    type_input = torch.cat([x[:,:,:,1:4], inter_type, rad_azi], dim=-1)
    cf_params = model_type(type_input)

    interaction_type = torch.softmax(cf_params, dim=-1)
    interaction_type_dist = torch.distributions.Categorical(interaction_type)
    return angle_dist, interaction_type_dist

def winert_step(x, env_x, action_location, action_type, device, test_range_step = 2000, Tx_batch_size = 1, Rx_batch_size= 1, trail_batch_size=10):
    """same as step but with action type"""
    new_x = torch.zeros_like(x).to(device)
    action_location = action_location.to(device)
    print ("action_location.shape", action_location.shape)
    print ("action_type", action_type)
    # make action_location_radian_1d, action_location_azimuth_1d, x_coord_2dim to CPU, for pyvista
    action_location_radian_1d = torch.reshape(action_location[:Rx_batch_size,:,:,0],
                                              torch.Size([Rx_batch_size * action_location.shape[1]  * action_location.shape[2], 1])).to(torch.float16)
    action_location_azimuth_1d = torch.reshape(action_location[:Rx_batch_size,:,:,1],
                                               torch.Size([Rx_batch_size * action_location.shape[1]  * action_location.shape[2], 1])).to(torch.float16)    
    x_coord_2dim = torch.reshape(x[:,:,:,1:4],
                                 torch.Size([Rx_batch_size * trail_batch_size, 3])).to('cpu').to(torch.float16)
    
    ray_directions = spherical_to_ray_direction(action_location_azimuth_1d, action_location_radian_1d)
    ray_directions = ray_directions.squeeze()

    all_cleaned_points, _ = env_x.get_rt_result_single_process(x_coord_2dim, ray_directions,test_range = test_range_step , progress_bar = False)
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

    new_x[:,:,:, 1:4]  = result_points.reshape(torch.Size([Tx_batch_size, Rx_batch_size, trail_batch_size, 3])).to(torch.float16).to(device)
    new_x[:,:,:, 0] = x[:,:,:, 0] + action_type.squeeze()  # add action type
    new_x[:,:,:, 4] = x[:,:,:, 4] + 1  # Increment step counter.
    return new_x

def winert_initalize_state(device, env_x, Tx_batch_size = 1, Rx_batch_size = 1, trail_batch_size=10, init_angle = True):
    """Trajectory starts at state = (X_0, t=0)."""
    # winert_env.init_value.shape torch.Size([10, 1800, 30, 4])
    assert Tx_batch_size <= 10, "Tx_batch_size should be less than 10, but it is {}".format(Tx_batch_size)
    assert Rx_batch_size <= 1800, "Rx_batch_size should be less than 1800, but it is {}".format(Rx_batch_size)
    if not init_angle:
        x = torch.zeros(torch.Size([Tx_batch_size, Rx_batch_size, 30, 5]), device=device)
        x[:,:,:,:4] = env_x.init_value[Tx_batch_size,Rx_batch_size ,:,:].to(device)
        x[:,:,:, 4] = 0  # Initialize step counter.
    elif init_angle and trail_batch_size <= 30:
        x = torch.zeros(torch.Size([Tx_batch_size, Rx_batch_size, trail_batch_size, 7]), device=device)
        x[:,:,:,:4] = env_x.node_attr_tensor[:Tx_batch_size,:Rx_batch_size,:trail_batch_size,1,:4].to(device) # Init from the node 1, 
        assert x[:,:,:,:4].shape == torch.Size([Tx_batch_size, Rx_batch_size, trail_batch_size, 4]), "x[:,:,:,:4].shape should be torch.Size([{}, {}, {}, 4]), but it is {}".format(Tx_batch_size, Rx_batch_size, trail_batch_size, x[:,:,:,:4].shape)
        x[:,:,:, 4] = 0  # Initialize step counter.
        x[:,:,:, 5] = env_x.edge_attr_tensor[:Tx_batch_size,:Rx_batch_size,:trail_batch_size,0,1] # incoming radian 
        x[:,:,:, 6] = env_x.edge_attr_tensor[:Tx_batch_size,:Rx_batch_size,:trail_batch_size,0,2] # incoming azimuth
    elif init_angle and trail_batch_size > 30:
        x = torch.zeros(torch.Size([Tx_batch_size, Rx_batch_size, trail_batch_size, 7]), device=device)
        num_trail_batch = trail_batch_size // 30
        # print("prepare x with num_trail_batch: ", num_trail_batch)
        for i in range(num_trail_batch):
            start_index = i * 30
            end_index = (i + 1) * 30
            x[:,:,start_index:end_index,:4] = env_x.node_attr_tensor[:Tx_batch_size,:Rx_batch_size,:30,1,:4]
            x[:,:,start_index:end_index,4] = 0
            x[:,:,start_index:end_index,5] = env_x.edge_attr_tensor[:Tx_batch_size,:Rx_batch_size,:30,0,1]
            x[:,:,start_index:end_index,6] = env_x.edge_attr_tensor[:Tx_batch_size,:Rx_batch_size,:30,0,2]
        # the last batch
        start_index = num_trail_batch * 30
        end_index = trail_batch_size
        x[:,:,start_index:end_index,:4] = env_x.node_attr_tensor[:Tx_batch_size, :Rx_batch_size,:trail_batch_size - start_index,1,:4]
        x[:,:,start_index:end_index,4] = 0
        x[:,:,start_index:end_index,5] = env_x.edge_attr_tensor[:Tx_batch_size,:Rx_batch_size,:trail_batch_size - start_index,0,1]
        x[:,:,start_index:end_index,6] = env_x.edge_attr_tensor[:Tx_batch_size,:Rx_batch_size,:trail_batch_size - start_index,0,2]
        x = x.to(device)
    else:
        raise ValueError("init_angle should be True or False, but it is {}".format(init_angle))
    return x

def setup_experiment_winert_wAngle(device, hid_dim=128, lr_model=1e-3, lr_logz=1e-2):
    # TODO: maybe two hidden dim for each model
    forward_model_probAngle = torch.nn.Sequential(torch.nn.Linear(5+2, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, 4)).to(device)

    backward_model_probAngle = torch.nn.Sequential(torch.nn.Linear(5+2, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, 4)).to(device)
    # 4 input features, 4 output features mean1, mean2, std1, std2
    forward_model_probType = torch.nn.Sequential(torch.nn.Linear(5+2-2+6, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, 6)).to(device)

    backward_model_probType = torch.nn.Sequential(torch.nn.Linear(5+2-2+6, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, 6)).to(device) 
    # 6 input features, 6 output features for the interaction type
    logZ = torch.nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = torch.optim.Adam(
        [
            {'params': forward_model_probAngle.parameters(), 'lr': lr_model},
            {'params': backward_model_probAngle.parameters(), 'lr': lr_model},
            # {'params': forward_model_probType.parameters(), 'lr': lr_model},
            # {'params': backward_model_probType.parameters(), 'lr': lr_model},
            {'params': [logZ], 'lr': lr_logz},
        ]
    )

    return (forward_model_probAngle, backward_model_probAngle,forward_model_probType , backward_model_probType, logZ, optimizer)

def setup_experiment_winert(device, hid_dim=64, lr_model=1e-3, lr_logz=1e-1):
    # TODO: maybe two hidden dim for each model
    forward_model_probAngle = torch.nn.Sequential(torch.nn.Linear(5, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, 4)).to(device)

    backward_model_probAngle = torch.nn.Sequential(torch.nn.Linear(5, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, 4)).to(device)
    # 4 input features, 4 output features mean1, mean2, std1, std2
    forward_model_probType = torch.nn.Sequential(torch.nn.Linear(9, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, 6)).to(device)

    backward_model_probType = torch.nn.Sequential(torch.nn.Linear(9, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, 6)).to(device) 
    # 6 input features, 6 output features for the interaction type
    logZ = torch.nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = torch.optim.Adam(
        [
            {'params': forward_model_probAngle.parameters(), 'lr': lr_model},
            {'params': backward_model_probAngle.parameters(), 'lr': lr_model},
            {'params': forward_model_probType.parameters(), 'lr': lr_model},
            {'params': backward_model_probType.parameters(), 'lr': lr_model},
            {'params': [logZ], 'lr': lr_logz},
        ]
    )

    return (forward_model_probAngle, backward_model_probAngle,forward_model_probType , backward_model_probType, logZ, optimizer)


def train_WINert(env, Tx_batch_size, Rx_batch_size, trail_batch_size, device, n_iterations=2000, seed = 42,  trajectory_length = 5, cmd_vis = False, debug = False, init_angle = True):
    seed_all(seed)
    torch.autograd.set_detect_anomaly(True)
    losses = []
    mean_angular_losses = []
    # mean_type_losses = []
    non_updated_count_list = []
    logPB_angle_list = []
    # logPB_type_list = []
    logPF_angle_list = []
    # logPF_type_list = []
    last_trajectory_angular_list = []
    if init_angle:
        print("Training WINeRT with angle")
        iterator = range(1,trajectory_length)
        forward_model_probAngle, backward_model_probAngle, forward_model_probType , backward_model_probType, logZ, optimizer = setup_experiment_winert_wAngle(device)  # Default hyperparameters used.

    else:
        print("Training WINeRT without angle")
        iterator = range(0,trajectory_length-1)
        forward_model_probAngle, backward_model_probAngle, forward_model_probType , backward_model_probType, logZ, optimizer = setup_experiment_winert(device)

    if cmd_vis:
        tbar = trange(n_iterations)
    else:
        tbar = range(n_iterations)    
    print ("Training WINeRT with n_iterations: ", n_iterations)
    print ("Training WINeRT with trajectory_length: ", trajectory_length)
    print ("Training WINeRT with Tx_batch_size: ", Tx_batch_size)
    print ("Training WINeRT with Rx_batch_size: ", Rx_batch_size)
    print ("Training WINeRT with trail_batch_size: ", trail_batch_size)
    print ("Training WINeRT with init_angle: ", init_angle)
    print ("Training WINeRT with debug: ", debug)
    print ("Training WINeRT with cmd_vis: ", cmd_vis)
    print ("Training WINeRT with seed: ", seed)
    if not debug and not cmd_vis:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project="winert_gflownet",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epochs": n_iterations,
                "Tx_batch_size": Tx_batch_size,
                "Rx_batch_size": Rx_batch_size,
            },
        )
               
    for it in tbar:
        optimizer.zero_grad()
        x = winert_initalize_state(device, env, Tx_batch_size = Tx_batch_size, Rx_batch_size = Rx_batch_size, trail_batch_size = trail_batch_size, init_angle = init_angle)
        if init_angle:
            # type, x,y,z, count, radian, azimuth
            # Trajectory stores all of the states in the forward loop.
            trajectory_angular = torch.zeros(torch.Size([Tx_batch_size, Rx_batch_size, x.shape[2] ,trajectory_length + 1, 4]), device=device)
            # 4: type, radian, azimuth, count
            trajectory = torch.zeros(torch.Size([Tx_batch_size, Rx_batch_size, x.shape[2] ,trajectory_length + 1, 7]), device=device)
            # 7: type, x, y, z, count, radian, azimuth

        else:
            trajectory_angular = torch.zeros(torch.Size([Tx_batch_size, Rx_batch_size, x.shape[2] ,trajectory_length + 1, 4]), device=device)
            trajectory = torch.zeros(torch.Size([Tx_batch_size, Rx_batch_size, x.shape[2] ,trajectory_length + 1, 5]), device=device)
            # 5: type, x, y, z, count
        
        logPF_angle = torch.zeros(torch.Size([Tx_batch_size, Rx_batch_size, x.shape[2]]), device=device)
        logPB_angle = torch.zeros(torch.Size([Tx_batch_size, Rx_batch_size, x.shape[2]]), device=device)
        # logPF_type = torch.zeros(torch.Size([batch_size, x.shape[1], x.shape[2]]), device=device)
        # logPB_type = torch.zeros(torch.Size([batch_size, x.shape[1], x.shape[2]]), device=device)

        # Forward loop to generate full trajectory and compute logPF.
        if init_angle:
            assert x.shape[-1] == 7, "x should have 7 dimensions, type radian azimuth and count , but x.shape[-1] is {}".format(x.shape[-1]) 
            # feed the first step with the initial angle
            trajectory_angular[:,:,:, 0 + 1, 0]  = x[:, :, :,0].to(device)
            trajectory_angular[:,:,:, 0 + 1, 1:3] = x[:, :, :,5:7].to(device)
            trajectory_angular[:,:,:, 0 + 1, 3]  = 0
            
            trajectory [:,:,:, 0 + 1, 0] = x[:, :, :,0].to(device)
            trajectory [:,:,:, 0 + 1, 1:4] = x[:, :, :,1:4].to(device)
            trajectory [:,:,:, 0 + 1, 4] = 0
            trajectory [:,:,:, 0 + 1, 5:7] = x[:, :, :,5:7].to(device)
        for t in iterator:
            angle_dist, interaction_type_dist = get_winert_policy_dist(forward_model_probAngle, forward_model_probType, (x[:, :, :,:5],x[:, :, :,5:7])  , init_angle = init_angle)
            
            action_angle = angle_dist.sample()
            action_type = interaction_type_dist.sample()
            
            logPF_angle += angle_dist.log_prob(action_angle).sum(dim=-1)
            # logPF_type += interaction_type_dist.log_prob(action_type)
            
            new_x = winert_step(x, env, action_angle, action_type, 'cpu', test_range_step= 2000, Tx_batch_size = Tx_batch_size, Rx_batch_size = Rx_batch_size, trail_batch_size = trail_batch_size).to(device) # this step have to be on CPU
            # Subtract the small value from new_x

            # Clamp each dimension separately and compute masks
            clamped_x = torch.clamp(new_x[:, :, :, 1], max=10.01)
            mask_x = clamped_x != 10.01

            clamped_y = torch.clamp(new_x[:, :, :, 2], max=5.01)
            mask_y = clamped_y != 5.01

            clamped_z = torch.clamp(new_x[:, :, :, 3], max=3.01)
            mask_z = clamped_z != 3.01
            
            # they should not sum up but a union
            non_updated_count = torch.sum(~mask_x | ~mask_y | ~mask_z)
            non_updated_count_list.append(non_updated_count.item() / (new_x.shape[0] * new_x.shape[1] * new_x.shape[2]) * 100)
            
            # Update x selectively
            x_update = x.clone()
            x_update[:, :, :, 1][mask_x] = clamped_x[mask_x]
            x_update[:, :, :, 2][mask_y] = clamped_y[mask_y]
            x_update[:, :, :, 3][mask_z] = clamped_z[mask_z]
            x = x_update
            # trajectory_angular and trajectory are filled from hop 2 (t = 1) to trajectory_length - 1 (t = trajectory_length - 2) 
            trajectory_angular[:,:,:, t + 1, 0]  = action_type.squeeze().to(device)
            trajectory_angular[:,:,:, t + 1, 1:3] = action_angle.to(device) #new_x
            trajectory_angular[:,:,:, t + 1, 3]  = t
            
            trajectory [:,:,:, t + 1, 0] = action_type.squeeze().to(device)
            trajectory [:,:,:, t + 1, 1:4] = x[:, :, :,1:4].to(device)
            trajectory [:,:,:, t + 1, 4] = t
            trajectory [:,:,:, t + 1, 5:7] = action_angle.to(device)
            
        # Backward loop to compute logPB from existing trajectory under the backward policy.
        # TODO: Fix the iteration range
        for t in range(trajectory_length , 2 - 1, -1):
            angle_dist, interaction_type_dist = get_winert_policy_dist(backward_model_probAngle, backward_model_probType , (trajectory[:,:,:, t, :5], trajectory[:,:,:, t, 5:7]), init_angle = init_angle)
            
            action_angle = trajectory_angular[:,:,:, t, 1:3]
            action_type = trajectory_angular[:,:,:, t, 0]
            logPB_angle = logPB_angle + angle_dist.log_prob(action_angle).sum(dim=-1)
            # logPB_type = logPB_type + interaction_type_dist.log_prob(action_type)

        predicted_trajectory = trajectory_angular[:,:,:, 1:, :]
        # print ("predicted_trajectory.shape", predicted_trajectory.shape)
        # print ("env.edge_attr_tensor[1,0,0,:2,1:3]", env.edge_attr_tensor[1,0,0,:2,1:3])
        # print ("predicted_trajectory[0,0,0,:2,1:3]", predicted_trajectory[0,0,0,:2,1:3])        
        log_reward, mean_angular_loss, _ = env.get_log_reward(predicted_trajectory, Tx_batch_size, Rx_batch_size, trail_batch_size = trail_batch_size, angular_loss_type='set_loss', type_loss = 'TF')  # Reward from the final state.

        # Compute Trajectory Balance Loss.
        if debug:
            print ("logPF_angle.shape", logPF_angle.shape)
            # print ("logPF_type.shape", logPF_type.shape)
            # print ("logPB_type.shape", logPB_type.shape)
            print ("log_reward.shape", log_reward.shape)
        # loss = (logZ + logPF_angle + logPF_type - logPB_angle - logPB_type - log_reward).pow(2).mean()
        loss = (logZ + logPF_angle - logPB_angle - log_reward).pow(2).mean()

        loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(loss.item())
        mean_angular_losses.append(mean_angular_loss.mean().item())
        # mean_type_losses.append(mean_type_loss.mean().item())
        logPB_angle_list.append(logPB_angle.mean().item())
        # logPB_type_list.append(logPB_type.mean().item())
        logPF_angle_list.append(logPF_angle.mean().item())
        # logPF_type_list.append(logPF_type.mean().item())
        last_mean_angular_loss = 0
        last_trajectory_angular_list.append(predicted_trajectory)
        if it % 10 == 0:
            if cmd_vis:
                mean_loss = np.array(losses[-10:]).mean()
                mean_angular_loss_value = np.array(mean_angular_losses[-10:]).mean()
                mean_non_updated_count_list = np.array(non_updated_count_list[-10:]).mean()
                tbar.set_description("Training iter {}: (loss={:.3f},MAL={:.3f}, NUratio={:.3f} estimated logZ={:.3f}, LR={}".format(
                    it,
                    mean_loss,
                    mean_angular_loss_value, 
                    mean_non_updated_count_list,
                    logZ.item(),
                    optimizer.param_groups[0]['lr'],
                    )
                )
                # print (f"slice of trajectory trajectory[0,0,0,1:3,:]: {trajectory_angular[0,0,0,1:,1:3]} at step {it}")
                # print (f"slice of env.edge_attr_tensor[0,0,0,1:3,:]: {env.edge_attr_tensor[1,0,0,:,1:3]} at step {it}")
                if it > 0 and mean_angular_loss_value > last_mean_angular_loss + 5:
                    print ("mean_angular_loss_value > last_mean_angular_loss, break")
                    import ipdb; ipdb.set_trace()
                last_mean_angular_loss = mean_angular_loss_value
            elif not debug and not cmd_vis:
                loss_value = np.array(losses[-10:]).mean()
                mean_angular_loss_value = np.array(mean_angular_losses[-10:]).mean()
                # mean_type_loss_value = np.array(mean_type_losses[-10:]).mean()
                non_updated_count_ratio = np.array(non_updated_count_list[-10:]).mean()
                logZ_value = logZ.item()
                learning_rate = optimizer.param_groups[0]['lr']
                logPB_angle_value = np.array(logPB_angle_list[-10:]).mean()
                # logPB_type_value = np.array(logPB_type_list[-10:]).mean()
                logPF_angle_value = np.array(logPF_angle_list[-10:]).mean()
                # logPF_type_value = np.array(logPF_type_list[-10:]).mean()
                wandb.log({"Training iter": it, 
                           "loss": loss_value,
                           "mean_angular_loss": mean_angular_loss_value,
                           "non_update_ratio": non_updated_count_ratio,
                           "estimated logZ": logZ_value,
                           "LR": learning_rate,
                           "logPB_angle": logPB_angle_value,
                           "logPF_angle": logPF_angle_value})
    if not cmd_vis:
        run.log_code()
    return (forward_model_probAngle, backward_model_probAngle, forward_model_probType , backward_model_probType, logZ)

def inference_winert(trajectory_length, forward_model_probAngle,forward_model_probType, env, device, batch_size=10_000):
    """Sample some trajectories."""

    with torch.no_grad():
        trajectory =  torch.zeros(torch.Size([batch_size, 1800, 30 ,trajectory_length + 1, 4]), device=device)
        trajectory[:, 0, 0] = env.init_value[batch_size,:,:,:].to(device)
        # x[:,:,:,:4] = env_x.init_value[batch_size,:,:,:].to(device)
        x = winert_initalize_state(device, env)

        for t in range(trajectory_length):
            # angle_dist, interaction_type_dist = get_winert_policy_dist(forward_model_probAngle, forward_model_probType, x)
            angle_dist, interaction_type_dist = get_winert_policy_dist(forward_model_probAngle, forward_model_probType, x)

            action_angle = angle_dist.sample()
            action_type = interaction_type_dist.sample()

            new_x = winert_step(x, env, action_angle, action_type, 'cpu', test_range_step= 2000, Tx_batch_size = Tx_batch_size, Rx_batch_size = Rx_batch_size, trail_batch_size = trail_batch_size).to(device) # this step have to be on CPU

            trajectory[:, t + 1, :] = new_x
            x = new_x

    return trajectory

def _debug_winert_step(x, env, device):
    x_init_test = winert_initalize_state(device, env).to(device)
    action_angle_test = env.edge_attr_tensor[:,:,:,0,1:3].to(device)
    action_type_test = env.node_attr_tensor[:,:,:,1,0].to(device)
    new_x_test = winert_step(x_init_test ,env, action_angle_test, action_type_test, device)
    gt_test = env.node_attr_tensor[:,:,:,1,1:4]
    for Tx in trange(gt_test.shape[0]):
        for Rx in range(gt_test.shape[1]):
            for t in range(gt_test.shape[2]):
                if env.distance(gt_test[Tx,Rx,t], new_x_test[Tx,Rx,t,1:4]) < 1e-2:
                    continue
                else:
                    raise ValueError(f"Tx {Tx}, Rx {Rx}, t {t}, gt_test {gt_test[Tx,Rx,t]}, new_x_test {new_x_test[Tx,Rx,t,1:4]}")
    
