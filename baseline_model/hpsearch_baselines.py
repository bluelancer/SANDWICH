import os
os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"]="1"
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from utils_downstream.Baseline_Model import BaselineMLP
from utils_downstream.data_utils import *
from utils_downstream.train_utils import *
from utilsgyms.utils_decisiontrans import *

# Hyperparameter Turning
from functools import partial
from pathlib import Path
import ray
# use 1 MIG GPU
# https://discuss.ray.io/t/how-to-change-the-logging-path/8817 
ray.init(num_gpus=1,include_dashboard=False, _temp_dir = "/proj/raygnn/workspace/ray_logs")
from ray.tune.schedulers import ASHAScheduler
import pickle
import ray.cloudpickle as r_pickle
from ray import tune
from utils_downstream.HPsearch_baseline_utils import *
import datetime

def main(gpus_per_trial, max_num_epochs = 100, num_samples = 1):
    # num_samples = 10 # number of trials, for EACH HP combination this do not changes "The actor ImplicitFunc is too large (962 MiB > FUNCTION_SIZE_ERROR_THRESHOLD=95 MiB)."
    # max_num_epochs = 30 # It means: max time units per trial. Trials will be stopped after max_t time units, https://discuss.ray.io/t/pytorch-tutorial-understanding/8172
    env_index, _, test_seed, _, _, test_data = parse_input(ds_task = True)
    seed_everything(test_seed) 
    data_postfix = "_noise_noise_5"
    config_file, current_base_path = get_config_basepath(allTx=True)

    # Torch:  Load the data
    train_feat_tensor, test_feat_tensor, _, train_label_tensor, test_label_tensor, result_dir = get_data_dict(env_index, config_file, current_base_path, data_postfix, test_data, return_path = "result_path", prep_baseline = True, cluster= "tetralith")
    assert train_label_tensor.size(-1) == test_label_tensor.size(-1), "Label dimension mismatch"
    # Initialize the model, loss function and optimizer
    input_dim = train_feat_tensor.size(1)
    output_dim = train_label_tensor.size(1)
    # # make tensor into dataset
    test_dataset = TensorDataset(test_feat_tensor, test_label_tensor)
    # RAY: search space
    config = {
        "batch_size": tune.grid_search([16, 32, 64, 128, 256, 512, 1024]),
        "loss": tune.grid_search(["log_angular","angular", "non_angular"]),
        "seed": test_seed,
    }
    scheduler = ASHAScheduler(
        metric="test_RSSI_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=20,
        reduction_factor=2,
    )
    #### PLAN A: use tune.run on a function trainable ########
    # THis is suggested in Torch tutorial: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    trainable = partial(train_baseline_mlp, env_index = env_index, config_file = config_file, current_base_path = current_base_path, data_postfix = data_postfix, test_data= test_data)
    result = tune.run(
        trainable,
        resources_per_trial={"cpu": 3, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path = config_file["DOWNSTREAM"]["ray_storage_tetralith"],
        checkpoint_at_end=False)
    # Doc on tune.run: https://docs.ray.io/en/releases-1.13.0/tune/api_docs/execution.html#tune-run

    best_trial = result.get_best_trial("test_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final RSSI loss: {best_trial.last_result['test_RSSI_loss']}")
    print (f"Best trial final Angle loss: {best_trial.last_result['test_angular_reward']}")
    best_trained_model = BaselineMLP(input_dim, output_dim)
    # Find device
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    best_trained_model.to(device)
    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="test_loss", mode="min")
    
    pickle.dump(best_checkpoint, open(result_dir +"/"+ "best_MLP_checkpoint.pkl", "wb"))
    pickle.dump(best_trial.config, open(result_dir  +"/"+  "best_MLP_trial_config.pkl", "wb"))
    print (f"Best checkpoint & config SAVED!")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "checkpoint.pt"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = r_pickle.load(fp)
        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_loss,RSSI_loss,Angle_reward, list_MLP_RSSI_loss, list_MLP_all_path_length, list_MLP_RSSI_all_gts, list_MLP_all_angle_reward_array = test_baseline_mlp_process(best_trained_model, test_dataset,best_trial.config["batch_size"], nn.L1Loss(reduction = 'none'), return_data = False, loss_type = best_trial.config["loss"])
        print(f"Env Index: {env_index}, Best trial RSSI_loss: {RSSI_loss}, Angle_reward: {Angle_reward}")
        result_dict = {"MLP_RSSI_loss": list_MLP_RSSI_loss, 
                       "path_length": list_MLP_all_path_length, 
                       "RSSI": list_MLP_RSSI_all_gts, 
                       "MLP_angle_reward": list_MLP_all_angle_reward_array,
                       "test_loss": test_loss} 
        pickle.dump(result_dict, open(result_dir  +"/"+  "MLP_result_dict.pkl", "wb"))
        print (f"Result dict SAVED!")
        datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    draw_multiple_bars_same_plot_ray_ds(result_dict,
                        ["MLP_RSSI_loss"],
                        ["RSSI", "path_length"],
                        result_dir,
                        datetime_str,
                        env_id=env_index,
                        load_from_file=True,
                        test_data = test_data)
    ray.shutdown()
if __name__ == "__main__":
    main(gpus_per_trial=0.09)

