import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from utils_downstream.Model import SimpleModel
from utils_downstream.data_utils import *
from utils_downstream.train_utils import *
from utilsgyms.utils_decisiontrans import *

# Hyperparameter Turning
from functools import partial
from pathlib import Path
import ray
# use 1 MIG GPU
ray.init(num_gpus=1,include_dashboard=False)
from ray.tune.schedulers import ASHAScheduler
import pickle
import ray.cloudpickle as r_pickle
from ray import tune
from utils_downstream.HPsearch_utils import *
import datetime

def main(gpus_per_trial, max_num_epochs = 60, num_samples = 1):
    # num_samples = 10 # number of trials, this do not changes "The actor ImplicitFunc is too large (962 MiB > FUNCTION_SIZE_ERROR_THRESHOLD=95 MiB)."
    env_index, _, test_seed, _, _, test_data = parse_input(ds_task = True)
    seed_everything(test_seed) 
    # https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.enable_reproducibility.html

    data_postfix = "_noise_noise_5"
    config_file, current_base_path = get_config_basepath(allTx=True)

    # Torch:  Load the data
    train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor, result_dir = get_data_dict(env_index, config_file, current_base_path, data_postfix, test_data, return_path = "result_path")
    assert train_feat_tensor.size(-1) == test_feat_tensor.size(-1) == pred_feat_tensor.size(-1), "Feature dimension mismatch"
    assert train_label_tensor.size(-1) == test_label_tensor.size(-1), "Label dimension mismatch"
    # Initialize the model, loss function and optimizer
    input_dim = train_feat_tensor.size(1)
    output_dim = train_label_tensor.size(1)
    # # make tensor into dataset
    test_dataset = TensorDataset(test_feat_tensor, test_label_tensor)
    pred_dataset = TensorDataset(pred_feat_tensor, test_label_tensor)
    # RAY: search space
    config = {
        "n_layers": tune.grid_search([0,1,2,7,8]),
        "lr": tune.loguniform(1e-4, 5e-3),
        "batch_size": tune.grid_search([512, 1024, 2048, 4096]),
        "epochs": tune.grid_search([30, 40, 50, 60]),
        "seed": test_seed,
    }
    scheduler = ASHAScheduler(
        metric="pred_loss_sum",
        mode="min",
        max_t=max_num_epochs,
        grace_period=30,
        reduction_factor=2,
    )
    #### PLAN A: use tune.run on a function trainable ########
    # THis is suggested in Torch tutorial: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    trainable = partial(train_rayds, env_index = env_index, config_file = config_file, current_base_path = current_base_path, data_postfix = data_postfix, test_data= test_data)
    result = tune.run(
        trainable,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path = config_file["DOWNSTREAM"]["ray_storage_berzelius"],
        checkpoint_at_end=False)
    # Doc on tune.run: https://docs.ray.io/en/releases-1.13.0/tune/api_docs/execution.html#tune-run

    best_trial = result.get_best_trial("pred_loss_sum", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['test_loss']}")
    print (f"Best trial final pred_loss_sum: {best_trial.last_result['pred_loss_sum']}")
    best_trained_model = SimpleModel(input_dim, output_dim, best_trial.config["n_layers"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)
    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="pred_loss_sum", mode="min")
    pickle.dump(best_checkpoint, open(result_dir +"/"+ "best_checkpoint.pkl", "wb"))
    pickle.dump(best_trial.config, open(result_dir  +"/"+  "best_trial_config.pkl", "wb"))
    print (f"Best checkpoint & config SAVED!")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "checkpoint.pt"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = r_pickle.load(fp)
        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        # test_loss, pred_test_diff_loss, pred_loss_sum = test_process(best_trained_model, test_dataset, pred_dataset, best_trial.config["batch_size"], nn.L1Loss(reduction = 'none'), return_data = False)
        test_loss, pred_test_diff_loss, pred_loss_sum, list_all_test_loss, list_all_pred_loss_sum, list_all_pred_test_diff_loss, list_all_path_length, list_all_gts = test_process(best_trained_model, test_dataset, pred_dataset, best_trial.config["batch_size"], nn.L1Loss(reduction = 'none'), return_data = True)
        print(f"Env Index: {env_index}, Best trial test_loss: {test_loss}, pred_test_diff_loss: {pred_test_diff_loss}, pred_loss_sum: {pred_loss_sum}")
        result_dict = {"test_loss": list_all_test_loss, 
                       "pred_loss": list_all_pred_loss_sum, 
                       "pred_test_diff": list_all_pred_test_diff_loss, 
                       "path_length": list_all_path_length,
                       "RSSI": list_all_gts} 
        pickle.dump(result_dict, open(result_dir  +"/"+  "result_dict.pkl", "wb"))
        print (f"Result dict SAVED!")
        datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        draw_multiple_bars_same_plot_ray_ds(result_dict,
                                ["test_loss", "pred_loss","pred_test_diff"],
                                ["path_length", "RSSI"],
                                result_dir,
                                datetime_str,
                                env_id=env_index)
    ray.shutdown()
if __name__ == "__main__":
    main(gpus_per_trial=0.02)

