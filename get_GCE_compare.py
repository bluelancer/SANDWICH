import pickle
from utilsgyms.draw_figure import *
from datetime import datetime
from utils_downstream.data_utils import *
from utils_downstream.train_utils import *
from tqdm import tqdm
from utils_downstream.Model import SimpleModel
from torch.utils.data import TensorDataset
from pathlib import Path
import ray.cloudpickle as r_pickle
from utils_downstream.HPsearch_utils import *
import pandas as pd
def find_latest_file(loading_path, file_pattern, test_data):
    all_files = os.listdir(loading_path)
    if file_pattern in ["all_log_ave_ang_loss_dict", "allTx_all_path_length_dict"]:
        searching_pattern = f"{file_pattern}_\d{{8}}-\d{{6}}.pkl"
    elif file_pattern in ["MLP_result_dict", "KNN_result_dict"]:
        # MLP_result_dict_test_2024-08-15_02-37-46.pkl
        searching_pattern = f"{file_pattern}_{test_data}_\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}.pkl"
    else:
        raise NotImplementedError 
    data_matches = []
    for file in all_files:
        if re.match(searching_pattern, file):
            if file_pattern in ["all_log_ave_ang_loss_dict", "allTx_all_path_length_dict"]:
                data_match = file.split("_")[-1].split(".")[0]
            elif file_pattern in ["MLP_result_dict", "KNN_result_dict"]:
                data_match_part1 = file.split("_")[-2].split(".")[0]
                data_match_part2 = file.split("_")[-1].split(".")[0]
                data_match = f"{data_match_part1}_{data_match_part2}"
            data_matches.append(data_match)
    if len(data_matches) == 0:
        raise ValueError(f"No model is found in {loading_path} with pattern {searching_pattern}")
    elif len(data_matches) == 1:
        data_match = data_matches[0]
    else:
        print (f"multiple models are found, using the latest one, {max(data_matches)}")
        data_match = max(data_matches)
    print (f"Loading {file_pattern} from {data_match}")
    if file_pattern in ["all_log_ave_ang_loss_dict", "allTx_all_path_length_dict"]:
        file_name = f"{file_pattern}_{data_match}"
    elif file_pattern in ["MLP_result_dict", "KNN_result_dict"]:
        file_name = f"{file_pattern}_{test_data}_{data_match}"
    return file_name, data_match

def reshape_array(array, new_shape):
    return array.reshape(new_shape)
NO_ENV_IDS = [9,10,13,18,29,31,32,33,43,45,47,48,49,59,64,67,71,72,77,88,90,92,93,94,99]
# DEBUG = True
def main():
    criterion = nn.L1Loss()
    df = pd.DataFrame()
    _, _, _, _, allTx, _, _,  _, _, _, _, _, _, _, _,_, _, _, data_postfix, _, _, noise_sample, _, _ = parse_input()
    env_iter = range(1,50)
    for env_index in env_iter:
        if env_index in NO_ENV_IDS:
            print (f"Env_id {env_index} is in the no env ids list, skipping...")
            return
        else:
            print (f"Env_id {env_index} is not in the no env ids list, processing")
        _, current_base_path = get_config_basepath(allTx=allTx, cluster = "tetralith")
        data_postfix = "_noise"
        if noise_sample is not None:
            data_postfix = f"{data_postfix}_noise_{noise_sample}"    

        test_data_iter =["test", "genz", "gendiag"] # will follow up gendiag
        config_file, current_base_path = get_config_basepath(allTx=True, cluster= "tetralith")

        _, train_dt_str = find_latest_train_str(current_base_path, data_postfix, env_index)
        output_datetime_str = train_dt_str 
        for test_data in test_data_iter:
            if env_index > 49 and test_data == "gendiag":
                continue
            else:
                train_feat_tensor, test_feat_tensor, pred_feat_tensor, train_label_tensor, test_label_tensor, result_dir= get_data_dict(env_index, config_file, current_base_path, data_postfix, test_data, return_path = "result_path", prep_baseline = False, cluster= "tetralith")

                # Initialize the model, loss function and optimizer
                input_dim = train_feat_tensor.size(1)
                output_dim = train_label_tensor.size(1)
                # # make tensor into dataset
                test_dataset = TensorDataset(test_feat_tensor, test_label_tensor)
                pred_dataset = TensorDataset(pred_feat_tensor, test_label_tensor)
                
                best_checkpoint = pickle.load(open(result_dir +"/"+ "best_checkpoint.pkl", "rb"))
                best_trial_config = pickle.load(open(result_dir  +"/"+  "best_trial_config.pkl", "rb"))
                best_trained_model = SimpleModel(input_dim, output_dim, best_trial_config["n_layers"])

                with best_checkpoint.as_directory() as checkpoint_dir:
                    tetralith_ckpt_dir = "/proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/ray_results/" + checkpoint_dir[56:]
                    data_path = Path(tetralith_ckpt_dir) / "checkpoint.pt"
                    with open(data_path, "rb") as fp:
                        best_checkpoint_data = r_pickle.load(fp)
                    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
                test_loss, pred_test_diff_loss, pred_loss_sum, list_all_test_loss, list_all_pred_loss_sum, list_all_pred_test_diff_loss, list_all_path_length, list_all_gts, list_all_test_outputs = test_process(best_trained_model, test_dataset, pred_dataset, best_trial_config["batch_size"], nn.L1Loss(reduction = 'none'), return_data = True, return_output = True)

                result_path = f"{current_base_path}/../../../../outputs/huggingface_test_result{data_postfix}/models/env_id_index/env_id_{env_index}/HF_{test_data}_Result{data_postfix}_{output_datetime_str}"
                MLP_result_dict_pickle,_ = find_latest_file(result_path, "MLP_result_dict", test_data)
                KNN_result_dict_pickle,_ = find_latest_file(result_path, "KNN_result_dict", test_data)
                MLP_result_dict = pickle.load(open(f"{result_path}/{MLP_result_dict_pickle}.pkl", "rb"))

                ndarray_all_test_outputs = np.concatenate(list_all_test_outputs)

                batched_MLP_pred_RSSI_list = MLP_result_dict["pred_RSSI"]
                MLP_pred_RSSI = np.concatenate(batched_MLP_pred_RSSI_list)
                KNN_result_dict = pickle.load(open(f"{result_path}/{KNN_result_dict_pickle}.pkl", "rb"))
                KNN_result_dict["KNN_RSSI_loss"] = KNN_result_dict["RSSI_loss"]
                KNN_result_dict_pred_RSSI = KNN_result_dict["pred_RSSI"].reshape(-1)
                MLP_RSSI_pred_test_loss = criterion(torch.tensor(MLP_pred_RSSI), torch.tensor(ndarray_all_test_outputs))
                KNN_RSSI_pred_test_loss = criterion(torch.tensor(KNN_result_dict_pred_RSSI), torch.tensor(ndarray_all_test_outputs))
                
                df = df._append({"MLP_RSSI_pred_test_loss": MLP_RSSI_pred_test_loss.item(), "KNN_RSSI_pred_test_loss": KNN_RSSI_pred_test_loss.item(), "test_data_name": test_data, "env_index": env_index}, ignore_index=True)

    df.to_csv(f"MLP_KNN_RSSI_result_df_{output_datetime_str}.csv", index=False)
if __name__ == "__main__":
    main()