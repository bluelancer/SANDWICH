import pickle
from utilsgyms.draw_figure import *
from utils_downstream.data_utils import *
from utils_downstream.train_utils import *
def find_latest_file(loading_path, file_pattern, test_data):

    all_files = os.listdir(loading_path)
    if file_pattern in ["all_log_ave_ang_loss_dict", "allTx_all_path_length_dict"]:
        searching_pattern = f"{file_pattern}_\d{{8}}-\d{{6}}.pkl"
    elif file_pattern in ["MLP_result_dict"]: # To be implemented KNN_result_dict
        # MLP_result_dict_test_2024-08-15_02-37-46.pkl
        searching_pattern = f"{file_pattern}_{test_data}_\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}.pkl"
    else:
        raise ValueError(f"file_pattern {file_pattern} is not found in {['all_log_ave_ang_loss_dict', 'allTx_all_path_length_dict', 'MLP_result_dict']}")
    data_matches = []
    for file in all_files:
        if re.match(searching_pattern, file):
            if file_pattern in ["all_log_ave_ang_loss_dict", "allTx_all_path_length_dict"]:
                data_match = file.split("_")[-1].split(".")[0]
            elif file_pattern in ["MLP_result_dict"]:
                data_match_part1 = file.split("_")[-2].split(".")[0]
                data_match_part2 = file.split("_")[-1].split(".")[0]
                data_match = f"{data_match_part1}_{data_match_part2}"
            data_matches.append(data_match)
    if len(data_matches) == 0:
        raise ValueError("No model is found")
    elif len(data_matches) == 1:
        data_match = data_matches[0]
    else:
        print (f"multiple models are found, using the latest one, {max(data_matches)}")
        data_match = max(data_matches)
    print (f"Loading {file_pattern} from {data_match}")
    if file_pattern in ["all_log_ave_ang_loss_dict", "allTx_all_path_length_dict"]:
        file_name = f"{file_pattern}_{data_match}"
    elif file_pattern in ["MLP_result_dict"]:
        file_name = f"{file_pattern}_{test_data}_{data_match}"
    return file_name, data_match
NO_ENV_IDS = [9,10,13,18,29,31,32,33,43,45,47,48,49,59,64,67,71,72,77,88,90,92,93,94,99]
def main():
    env_index, _, _, _, _, _, _,  _, _, _, _, _, _, _, _,_, _, _, data_postfix, _, _, noise_sample, _, add_type_loss = parse_input()
    for test_data in ["test", "genz","gendiag"]:
        _, current_base_path = get_config_basepath(allTx=True, cluster = "berzelius")
        data_postfix = "_noise"
        data_postfix_dt = f"{data_postfix}_20"
        if noise_sample is not None:
            data_postfix_sandwich = f"{data_postfix}_noise_{noise_sample}" 
        data_postfix_DTwType = data_postfix_dt + "_wType"
        
        _, train_dt_str_sandwich = find_latest_train_str(current_base_path, data_postfix_sandwich, env_index) 
        _, train_dt_str_dt = find_latest_train_str(current_base_path, data_postfix_dt, env_index)
        _, train_dt_str_DTwType = find_latest_train_str(current_base_path, data_postfix_DTwType, env_index)
        
        result_path_SANDWICH = f"{current_base_path}/../../../../outputs/huggingface_test_result{data_postfix_sandwich}/models/env_id_index/env_id_{env_index}/HF_{test_data}_Result{data_postfix_sandwich}_{train_dt_str_sandwich}"
        result_path_DT = f"{current_base_path}/../../../../outputs/huggingface_test_result{data_postfix_dt}/models/env_id_index/env_id_{env_index}/HF_{test_data}_Result{data_postfix_dt}_{train_dt_str_dt}"
        result_path_DTwType = f"{current_base_path}/../../../../outputs/huggingface_test_result{data_postfix_DTwType}/models/env_id_index/env_id_{env_index}/HF_{test_data}_Result{data_postfix_DTwType}_{train_dt_str_DTwType}"
        
        all_log_ave_ang_loss_dict_pickle_sandwich,data_match_ave_ang_sandwich = find_latest_file(result_path_SANDWICH, "all_log_ave_ang_loss_dict", test_data)
        all_log_ave_ang_loss_dict_pickle_dt,data_match_ave_ang_dt = find_latest_file(result_path_DT, "all_log_ave_ang_loss_dict", test_data)
        all_log_ave_ang_loss_dict_pickle_DTwType,data_match_ave_ang_DTwType = find_latest_file(result_path_DTwType, "all_log_ave_ang_loss_dict", test_data)
        
        allTx_all_path_length_dict_pickle_sandwich,_ = find_latest_file(result_path_SANDWICH, "allTx_all_path_length_dict", test_data)
        allTx_all_path_length_dict_pickle_dt,_ = find_latest_file(result_path_DT, "allTx_all_path_length_dict", test_data)
        allTx_all_path_length_dict_pickle_DTwType,_ = find_latest_file(result_path_DTwType, "allTx_all_path_length_dict", test_data)
        
        all_log_ave_ang_loss_dict_sandwich = pickle.load(open(f"{result_path_SANDWICH}/{all_log_ave_ang_loss_dict_pickle_sandwich}.pkl", "rb"))
        all_log_ave_ang_loss_dict_dt = pickle.load(open(f"{result_path_DT}/{all_log_ave_ang_loss_dict_pickle_dt}.pkl", "rb"))
        all_log_ave_ang_loss_dict_DTwType = pickle.load(open(f"{result_path_DTwType}/{all_log_ave_ang_loss_dict_pickle_DTwType}.pkl", "rb"))
        
        allTx_all_path_length_dict_pickle_sandwich, _ = find_latest_file(result_path_SANDWICH, "allTx_all_path_length_dict", test_data)
        allTx_all_path_length_dict_pickle_dt, _ = find_latest_file(result_path_DT, "allTx_all_path_length_dict", test_data)
        allTx_all_path_length_dict_pickle_DTwType, _ = find_latest_file(result_path_DTwType, "allTx_all_path_length_dict", test_data)
        
        all_path_length_dict_sandwich = pickle.load(open(f"{result_path_SANDWICH}/{allTx_all_path_length_dict_pickle_sandwich}.pkl", "rb"))
        all_path_length_dict_dt = pickle.load(open(f"{result_path_DT}/{allTx_all_path_length_dict_pickle_dt}.pkl", "rb"))
        all_path_length_dict_DTwType = pickle.load(open(f"{result_path_DTwType}/{allTx_all_path_length_dict_pickle_DTwType}.pkl", "rb"))
        
        draw_multiple_bars_same_plot([all_log_ave_ang_loss_dict_sandwich, all_log_ave_ang_loss_dict_dt , all_log_ave_ang_loss_dict_DTwType],
                                    [all_path_length_dict_sandwich, all_path_length_dict_dt,all_path_length_dict_DTwType],
                                    data_match_ave_ang_sandwich,
                                    result_path_DT,
                                    trival_sample = True,
                                    test_data = test_data)
if __name__ == "__main__":
    main()