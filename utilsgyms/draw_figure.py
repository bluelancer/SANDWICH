from utilsgyms.utils_decisiontrans import *
from utilsgyms.DecisionTransformerGymDataCollatorSimple import *
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from pypalettes import get_hex

def draw_per_Rx_reward_by_pathlength(reward_list,
         path_length_list,
         path_id_list,
         datetime_str,
         Tx_id = 1,
         rx_id = None,
         add_noise = False,
         base_dir = "/proj/raygnn_storage/huggingface_test_result/"):
    # Let's plot the rewards on a bar chart, show axis labels, and save the plot
    plt.figure()
    plt.bar(reward_list, path_id_list, label='Rewards')
    plt.bar(path_length_list, path_id_list, label='Path Length')
    plt.xlabel("Path ID")
    plt.ylabel("Reward/Path Length")
    plt.legend()
    plt.show()
    assert rx_id is not None, "Please provide the Rx ID"
    if add_noise:
        filename = f"Tx_{Tx_id}_Rx_{rx_id}_noise_rewards_dict_{datetime_str}.png"
    else:
        filename = f"Tx_{Tx_id}_Rx_{rx_id}_rewards_dict_{datetime_str}.png"
    plt.savefig(base_dir + filename)
    return

def draw_boxplot(all_rewards_dict,
                 all_path_length_dict, 
                 datetime_str,
                 base_dir,
                 Tx_id =1):
    reward_by_path_length = {}
    for key in all_rewards_dict.keys():
        for path_id in all_rewards_dict[key].keys():
            length = all_path_length_dict[key][path_id]
            if length not in reward_by_path_length.keys():
                reward_by_path_length[length] = []
            reward_by_path_length[length].append(all_rewards_dict[key][path_id])
    
    pickle.dump(reward_by_path_length, open(f"{base_dir}/reward_by_path_length_{datetime_str}.pkl", "wb"))
    # do a boxplot for the rewards by path length      
    path_lengths = []
    rewards = []
    for key, values in reward_by_path_length.items():
        path_length = int(key.item())
        for value in values:
            path_lengths.append(path_length)
            rewards.append(value)
    # Create DataFrame
    df = pd.DataFrame({
        "Path Length": path_lengths,
        "Reward": rewards
    })
    plt.figure()
    sns.boxplot(data=df,x="Path Length", y="Reward")
    filename = f"Tx_{Tx_id}_rewards_boxplot_{datetime_str}.png"
    plt.savefig(base_dir + filename)
    return

def draw_multiple_bars_same_plot(list_all_rewards_dict,
                                 list_all_path_length_dict,
                                 datetime_str,
                                 base_dir,
                                 trival_sample = False,
                                 test_data = "test", append_baseline = False, baseline_reward = None, baseline_path_length = None):
    assert test_data in ["test", "genz","gendiag"], "The test data must be either 'test' or 'genz'"
    if trival_sample:
        base_dict1 = {
                "Path Length": 1,
                "-log(Angular Loss)": -np.log(1e-6),
                "Data Augmentation": "SANDWICH",
            }
        base_dict2 = {
            "Path Length": 1,
            "-log(Angular Loss)": -np.log(1e-6),
            "Data Augmentation": " Vanilla Decision Transformer",
        }
        base_dict3 = {
            "Path Length": 1,
            "-log(Angular Loss)": -np.log(1e-6),
            "Data Augmentation": "Decision Transformer with State Supervision",
        }
        if len(list_all_rewards_dict) == 1:  
            list_of_dicts = [base_dict1] * 1711 * 15
        elif len(list_all_rewards_dict) == 2:
            list_of_dicts = [base_dict1, base_dict2] * 1711 * 15
        elif len(list_all_rewards_dict) == 3:
            list_of_dicts = [base_dict1, base_dict2, base_dict3] * 1711 * 15
        else :
            raise ValueError("The number of models must be 1, 2, or 3")
            
        combined_df = pd.DataFrame(list_of_dicts)
    else:
        combined_df = pd.DataFrame()

    # Iterate over each set of reward and path length dictionaries
    for index, (all_rewards_dict, all_path_length_dict) in enumerate(zip(list_all_rewards_dict, list_all_path_length_dict)):
        reward_by_path_length = {}  # Reset this dictionary for each transaction
        for key in all_rewards_dict.keys():
            for path_id in all_rewards_dict[key].keys():
                length = all_path_length_dict[key][path_id]
                if length not in reward_by_path_length:
                    reward_by_path_length[length] = []
                reward_by_path_length[length].append(all_rewards_dict[key][path_id])
                
        # Prepare DataFrame for this specific transaction
        path_lengths = []
        rewards = []
        if append_baseline:
            assert baseline_reward is not None, "Please provide the baseline reward"
            assert baseline_path_length is not None, "Please provide the baseline path length"
            baseline_reward_list = list(baseline_reward)
            baseline_path_length_list = list(baseline_path_length)
            path_lengths.extend(baseline_path_length_list)
            rewards.extend(baseline_reward_list)

        for length, values in reward_by_path_length.items():
            path_lengths.extend([length] * len(values))
            rewards.extend(values)
        if index == 0:
            temp_df = pd.DataFrame({
                "Path Length": path_lengths,
                "-log(Angular Loss)": rewards,
                "Data Augmentation": "SANDWICH", # if index == 0   #se " Vanilla Decision Transformer",
            })
        elif index == 1:
            temp_df = pd.DataFrame({
                "Path Length": path_lengths,
                "-log(Angular Loss)": rewards,
                "Data Augmentation": " Vanilla Decision Transformer", # if index == 0   #se " Vanilla Decision Transformer",
            })
        elif index == 2:
            temp_df = pd.DataFrame({
                "Path Length": path_lengths,
                "-log(Angular Loss)": rewards,
                "Data Augmentation": "Decision Transformer with State Supervision", # if index == 0   #se " Vanilla Decision Transformer",
            })
        else:
            raise ValueError("The number of models must be 1, 2, or 3")
        combined_df = pd.concat([combined_df, temp_df[temp_df["Path Length"] > 1]], ignore_index=True)
    # Ensure that the 'Path Length' and 'Transaction ID' columns are correctly typed for plotting
    combined_df['Path Length'] = combined_df['Path Length'].astype(str)
    combined_df['Data Augmentation'] = combined_df['Data Augmentation'].astype('category')

    # Plotting
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))
    palette = get_hex("Acadia", keep_first_n=3)
    sns.boxplot(data=combined_df[combined_df["Path Length"].astype(int) > 1], x="Path Length", y="-log(Angular Loss)", hue="Data Augmentation",showfliers = False, palette=palette)
    ## add a constant value as a reference line
    # draw average of the baseline
    ave_rewards_SANDWICH = combined_df[combined_df["Data Augmentation"] == "SANDWICH"]["-log(Angular Loss)"].mean()
    ave_reward_dt = combined_df[combined_df["Data Augmentation"] == " Vanilla Decision Transformer"]["-log(Angular Loss)"].mean()
    ave_reward_DTwType = combined_df[combined_df["Data Augmentation"] == "Decision Transformer with State Supervision"]["-log(Angular Loss)"].mean()
    print (f"Average of the rewards with SANDWICH: {ave_rewards_SANDWICH}")
    print (f"Average of the rewards with Vanilla Decision Transformer: {ave_reward_dt}")
    plt.axhline(y = ave_rewards_SANDWICH, color='green', linestyle='solid', label='SANDWICH ave. thru paths')
    plt.axhline(y = ave_reward_dt, color='blue', linestyle='dotted', label='Vanilla Decision Transformer ave. thru paths')
    plt.axhline(y = ave_reward_DTwType, color='orange', linestyle='dashed', label='Decision Transformer w. State Supervision ave. thru paths')
    if test_data == "test":
        plt.axhline(y = -np.log(0.087), color='r', linestyle='dashdot', label='WINERT BaseLine')
        # plt.axhline(y = -np.log(0.330), color='purple', linestyle='--', label='ICLR 2023 BaseLine: MLP')
        # plt.axhline(y = -np.log(0.212), color='black', linestyle='--', label='ICLR 2023 BaseLine: kNN')
    elif test_data == "genz":
        plt.axhline(y = -np.log(0.084), color='r', linestyle='dashdot', label='ICLR 2023 BaseLine: WINERT')
        plt.axhline(y = -np.log(0.350), color='purple', linestyle='--', label='ICLR 2023 BaseLine: MLP')
        plt.axhline(y = -np.log(0.226), color='black', linestyle='--', label='ICLR 2023 BaseLine: kNN')
    else:
        plt.axhline(y = -np.log(0.084), color='r', linestyle='--', label='ICLR 2023 BaseLine: WINERT')
        plt.axhline(y = -np.log(0.350), color='purple', linestyle='--', label='ICLR 2023 BaseLine: MLP')
        plt.axhline(y = -np.log(0.226), color='black', linestyle='--', label='ICLR 2023 BaseLine: kNN')
    # show labels
    plt.legend(loc='upper right')
    plt.title(f"-log(Angular Loss) by Path Length Across {test_data} Data")
    plt.xlabel("Path Length")
    plt.ylabel("-log(Angular Loss)")
    
    # Save and display the plot
    # filename = f"{test_data}_rewards_barplot_{datetime_str}.png"
    # plt.savefig(f"{base_dir}/{filename}")
    # plt.show()
    with PdfPages(f"{base_dir}/{test_data}_rewards_barplot_{datetime_str}.pdf") as pdf:
        pdf.savefig(bbox_inches='tight') 
    return