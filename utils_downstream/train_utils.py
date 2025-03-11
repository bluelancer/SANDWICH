import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from scipy.stats import gaussian_kde
from scipy.special import rel_entr
from scipy.stats import entropy
import re
import pickle

def draw_multiple_bars_same_plot_ray_ds(result_dict,y_key_list, plot_type,base_dir,datetime_str,env_id = None, load_from_file = False, test_data = "test", draw_baseline = False):
    # Assert that the keys in the result_dict are the same as the keys in the key_list
    assert test_data in ["test", "genz", "gendiag"]
    os.makedirs(base_dir, exist_ok=True)
    # Step 1: Load the old result_dict
    if load_from_file:
        dt_RSSI_result_dict = load_dt_RSSI_dict(base_dir)
        dt_angle_reward_result_dict = load_dt_angle_reward_dict(base_dir)
    else:
        dt_RSSI_result_dict = None
        dt_angle_reward_result_dict = None
        # TODO: Add the logic to draw the boxplot for the path_length
    for x_bar in plot_type:
        # the x_bar is either RSSI or path_length, while y_key is different from dt, in baseline we have two metrics
        combined_df = prep_init_RSSI_df(dt_RSSI_result_dict = dt_RSSI_result_dict, y_key_list = ['test_loss', 'pred_loss'], x_key= x_bar, use_dt_RSSI = load_from_file)
        for y_key in y_key_list:
            for x_value, y_value in zip(result_dict[x_bar], result_dict[y_key]):

                temp_df = pd.DataFrame({
                    x_bar: x_value,
                    "MAE": y_value,
                    "metric": y_key,
                })
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        if x_bar.endswith("path_length"):
            # draw RSSI loss against path_length
            combined_df["path_length"] = combined_df[x_bar].astype(int)
            draw_boxplot_ray_ds(combined_df[combined_df["path_length"].astype(int) > 1], base_dir, datetime_str, env_id = env_id, x_label = "path_length", y_label = "MAE", hue_type = "metric", test_data = test_data, draw_baseline = draw_baseline)
        elif x_bar.endswith("RSSI"): 
            # draw error divergence against RSSI
            draw_2D_sns_kde_plot(combined_df, base_dir, datetime_str, env_id = env_id, x_label = x_bar, y_label = "MAE", test_data = test_data, draw_baseline = draw_baseline)
    return
    
    
def load_dt_angle_reward_dict(base_dir):
    # find allTx_all_rewards_dict_20240729-233044.pkl
    search_pickle = rf"allTx_all_rewards_dict_\d{{8}}-\d{{6}}.pkl"
    list_of_files = os.listdir(f"{base_dir}/../")
    found_file = False
    for file in list_of_files:
        reMatch = re.match(search_pickle, file)
        if reMatch is not None:
            old_angle_reward_result_dict = pickle.load(open(f"{base_dir}/../{file}", "rb"))
            found_file = True
            break
    if not found_file:
        raise ValueError(f"Cannot find the old angle reward result dict file in {base_dir}/../")  
    return old_angle_reward_result_dict

def load_dt_RSSI_dict(base_dir):
    old_RSSI_result_dict = pickle.load(open(f"{base_dir}/../result_dict.pkl", "rb"))
    return old_RSSI_result_dict

def prep_init_RSSI_df(dt_RSSI_result_dict = None, y_key_list = None, x_key = None, use_dt_RSSI = True):
    combined_df = pd.DataFrame() 
    if use_dt_RSSI:
        assert dt_RSSI_result_dict is not None
        for y_key in y_key_list:
            for x_value, y_value in zip(dt_RSSI_result_dict[x_key], dt_RSSI_result_dict[y_key]):
                temp_df = pd.DataFrame({
                    x_key: x_value,
                    "MAE": y_value,
                    "metric": y_key,
                })
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
    return combined_df
def draw_baseline_reward_angle_plot(result_dict, base_dir, datetime_str, env_id = None, load_from_file = False):
    pass

def draw_boxplot_ray_ds(dataframe, base_dir, datetime_str, env_id = None, x_label = "path_length", y_label = "MAE", hue_type = "metric", test_data = "test", draw_baseline = False):
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataframe, x=x_label, y=y_label, hue=hue_type, showfliers = False)
    # show labels
    plt.legend(loc='upper left')
    plt.title(f"{y_label} by {x_label} distribution plot (Env ID: {env_id} on test_data: {test_data})")
    plt.xlabel(f"{x_label} (#Hops)")
    plt.ylabel(f"{y_label} (MAE)")
    # Save and display the plot
    filename = f"{x_label}_against_{y_label}_barplot_{datetime_str}.png"
    plt.savefig(f"{base_dir}/{filename}")
    plt.show()
    return
    

def draw_2D_sns_kde_plot(dataframe, base_dir, datetime_str, env_id = None, x_label = "RSSI", y_label = "MAE", hue_type = "metric", test_data = "test", draw_baseline = False):
    # Plotting
    plt.figure(figsize=(10, 6))

    dataframe[x_label] = dataframe[x_label].astype(float) 
    sampled_df = dataframe.sample(frac=0.1, replace=True, random_state=1)
    sampled_df_wo_diff = sampled_df[sampled_df["metric"] != "pred_test_diff"]
    # Save and display the plot
    kde_pred = gaussian_kde(sampled_df_wo_diff[sampled_df_wo_diff["metric"] == "pred_loss"][[x_label, y_label]].T, bw_method=0.1)
    kde_test = gaussian_kde(sampled_df_wo_diff[sampled_df_wo_diff["metric"] == "test_loss"][[x_label, y_label]].T, bw_method=0.1)
    ### compute the KL divergence between the two distributions ####
    
    # Create a grid of points
    x = np.linspace(-70, -20, 100)
    y = np.linspace(-2.0, 8.0, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.ravel(), Y.ravel()])
    
    # Evaluate the KDEs on the grid points
    pdf_pred = kde_pred(grid_points)
    pdf_test = kde_test(grid_points)

    # Compute the KL divergence
    kl_div_test_pred = entropy(pdf_test, pdf_pred)

    # show labels
    sampled_df_wo_diff.loc[sampled_df_wo_diff[hue_type] == "pred_loss", hue_type] = f"SANDWICH (KL div: {kl_div_test_pred:.4f})"
    sampled_df_wo_diff.loc[sampled_df_wo_diff[hue_type] == "test_loss", hue_type] = "GCP (Baseline)"
    if draw_baseline:
        kde_mlp = gaussian_kde(sampled_df_wo_diff[sampled_df_wo_diff["metric"] == "MLP_RSSI_loss"][[x_label, y_label]].T, bw_method=0.1)
        pdf_mlp = kde_mlp(grid_points)
        kl_div_test_mlp = entropy(pdf_test, pdf_mlp)    
        sampled_df_wo_diff.loc[sampled_df_wo_diff[hue_type] == "MLP_RSSI_loss", hue_type] = f"MLP_baseline (KL div: {kl_div_test_mlp:.4f})"
    sns.kdeplot(data = sampled_df_wo_diff, x = x_label, y = y_label, hue = hue_type, clip=((-70,10),(-4.0, 12.0)), levels=5, thresh=.1,)
    plt.title(f"{y_label} by {x_label} distribution plot (Env ID: {env_id}, Test data: {test_data})")
    plt.xlabel(f"{x_label} (dBm)")
    plt.ylabel(f"{y_label} (MAE)")
    filename = f"{x_label}_against_{y_label}_density_{datetime_str}.png"
    plt.savefig(f"{base_dir}/{filename}")
    plt.show()
    return

def kl_divergence(p, q):
    return np.sum(rel_entr(p, q))