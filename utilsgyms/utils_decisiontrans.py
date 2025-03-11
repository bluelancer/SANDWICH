import torch
device = torch.device('cpu')

import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm
from utilsgyms.utils_preprocess import *
import re
import json
from datasets import load_from_disk
import configparser
from utilsgyms.TrainableDT import *

# Global variables, for attaching surface index
NODE_ATTR_TENSOR = None
SURFACE_INDEX_TENSOR = None

def parse_input(return_dict = False, ds_task = False):
    config = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_index", type=int, default=1)
    parser.add_argument("--add_type_loss", type=bool, default=True)
    parser.add_argument("--eval_from", type=str, default=None)
    parser.add_argument("--test_seed", type=int, default=42)
    parser.add_argument("--test_data", type=str, default="test")
    
    if not ds_task:
        parser.add_argument("--add_noise", type=bool, default=False)
        parser.add_argument("--Tx", type=int, default=1)
        parser.add_argument("--allTx", type=bool, default=False)
        parser.add_argument("--Rx", type=int, default=None)
        parser.add_argument("--Rx1", type=int, default=None)
        parser.add_argument("--Rx2", type=int, default=None)
        parser.add_argument("--path", type=int, default=6)
        parser.add_argument("--debug_path", type=bool, default=False)
        parser.add_argument("--optim_load", type=bool, default=False)
        parser.add_argument("--small_batch", type=bool, default=False)
        parser.add_argument("--big_batch", type=bool, default=False)
        parser.add_argument("--debug", type=bool, default=False)
        parser.add_argument("--produce_fig", type=bool, default=False)
        parser.add_argument("--wandb", type=bool, default=False)
        parser.add_argument("--replot_baseline_only", type=bool, default=False)
        parser.add_argument("--Ofunc", type=bool, default=False)
        parser.add_argument("--noise_sample", type=int, default=20)

    args = parser.parse_args()
    
    env_index = args.env_index
    add_type_loss = args.add_type_loss
    eval_from = args.eval_from
    test_seed = args.test_seed
    test_data = args.test_data
    if not ds_task:
        debug_path_flag = args.debug_path
        path_id = args.path
        Tx_id = args.Tx
        allTx = args.allTx
        Rx_id = args.Rx
        Rx1_id = args.Rx1
        Rx2_id = args.Rx2
        add_noise = args.add_noise
        optim_load = args.optim_load
        small_batch = args.small_batch
        big_batch = args.big_batch
        debug = args.debug
        produce_fig = args.produce_fig
        wandb = args.wandb
        replot_baseline_only = args.replot_baseline_only
        ofunc = args.Ofunc
        noise_sample = args.noise_sample
        
    config["env_index"] = env_index
    config["add_type_loss"] = add_type_loss
    config["eval_from"] = eval_from
    config["test_seed"] = test_seed
    config["test_data"] = test_data
    if not ds_task:
        config["Tx_id"] = Tx_id
        config["allTx"] = allTx
        config["Rx_id"] = Rx_id
        config["Rx1_id"] = Rx1_id
        config["Rx2_id"] = Rx2_id
        config["debug_path_flag"] = debug_path_flag
        config["path_id"] = path_id
        config["add_noise"] = add_noise
        config["optim_load"] = optim_load
        config["small_batch"] = small_batch
        config["big_batch"] = big_batch
        config["debug"] = debug
        config["produce_fig"] = produce_fig
        config["wandb"] = wandb
        config["replot_baseline_only"] = replot_baseline_only
        config["Ofunc"] = ofunc
        config["noise_sample"] = noise_sample
        
        if Rx1_id is not None or Rx2_id is not None:
            assert Rx_id is None, "Rx_id is not None"
            rx_range = range(Rx1_id, Rx2_id)
        else:
            assert Rx1_id is None, "Rx1_id is not None"
            assert Rx2_id is None, "Rx2_id is not None"
            rx_range = None
        if not debug_path_flag:
            path_id = None
            config["path"] = path_id
    
        if allTx:
            gen_func = gen_data_sample_across_rx
            data_prefix = "allTx"
            config["Tx_id"] = 0
            Tx_id = 0
        elif not ofunc:
            raise NotImplementedError #"new function is not implemented yet"
        else:
            gen_func = gen_data_sample_across_rx
            data_prefix = f"Tx_{Tx_id}"
    
    if ds_task: 
        data_postfix = "_noise"
    elif add_noise: 
        data_postfix = "_noise"
    else:
        data_postfix = ""

    config["data_postfix"] = data_postfix     
    if not ds_task:
        config["data_prefix"] = data_prefix
        config["Ofunc"] = ofunc   
        config["gen_func"] = gen_func
    print ("config = ", config)
    if return_dict:
        return config
    if ds_task:
        return env_index, add_type_loss, test_seed, eval_from, data_postfix, test_data
    else:
        return env_index, debug_path_flag, path_id, Tx_id, allTx, Rx_id, rx_range,  add_noise, optim_load, small_batch, big_batch, debug, produce_fig, wandb, replot_baseline_only,test_seed, eval_from, data_prefix, data_postfix, ofunc, gen_func, noise_sample, test_data, add_type_loss
    
def load_kwargs(path = "/proj/raygnn/workspace/WINeRT/GFlowNet/torchgfn/tutorials/notebooks/best_kwargs.pkl", mode = "normal", get_unnormed_kwags = False, ds_collector = None):
    with open(path, "rb") as f:
        if mode == "cuda":
            best_kwargs = pickle.load(f)
        else:
            best_kwargs = pickle.load(f)
    best_kwargs_keys = best_kwargs.keys()
    if "states" in best_kwargs_keys:
        pass
    else:
        raise ValueError("best_kwargs does not contain states")
    if get_unnormed_kwags:
        if ds_collector is None:
            raise NotImplementedError
        else:
            state_mean = ds_collector.state_mean.astype(np.float32)
            state_std = ds_collector.state_std.astype(np.float32)
            state_mean = torch.from_numpy(state_mean)
            state_std = torch.from_numpy(state_std)
            best_kwargs['states_unorm'] = (best_kwargs['states'].cpu() * state_std.cpu()) + state_mean.cpu()
            print ("re-unnormed states GET!!!")
    return best_kwargs

def prepare_gen_kwargs_list(add_noise,
                            AllTx = False,
                            Tx_id = 1,
                            Rx_id = None,
                            rx_range = None,
                            debug_path_flag = False,
                            path_id = None,
                            local_node = True,
                            debug = False,
                            do_assertion = True,
                            append_receipt_loss = True,
                            noise_sample = 20,
                            test_data = False):
    config = {}
    config["Rx_id"] = [Rx_id] 
    config["rx_range"] = [rx_range]
    config["debug_path"] = [debug_path_flag] 
    config["path"] = [path_id] 
    config["add_noise"] = [add_noise] 
    config["local_node"] = [True] 
    config["append_receipt_loss"] = [append_receipt_loss] 
    config["do_assertion"] = [do_assertion] 
    config["debug"] = [debug] 
    config["noise_sample"] = [noise_sample]
    config ["test_pp"] = [test_data]
    if AllTx:
        Tx_iterator = range(10)
        assert not debug_path_flag, "AllTx is enabled, debug_path_flag should be False" 
        for key in config.keys():
            config[key] = [config[key][0]] * len(Tx_iterator)
    else:
        Tx_iterator = [Tx_id] 
        if debug_path_flag:
            assert path_id is not None, "path_id is None"
            assert Rx_id is not None, "Rx_id is None"
    config["Tx_id"] = Tx_iterator

    gen_kwargs_list = config
    print ("gen_kwargs_list = ", gen_kwargs_list)
    len_Tx = len(gen_kwargs_list["Tx_id"])
    for key in gen_kwargs_list.keys():
        try:
            assert len(gen_kwargs_list[key]) == len_Tx, f"len(gen_kwargs_list[{key}]) = {len(gen_kwargs_list[key])}, len_Tx = {len_Tx}"
        except:
            print (f"key = {key}, gen_kwargs_list[key] = {gen_kwargs_list[key]}")
            raise ValueError
 
    return gen_kwargs_list

def seed_everything(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def find_trainpp_repo(current_base_path,data_prefix, data_postfix, env_index):
    all_pp_repo = os.listdir(current_base_path)
    dt_pp_pattern = rf"^{data_prefix}_HFds_{data_postfix}__\d{{8}}_\d{{6}}_env_index_{env_index}$" 
    # dt_pp_pattern = rf"^{data_prefix}_HFds_{data_postfix}__\d{{8}}_\d{{6}}_obj_{env_index}$"
    matching_repo = []
    for repo in all_pp_repo:
        if re.match(dt_pp_pattern, repo):
        # Extracnt the datetime part
            datetime_match = re.search(r"_\d{8}_\d{6}", repo)
            if datetime_match:
                matching_repo.append(repo)
    if len(matching_repo) == 0:
        raise ValueError("No matching repo found")
    elif len(matching_repo) > 1:
        import ipdb; ipdb.set_trace()
        raise ValueError("Multiple matching repo found")
    else:
        print (f"Found matching repo for training dataset: {matching_repo[0]}")
    train_ds_path = matching_repo[0]
    return_path = f"{current_base_path}/{train_ds_path}"
    return return_path

def load_data_from_disk(train_ds_path, env_index, storage_dir = None, surface_index_path = None, add_noise =False, add_surface_index = False, ):
    train_ds = load_from_disk(train_ds_path)
    train_ds_withRx = train_ds.map(attach_Rxloc)
    print (f"========== loading Data from {train_ds_path} ==========")
    if not bool(add_noise):
        noise_free_data = train_ds_withRx.filter(lambda example: example["reachability"] != "None")
        train_ds_withRx = noise_free_data
    if bool(add_surface_index):
        print ("========== Loading surface index ==========")
        assert storage_dir is not None, "storage_dir is None"
        assert surface_index_path is not None, "surface_index_path is None"
        load_node_attr_tensor(env_index, storage_dir, surface_index_path)
        prepare_surface_index_tensor()
        train_ds_wRx_wSurface = train_ds_withRx.map(attach_SurfaceIndex)
        train_ds_withRx = train_ds_wRx_wSurface
    print ("========== Training Data is loaded ==========")
    return train_ds_withRx


def attach_Rxloc(example):
    rxloc = example['obs'][example['path_length']-1][1:4]# shape = (3,)
    obs_list = example['obs'] # shape = (5, 6)
    obs_ndarray = np.array(obs_list)
    # concatenate the rxloc each row
    obs_ndarray = np.concatenate((obs_ndarray, np.tile(rxloc, (5,1))), axis = 1)
    example['obs'] = obs_ndarray.tolist()
    return example

def attach_SurfaceIndex(example):
    assert SURFACE_INDEX_TENSOR is not None, "SURFACE_INDEX_TENSOR is None"
    info = json.loads(example["infos"][0])
    Tx_id = info["Tx_id"]
    Rx_id = info["Rx_id"]
    path_id = example["path_id"]
    # surface_id_seq_onehot = SURFACE_INDEX_TENSOR[Tx_id, Rx_id, path_id, :, :]
    surface_id_seq_onehot = get_surface_index_tensor(Tx_id, Rx_id, path_id)
    assert surface_id_seq_onehot.shape == (6, 18), f"surface_id_seq_onehot.shape = {surface_id_seq_onehot.shape}"
    obs_np = np.array(example["obs"])
    acts_np = np.array(example["acts"])
    obs_wSurfaceId = np.concatenate((obs_np, surface_id_seq_onehot[1:,...]), axis = 1)
    acts_wSurfaceId = np.concatenate((acts_np, surface_id_seq_onehot[2:,...]), axis = 1)
    example["obs"] = obs_wSurfaceId.tolist()
    example["acts"] = acts_wSurfaceId.tolist()
    return example

def get_surface_index_tensor(Tx, Rx, path):
    return SURFACE_INDEX_TENSOR[Tx, Rx, path, :, :]

def prepare_HP_for_validation(collator):
    max_ep_len = 4
    device = "cpu" # use cpu for validation
    scale = 1000
    TARGET_RETURN = 13.815510 / scale  # evaluation is conditioned on a return of 1000, scaled accordingly
    state_mean = collator.state_mean.astype(np.float32)
    state_std = collator.state_std.astype(np.float32)
    state_dim = collator.state_dim
    act_dim = collator.act_dim
    return max_ep_len, device, scale, TARGET_RETURN, state_mean, state_std, state_dim, act_dim

def load_node_attr_tensor(env_index, 
                          storage_dir,
                          surface_index_path,
                          debug = True):
    train_env = WINeRTEnv(
        storage_dir = storage_dir,
        surface_index_path = surface_index_path,
        env_index = env_index,
        debug= debug)
    global NODE_ATTR_TENSOR 
    NODE_ATTR_TENSOR = train_env.node_attr_tensor
    print ("NODE_ATTR_TENSOR is loaded")
    return

def prepare_surface_index_tensor():
    assert NODE_ATTR_TENSOR is not None, "NODE_ATTR_TENSOR is None"
    surface_id_seq = torch.tensor(NODE_ATTR_TENSOR[:, :, :, :, 6]).type(torch.int64)
    surface_id_seq_non_neg = surface_id_seq + 3
    surface_id_seq_onehot = torch.nn.functional.one_hot(surface_id_seq_non_neg, num_classes=18).cpu().numpy()
    assert surface_id_seq_onehot.shape[-3:] == (30, 6, 18), f"surface_id_seq_onehot.shape = {surface_id_seq_onehot.shape}"
    global SURFACE_INDEX_TENSOR
    SURFACE_INDEX_TENSOR = surface_id_seq_onehot
    return

def get_config_basepath(config_file_name = 'config.cfg',
                        allTx = False, 
                        cluster = "berzelius"):
    if allTx:
        folder = "all_Tx_data"
    else:
        folder = "per_Tx_data"
        raise NotImplementedError
    config_file = configparser.ConfigParser()
    config_file._interpolation = configparser.ExtendedInterpolation()
    config_file.read(config_file_name)
    if cluster == "berzelius":
        current_base_path = f"{config_file['DIRECTORIES']['berzelius_storage_dir']}/train/{folder}"
    elif cluster == "tetralith":
        current_base_path = f"{config_file['DIRECTORIES']['tetralith_train_storage_dir']}"
    else:
        raise ValueError("cluster is not recognized")
    return config_file, current_base_path
    
# THis is not used as one cannnot design the metrics for the trajectory prediction, in regard to DT
def test_metrics(pred, add_surface_index = False):
    # 1, pred EvalPrediction: https://huggingface.co/docs/transformers/en/internal/trainer_utils#transformers.EvalPrediction
    # labels = pred.label_ids
    # predictions = pred.predictions
    # seem not these 2 
    # 2, Seems a bug: https://github.com/huggingface/peft/issues/577
    # 3, OK there seems to be a bug for token generation for trainer: https://github.com/huggingface/trl/issues/862
    # 3, related: https://github.com/huggingface/transformers/issues/31462
    action_preds = pred.label_ids 
    attention_mask = pred.inputs["attention_mask"]
    action_targets = pred.inputs["actions"]
    states = pred.inputs["states"] # This is identified same as train state
    
    act_dim = action_preds.shape[2]
    action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    
    type_loss = torch.mean(TrainableDT.type_loss_func(action_preds[:,:6], action_targets[:,:6])) # first 6 columns are type, one hotted
    Angular_loss = torch.mean(abs(action_preds[:,6:8] - action_targets[:,6:8])) # last 2 columns are azimuth and radian
    log_ang_loss = torch.log(Angular_loss +1e-5) # if Angular_loss < 1e-5, we consider type loss shoud be dominant
    loss = type_loss + log_ang_loss
    
    if add_surface_index:
        surface_loss = torch.mean(TrainableDT.surface_loss_func(action_preds[:,8:], action_targets[:,8:])) 
        loss = loss + surface_loss
        
    return {"test_loss": loss,
            "test_Angular_loss": Angular_loss,
            "test_type_loss": type_loss,
            "test_log_ang_loss": log_ang_loss,
            "test_original_loss": type_loss + log_ang_loss}