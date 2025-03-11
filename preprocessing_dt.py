import os

import configparser

config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.cfg')

os.environ["WANDB_PROJECT"]=config['WANDB']['WANDB_PROJECT']
os.environ["WANDB_CACHE_DIR"]=config['WANDB']['WANDB_CACHE_DIR']
from datasets import Dataset
import numpy as np
# for plotting
from visualization.get_scene import *
from utilsgyms.utils_winert_dataset import *
from utilsgyms.utils_decisiontrans import *
from utilsgyms.utils_preprocess import *

from datetime import datetime
import warnings


warnings.filterwarnings('ignore', category=UserWarning, message=".*To copy construct from a tensor.*")

def __main__():
    env_index, _, _, Tx_id, allTx, Rx_id, rx_range, add_noise,_, _, _, debug, _, _, _, test_seed, _, data_prefix, data_postfix, Ofunc, gen_func, noise_sample, _,add_surface_index  = parse_input()
    seed_everything(test_seed)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # if test_data:
    #     print ("========== Test Preprocessing config ==========")
    #     assert not add_noise, "add_noise is True"
    os.environ["WANDB_DISABLED"] = "true"
    print ("========== Training config ==========")

    if allTx:
        gen_kwargs_list = prepare_gen_kwargs_list(add_noise, AllTx=allTx, Tx_id = Tx_id, Rx_id=Rx_id, rx_range= rx_range, debug=debug, noise_sample = noise_sample, test_data = False)
    else:
        gen_kwargs_list = prepare_gen_kwargs_list(add_noise, AllTx=False, Tx_id = Tx_id, Rx_id=Rx_id, rx_range= rx_range, debug=debug, noise_sample = noise_sample, test_data = False)
    current_base_path = config['DIRECTORIES']['base_path_test']
    if noise_sample is not None:
        data_postfix = f"{data_postfix}_noise_{noise_sample}"
    if allTx:
        cache_folder = "HFcache/all_Tx_data"
        folder = "processed_data/all_Tx_data"
    else:
        cache_folder = "HFcache/per_Tx_data"
        folder = "processed_data/per_Tx_data"
    # if test_data:
    #     cache_folder = "test" + cache_folder
    #     folder = "test" + folder
    cache_dir = f"{current_base_path}/{cache_folder}/{data_prefix}_HFCache_{data_postfix}_{datetime_str}_env_index_{env_index}/"

    np.random.seed(test_seed)    
    gen_kwargs_list['env_index'] = env_index
    ds = Dataset.from_generator(gen_func, 
                                cache_dir= cache_dir, 
                                keep_in_memory = True,
                                gen_kwargs=gen_kwargs_list,
                                num_proc=1)
    ds.save_to_disk(f"{current_base_path}/{folder}/{data_prefix}_HFds_{data_postfix}__{datetime_str}_env_index_{env_index}/")
if __name__ == "__main__":
    __main__()