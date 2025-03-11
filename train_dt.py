import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["WANDB_DISABLED"] = "true" # we diable weights and biases logging for this tutorial
os.environ["WANDB_PROJECT"]="wirelessRT"
os.environ["WANDB_CACHE_DIR"]="../wandb_cache"
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import numpy as np
np.random.seed(42)
import random
random.seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()
# for plotting
from visualization.get_scene import *
from visualization.vis_winert import *
from utilsgyms.utils_winert_dataset import *
from utilsgyms.utils_decisiontrans import *
from utilsgyms.utils_preprocess import *
from utilsgyms.TrainableDT import *
from utilsgyms.DecisionTransformerGymDataCollatorTensor import *
from utilsgyms.draw_figure import *
from utilsgyms.TrainerDT import *
from transformers import DecisionTransformerConfig, Trainer, TrainingArguments
from datetime import datetime

# TODO: use Hyperparameter tuning with Ray Tune https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
# TODO: add surface index prediction, Done!
SURFACE_LOSS = False
def __main__():
    env_index, _, _, _, allTx, _, _, add_noise, _, _, _, _, _, wandb, _, test_seed, _, data_prefix, data_postfix, _, _, noise_sample, _,add_type_loss= parse_input()
    seed_everything(test_seed)

    if bool(wandb):
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = None
    print ("========== Training config ==========")
    if noise_sample is not None:
        data_postfix = f"{data_postfix}_noise_{noise_sample}"
    data_postfix = "_noise"

    if noise_sample is not None and noise_sample != 20:
        data_postfix = f"{data_postfix}_noise_{noise_sample}"
    else :
        data_postfix = f"{data_postfix}_{noise_sample}"

    config_file, current_base_path = get_config_basepath(allTx=allTx)
    if not add_noise:
        current_base_path = current_base_path.replace("train", "trainwoAug")
        type_loss = add_type_loss
    else:
        type_loss = True
    add_surface_index = False
    # Training data
    train_ds_path = find_trainpp_repo(current_base_path,data_prefix, data_postfix, env_index) 
    train_ds_withRx = load_data_from_disk(train_ds_path, env_index, add_noise=add_noise, add_surface_index = add_surface_index) 

    collator = DecisionTransformerGymDataCollatorTensor(train_ds_withRx, batch_operations = True, debug = False)
    config = DecisionTransformerConfig(state_dim=collator.state_dim,
                                       act_dim=collator.act_dim,
                                       n_layer = int(config_file['Hyperparameters']['n_layer']),
                                       max_ep_len= int(config_file["DATAParam"]['max_ep_len']),
                                       resid_pdrop= float(config_file['Hyperparameters']['resid_pdrop']),
                                       embd_pdrop = float(config_file['Hyperparameters']['embd_pdrop']),
                                       action_tanh = False, # This have to be False, VERY IMPORTANT
                                       use_cache=True)
    
    model = TrainableDT(config, len_ds = len(train_ds_withRx), surface_id = add_surface_index, type_loss = type_loss)
    print ("========== Model is initialized ==========")
    epoch = int(config_file['RUN_CONFIG']['epochs']) 
    batch_size = int(config_file['RUN_CONFIG']['batch']) 
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    training_args = TrainingArguments(
        data_seed = 42,
        output_dir=f"{current_base_path}/../../workspace/hf_ckpt/{datetime_str}",
        remove_unused_columns=False,
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size, # for debug 
        learning_rate=float(config_file['Hyperparameters']['learning_rate']),
        weight_decay=float(config_file['Hyperparameters']['weight_decay']),
        warmup_ratio=float(config_file['Hyperparameters']['warmup_ratio']),
        logging_steps = float(config_file['Hyperparameters']['logging_steps']),
        optim=str(config_file['Hyperparameters']['optim']),
        max_grad_norm=float(config_file['Hyperparameters']['max_grad_norm']),
        use_cpu = False,
        report_to=report_to,
        dataloader_num_workers=32,        
        seed=test_seed,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    trainer = TrainerDT(
        model=model,
        args=training_args,
        train_dataset=train_ds_withRx,
        data_collator=collator,
    )

    trainer.train()
    print ("========== Model is trained ==========")
    model = model.to("cpu")
    model.eval()
    print ("========== Model is in evaluation mode ==========")
    os.makedirs(f"{config_file['DIRECTORIES']['berzelius_storage_dir']}/../../outputs/huggingface_test_result{data_postfix}/models/huggingface_test_result{data_postfix}_{datetime_str}_{env_index}/", exist_ok=True)
    model.save_pretrained(f"{config_file['DIRECTORIES']['berzelius_storage_dir']}/../../outputs/huggingface_test_result{data_postfix}/models/huggingface_trained_dt{data_postfix}_{datetime_str}_{env_index}/")
    model.save_pretrained(f"{config_file['DIRECTORIES']['berzelius_storage_dir']}/../../outputs/huggingface_test_result{data_postfix}/models/env_id_{env_index}/huggingface_trained_dt{data_postfix}_{datetime_str}_{env_index}/")
    print (f"=== Model is saved, at {config_file['DIRECTORIES']['berzelius_storage_dir']}/../../outputs/huggingface_test_result{data_postfix}/models/huggingface_trained_dt{data_postfix}_{datetime_str}_{env_index}/ ===")

if __name__ == "__main__":
    __main__()

