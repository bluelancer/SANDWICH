import torch
from transformers import Trainer
from utilsgyms.DecisionTransformerGymDataCollatorTensor import *
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader
from copy import deepcopy
from transformers.utils import is_datasets_available
import datasets
import numpy
import random

class TrainerDT(Trainer):
    # This can not do non-shuffled training: https://discuss.huggingface.co/t/non-shuffle-training/6986
    # think about dataloader without replica: https://github.com/pytorch/pytorch/issues/2052
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # inherit from Trainer
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        self.gen = g_cpu
        
    def get_eval_dataloader(self, eval_dataset):
        raise NotImplementedError("This method is not implemented in this class.")
        test_collator = DecisionTransformerGymDataCollatorTensor(eval_dataset, batch_operations = True, debug = False)
        eval_dataset = self._remove_unused_columns(eval_dataset, description="eval")
        data_loader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": test_collator,
            "num_workers": 4, # self.args.dataloader_num_workers, but we use 4, as recommended by pytorch
            "pin_memory": self.args.dataloader_pin_memory, # same as training
            "persistent_workers": self.args.dataloader_persistent_workers, # same as training
        }
        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            data_loader_params["sampler"] = None #self._get_eval_sampler(eval_dataset) # None
            data_loader_params["drop_last"] = self.args.dataloader_drop_last
            data_loader_params["worker_init_fn"] = seed_worker_trainer # seed_worker
            data_loader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            # Debugg For reproducibility
            # data_loader_params["generator"] = self.gen
            
        return self.accelerator.prepare(DataLoader(eval_dataset, **data_loader_params))
    
    def get_train_dataloader(self) -> DataLoader:
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler() # None
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker_trainer # seed_worker
            # Transformers = "4.42.3" ONLY
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            # DEBUG: for reproducibility
            # dataloader_params["generator"] = self.gen

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

def seed_worker_trainer(worker_id):
    worker_seed = 0
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)