from torch.utils.data import Dataset
# from torch.utils.data import ConcatDataset

import h5py
import os
import sys
import numpy as np

def concatenate(file_names_to_concatenate, entry_key = 'data', type = "train" ):
    # Concatenate multiple hdf5 files into one
    vsources = []
    total_length = 0
    layout_dict = {}
    for i, filename in enumerate(file_names_to_concatenate):
        with h5py.File(filename, 'r') as f:
            sh = f[entry_key].shape
        print ("parsing file: ", filename, ", entry_key: ", entry_key, ", shape: ", sh)
        vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
        vsources.append(vsource)
        total_length += vsource.shape[0]
        
    layout = h5py.VirtualLayout(shape=(total_length,) + sh[1:],
                                dtype=np.float64)
    offset = 0
    for vsource in vsources:
        length = vsource.shape[0]
        layout[offset:offset+length] = vsource
        offset += length
    
    with h5py.File(type + "_"+ entry_key + ".h5", 'w', libver='latest') as f:
        f.create_virtual_dataset(entry_key, layout, fillvalue=0)
    
class WINeRT_Dataset(Dataset):
    # cite: https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
    def __init__(self, hdf5_repo_path, type = "train"):
        """do not open hdf5 here!!"""
        # Search for hdf5 files in the repo
        self.hdf5_repo_path = hdf5_repo_path
        self.hdf5_dir_list = [os.path.join(self.hdf5_repo_path, file) for file in os.listdir(self.hdf5_repo_path) if file.endswith(".h5")]
        
        self.train_hdf5_dir_list = []
        self.test_hdf5_dir_list = []
        
        for path in self.hdf5_dir_list:
            if path.endswith(("train-dataset_0.h5", "train-dataset_1.h5")):
                self.train_hdf5_dir_list.append(path)
            else:
                self.test_hdf5_dir_list.append(path)
        # Assign dataset type
        self.type = type
        # Assign dataset length to be called by __len__()
        self.length = 0
        if self.type == "train":
            for file in self.train_hdf5_dir_list:
                with h5py.File(name=file, mode='r') as f:
                    self.length +=len(f['floor_idx'])
        elif self.type == "test":
            for file in self.test_hdf5_dir_list:
                with h5py.File(name=file, mode='r') as f:
                    self.length +=len(f['floor_idx'])
        else:
            raise ValueError("hdf5 file not found")
        # reorganize the hdf5 files
        self.keys = ['channels','tx','rx','floor_idx','interactions']
        for key in self.keys:
            if self.type == "train":
                concatenate(self.train_hdf5_dir_list, entry_key = key, type = self.type)
            elif self.type == "test":
                concatenate(self.test_hdf5_dir_list, entry_key = key, type = self.type)
            else:
                raise ValueError("hdf5 file not found")

    def open_hdf5(self):
        assert len(self.hdf5_dir_list) > 0, "no hdf5 file found"
        self.dataset_dict = {}
        for key in self.keys:
            hdf5_path = self.type + "_" + key + ".h5"
            self.dataset_dict[key] = h5py.File(hdf5_path, 'r')

    def __getitem__(self, item: int):
        # TODO: This is not efficient, need to be improved, around 30s per item
        if not hasattr(self, 'dataset_dict'):
            self.open_hdf5()
        # print ("file")
        x = {}
        for key in self.dataset_dict.keys():
            if key == 'interactions':
                continue
            x[key] = np.array(self.dataset_dict[key][key][item])
        y = np.array(self.dataset_dict['interactions']['interactions'][item])
        return x, y
    
    def __del__(self):
        if hasattr(self, 'dataset_dict'):
            for key in self.dataset_dict.keys():
                self.dataset_dict[key].close()
            
    def __len__(self):
        return self.length
    
# Example ussage:    
# train_dataset = WINeRT_Dataset("raytracingdata/wi3rooms/", type = "train")
# train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=1,
#         num_workers=1
#     )
# for i, (input, target) in enumerate(train_loader):
#     print(input.keys())
#     print(target.shape)
#     break