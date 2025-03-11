# import gym
import os
from dataclasses import dataclass
import numpy as np
import torch
from datetime import datetime 


@dataclass
class DecisionTransformerGymDataCollatorSimple:
    return_tensors: str = "pt"
    max_len: int = 20 # subsets of the episode we use for training
    state_dim: int = 6  # size of state space, should be 6 
    act_dim: int = 3  # size of action space, should be 3
    max_ep_len: int = 4 # max episode length in the dataset
    scale: float = 1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, dataset) -> None:
        self.act_dim =  len(dataset[0]["acts"][0]) #len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["obs"][0]) #len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        for obs in dataset["obs"]:
            states.extend(obs)
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)
        
    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features, batch_operations=False):
        start_time = datetime.now() 
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        ### TODO: Tensorize this loop        
        import ipdb; ipdb.set_trace()
        for ind in batch_inds:
            feature = self.dataset[int(ind)]
            
            # get sequences from dataset
            s.append(np.array(feature["obs"]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature["acts"]).reshape(1, -1, self.act_dim))
            r.append(np.array(np.array(feature['rews']).reshape(1, -1, 1)))    
            d.append(np.array(feature["terminal"]).reshape(1, -1))
            
            # I am using a home made padding cutoff
            timesteps.append(np.arange(1, s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = 0  # padding cutoff, home made
            rtg.append(
                self._discount_cumsum(np.array(feature["rews"]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            
            if rtg[-1].shape[1] < s[-1].shape[1]:
                import ipdb; ipdb.set_trace()
                # Let try to append the zeros to the header of the array
                # Given: timesteps.shape = torch.Size([64, 19]), a.shape = torch.Size([64, 19, 3]), r.shape = torch.Size([64, 19, 1]), d.shape = torch.Size([64, 19]), rtg.shape = torch.Size([64, 20, 1]), s.shape = torch.Size([64, 20, 6])
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
                timesteps[-1] = np.concatenate([np.zeros((1, 1)), timesteps[-1]], axis=1)
                # a[-1].shape[1] should be 4, s[-1].shape[1] should be 5
                a[-1] = np.concatenate([a[-1], np.ones((1, 1, self.act_dim)) * -10], axis=1)
                r[-1] = np.concatenate([r[-1],np.zeros((1, 1, 1))], axis=1)
                d[-1] = np.concatenate([np.ones((1, 1)) * 2, d[-1]], axis=1)
                
            # if a[-1].shape[1]< timesteps[-1].shape[1]:
            #     raise NotImplementedError
            
            # padding and state + reward normalization
            tlen = s[-1].shape[1]

            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            # if a[-1].shape[1] == 3:
            #     raise NotImplementedError
                
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            # timesteps[-1] = np.concatenate([10 * np.ones((1, self.max_len - tlen)), timesteps[-1]], axis=1) 
            # Ok so we use -10 to pad the timesteps, 
            # -1 as init, 0 as padding cutoff, 
            # THIS IS NOT CORRECT! for https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            # 1 as start of episode, 2 as end of episode, 3 as end of trajectory
            mask.append(
                np.concatenate([
                    np.zeros((1, self.max_len - tlen)),
                    np.ones((1, tlen -1)),
                    np.zeros((1, 1)),
                    ], 
                               axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        if a.shape[0] == timesteps.shape[0] + 1:
            timesteps = np.concatenate([np.zeros((1, 1)), timesteps], axis=1)
            assert a.shape[0] == timesteps.shape[0]
            assert timesteps[-1] == 3

        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()
        # if ind == batch_inds[0]:
        #     print (f"states = {s} at index {ind}")
        #     print (f"actions = {a} at index {ind}")
        #     print (f"rewards = {r} at index {ind}")
        #     print (f"returns_to_go = {rtg} at index {ind}")
        #     print (f"timesteps = {timesteps} at index {ind}")
        #     print (f"attention_mask = {mask} at indeADx {ind}")
        end_time = datetime.now()
        total_time = end_time - start_time
        print(f"Duration of the data collator: {total_time.total_seconds()} seconds, for {batch_size} samples, batch_operations = {batch_operations}")
        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }
