# import gym
import os
from dataclasses import dataclass
import numpy as np
import torch
from datetime import datetime 
import cProfile

@dataclass
class DecisionTransformerGymDataCollatorTensor:
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

    def __init__(self, dataset, batch_operations=False, debug = False) -> None:
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
        self.batch_operations = batch_operations
        self.debug = debug
        
    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
    
    def _discount_cumsum_along_axis(self, x, gamma=1.0):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(len(x) - 1)):  # Use len(x) to handle 1D arrays
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
    
    def _batch_preprocess(self, batch_inds):
        sS, aS, rS, dS, rtgS, timestepsS, maskS = [], [], [], [], [], [], []
        features = self.dataset[batch_inds]
        sS = np.array(features["obs"]).reshape(len(batch_inds), -1, self.state_dim)
        aS = np.array(features["acts"]).reshape(len(batch_inds), -1, self.act_dim)
        rS = np.array(np.array(features['rews']).reshape(len(batch_inds), -1, 1))
        dS = np.array(features["terminal"]).reshape(len(batch_inds), -1)
        timestepsS = np.repeat(np.arange(1, sS.shape[1]).reshape(1, -1), len(batch_inds), axis=0)
        timestepsS[timestepsS >= self.max_ep_len] = 0  # padding cutoff, home made
        rtgS = np.apply_along_axis(self._discount_cumsum_along_axis, 1, np.array(features["rews"]), gamma=1.0).reshape(len(batch_inds), -1, 1)
        
        if rtgS.shape[1] < sS.shape[1]:
            # Let try to append the zeros to the header of the array
            # Given: timesteps.shape = torch.Size([64, 19]), a.shape = torch.Size([64, 19, 3]), r.shape = torch.Size([64, 19, 1]), d.shape = torch.Size([64, 19]), rtg.shape = torch.Size([64, 20, 1]), s.shape = torch.Size([64, 20, 6])
            rtgS = np.concatenate([rtgS, np.zeros((len(batch_inds), 1, 1))], axis=1)
            timestepsS = np.concatenate([np.zeros((len(batch_inds), 1)), timestepsS], axis=1)
            # a[-1].shape[1] should be 4, s[-1].shape[1] should be 5
            aS = np.concatenate([aS, np.ones((len(batch_inds), 1, self.act_dim)) * -10], axis=1)
            rS = np.concatenate([rS, np.zeros((len(batch_inds), 1, 1))], axis=1)
            dS = np.concatenate([np.ones((len(batch_inds), 1)) * 2, dS], axis=1)
        
        tlenS = sS.shape[1]
        sS = np.concatenate([np.zeros((len(batch_inds), self.max_len - tlenS, self.state_dim)), sS], axis=1)
        sS = (sS - self.state_mean) / self.state_std
        aS = np.concatenate(
            [np.ones((len(batch_inds), self.max_len - tlenS, self.act_dim)) * -10.0, aS],
            axis=1,
        )
        rS = np.concatenate([np.zeros((len(batch_inds), self.max_len - tlenS, 1)), rS], axis=1)
        dS = np.concatenate([np.ones((len(batch_inds) , self.max_len - tlenS)) * 2, dS], axis=1)
        rtgS = np.concatenate([np.zeros((len(batch_inds), self.max_len - tlenS, 1)), rtgS], axis=1) / self.scale
        timestepsS = np.concatenate([np.zeros((len(batch_inds), self.max_len - tlenS)), timestepsS], axis=1)
        maskS = np.concatenate([
            np.zeros((len(batch_inds), self.max_len - tlenS)),
            np.ones((len(batch_inds), tlenS - 1)),
            np.zeros((len(batch_inds), 1)),
        ], axis=1)
        
        s = torch.from_numpy(sS).float()
        a = torch.from_numpy(aS).float()
        r = torch.from_numpy(rS).float()
        d = torch.from_numpy(dS)
        rtg = torch.from_numpy(rtgS).float()
        timesteps = torch.from_numpy(timestepsS).long()
        mask = torch.from_numpy(maskS).float()
        return s, a, r, d, rtg, timesteps, mask
    
    def debug_run(self):
        profiler = cProfile.Profile()
        profiler.enable()
        batch_inds = np.arange(self.n_traj)
        start_time_batch = datetime.now()
        s, a, r, d, rtg, timesteps, mask = self._batch_preprocess(batch_inds)
        end_time_batch = datetime.now()
        total_time_batch = end_time_batch - start_time_batch
        profiler.disable()
        profiler.dump_stats("batch_preprocess.prof")
        print(f"Duration of the batch_preprocess: {total_time_batch.total_seconds()} seconds")
        start_time_loop = datetime.now()
        s1, a1, r1, d1, rtg1, timesteps1, mask1 = self._loop_preprocess(batch_inds)
        end_time_loop = datetime.now()
        total_time_loop = end_time_loop - start_time_loop
        print(f"Duration of the loop_preprocess: {total_time_loop.total_seconds()} seconds")
        assert torch.allclose(s, s1), f"Expected s: {s1[0]}, got: {s[0]}"
        assert torch.allclose(a, a1), f"Expected a: {a1[0]}, got: {a[0]}"
        assert torch.allclose(r, r1), f"Expected r: {r1[0]}, got: {r[0]}"
        assert torch.allclose(d, d1), f"Expected d: {d1[0]}, got: {d[0]}"
        assert torch.allclose(rtg, rtg1), f"Expected rtg: {rtg1[0]}, got: {rtg[0]}"
        assert torch.allclose(timesteps, timesteps1), f"Expected timesteps: {timesteps1[0]}, got: {timesteps[0]}"
        assert torch.allclose(mask, mask1), f"Expected mask: {mask1[0]}, got: {mask[0]}"
        raise ValueError("Debugging")
    
    def _loop_preprocess(self, batch_inds, ):
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
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
                # Let try to append the zeros to the header of the array
                # Given: timesteps.shape = torch.Size([64, 19]), a.shape = torch.Size([64, 19, 3]), r.shape = torch.Size([64, 19, 1]), d.shape = torch.Size([64, 19]), rtg.shape = torch.Size([64, 20, 1]), s.shape = torch.Size([64, 20, 6])
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
                timesteps[-1] = np.concatenate([np.zeros((1, 1)), timesteps[-1]], axis=1)
                # a[-1].shape[1] should be 4, s[-1].shape[1] should be 5
                a[-1] = np.concatenate([a[-1], np.ones((1, 1, self.act_dim)) * -10], axis=1)
                r[-1] = np.concatenate([r[-1],np.zeros((1, 1, 1))], axis=1)
                d[-1] = np.concatenate([np.ones((1, 1)) * 2, d[-1]], axis=1)
                
            # padding and state + reward normalization
            tlen = s[-1].shape[1]

            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )

            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)

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
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()
        return s, a, r, d, rtg, timesteps, mask  
          
    def __call__(self, features):
        start_time = datetime.now() 
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        if self.debug:
            self.debug_run()
        else:
            batch_inds = np.random.choice(
                np.arange(self.n_traj),
                size=batch_size,
                replace=True,
                p=self.p_sample,  # reweights so we sample according to timesteps
            )
            
        if self.batch_operations:
            s, a, r, d, rtg, timesteps, mask = self._batch_preprocess(batch_inds)
        else:
            s, a, r, d, rtg, timesteps, mask = self._loop_preprocess(batch_inds)
        # from here we can use the same code for both batch_operations and not batch_operations
        if a.shape[0] == timesteps.shape[0] + 1:
            timesteps = np.concatenate([np.zeros((1, 1)), timesteps], axis=1)
            assert a.shape[0] == timesteps.shape[0]
            assert timesteps[-1] == 3
        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }
        
