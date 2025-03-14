from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"
    max_len: int = 4 #subsets of the episode we use for training
    state_dim: int = 6  # size of state space, should be 6 
    act_dim: int = 3  # size of action space, should be 3
    max_ep_len: int = 4 # max episode length in the dataset
    # scale: float =1.0
    scale: float = 1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, dataset) -> None:
        self.act_dim =  len(dataset[0]["acts"][0])#len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["obs"][0]) #len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        for obs in dataset["obs"]:
            states.extend(obs)
            traj_lens.append(len(obs))
            # print (f"obs = {obs}")
        self.n_traj = len(traj_lens)
        print (f"self.n_traj = {self.n_traj}")
        states = np.vstack(states)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        print (f"self.state_mean, self.state_std = {self.state_mean, self.state_std}")
        traj_lens = np.array(traj_lens)
        print (f"traj_lens = {traj_lens}")
        self.p_sample = traj_lens / sum(traj_lens)
        
        
    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
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
        
        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]
            # This line creates bugs
            # si = random.randint(0, len(feature["rews"]) - 1)
            si = 0
            # get sequences from dataset
            s.append(np.array(feature["obs"][si : si + self.max_len]).reshape(1, -1, self.state_dim))
            if s[-1].shape[1] == 3:
                print (f"si + self.max_len {si + self.max_len}, si: {si}")
                print ("feature['obs'][si : si + self.max_len] ", feature["obs"][si : si + self.max_len])
            a.append(np.array(feature["acts"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["rews"][si : si + self.max_len]).reshape(1, -1, 1))
            # TODO: think about how to handle dones
            # d.append(np.array(feature["dones"][si : si + self.max_len]).reshape(1, -1))
            d.append(np.array(feature["terminal"][si : si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            # import ipdb; ipdb.set_trace()
            rtg.append(
                self._discount_cumsum(np.array(feature["rews"][si:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            if a[-1].shape[1] == 3:
                print (f"a[-1].shape, {a[-1].shape}, a[-1]: {a[-1]}, self.max_len: {self.max_len}, tlen: {tlen}")
                raise NotImplementedError
                
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        # for i in range(len(a)):   
        #     print ("a[i].shape", a[i].shape)
        #     print( "i = ", i)
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }
