#!/usr/bin/env python3

from rl_agent import Agent_rl
from train import *
from policies import *
from environment import *
import torch.optim as optim
import sys

class Counter():

    def __init__(self, arr: list):
        self.arr = arr
        self.max = self.get_size(self.arr)
        self.data = [0]*len(arr)
    
    @staticmethod
    def get_size(arr):
        num=1
        for i in arr:
            num *= i+1
        return num
    
    def next(self):
        for _ in range(self.max):
            yield self.data
            for i in range(len(self.data)):
                if self.data[i]<self.arr[i]:
                    self.data[i] += 1
                    break
                else:
                    self.data[i] = 0

class Params():

    def __init__(self):
        self.parameters = []

    def _add_from_keys(self, keys, values):
        counter = Counter([len(value)-1 for value in values])
        size = len(keys)
        for indexes in counter.next():
            current_values = [values[i][indexes[i]] for i in range(size)]
            yield zip(keys, current_values)

    def add(self, p):
        multiple_keys = []
        multiple_values = []
        for key, value in p.items():
            if type(value) == list:
                multiple_keys.append(key)
                multiple_values.append(value)
        for params in self._add_from_keys(multiple_keys, multiple_values):
            p.update(dict(params))
            self.parameters.append(p.copy())
    
    @property
    def all_parameters(self):
        return self.parameters


if __name__ == '__main__':
    
    ######################## local params
    x = 0.07
    y = 0.02
    num_of_bins = 16
    random_init = True
    velocity_handler = SimpleTransformer2(x=x, y=y)
    rewarder = SimpleRightHand(init_dest=70)
    ########################

    ######################## global params
    save_every = 25
    device="cuda"
    verbose = False
    eval = False
    ########################    
    
    ######################## environment initialization
    env = SimpleEnvTrain(
        velocity_handler, 
        rewarder, 
        num_of_bins=num_of_bins, 
        random_init=random_init,
        verbose=verbose)

    num_actions = env.total_bins
    num_observations = env.total_observations
    ########################

    #TODO add exploration for predict
    #TODO add noise
    #TODO normalize states (only for train?)

    ######################## defining parameters
    DQNParams = {
        "algorithm" : "DQN",
        "DQN" : "DQN",
        "hidden_layers" : None,
        "lr" : 0.0001,
        "buffer_size" : 1000,
        "batch_size" : 32,
        "gamma" : 0.99,
        "tau" : 1.0,
        "eps" : 0.3,
        "entropy_coef" : 1,
        "target_update" : [10, 100],
        "max_grad_norm" : 10,
        "off_policy" : True,
        "optimizer" : optim.Adam
    }

    DQNRainbowParams = {
        "algorithm" : "DQNRainbow",
        "DQN" : "DQN",
        "hidden_layers" : None,
        "lr" : 0.0001,
        "buffer_size" : 1000,
        "batch_size" : 32,
        "gamma" : 0.99,
        "tau" : 1.0,
        "eps" : 0.3,
        "entropy_coef" : 1,
        "target_update" : [10, 100],
        "max_grad_norm" : 10,
        "off_policy" : True,
        "optimizer" : optim.Adam
    }

    QNetworkParams = {
        "algorithm" : "QNetwork",
        "hidden_layers" : None,
        "lr" : 0.0001,
        "buffer_size" : 1000,
        "batch_size" : 32,
        "gamma" : 0.99,
        "eps" : 0.3,
        "entropy_coef" : 1,
        "off_policy" : True,
        "optimizer" : optim.Adam
    }
    
    AACParams = {
        "algorithm" : "AAC",
        "hidden_layers" : None,
        "lr" : 0.0007,
        "buffer_size" : None,
        "batch_size" : -1,
        "gamma" : 0.99,
        "eps" : 0.3,
        "entropy_coef" : 1,
        "max_grad_norm" : 0.5,
        "off_policy" : False,
        "optimizer" : optim.Adam,
        "ortho_init" : True,
        "value_coef" : [0.1, 0.5],
        "gae_coef" : [0.0, 1.0],
        "normalize_advantages" : [False, True]
    }
    
    PPOParams = {
        "algorithm" : "PPO",
        "hidden_layers" : None,
        "lr" : 0.0007,
        "buffer_size" : None,
        "batch_size" : -1,
        "gamma" : 0.99,
        "eps" : 0.3,
        "entropy_coef" : 1 ,
        "max_grad_norm" : 0.5,
        "off_policy" : False,
        "optimizer" : optim.Adam,
        "ortho_init" : True,
        "value_coef" : [0.1, 0.5],
        "gae_coef" : [0.0, 0.95],
        "normalize_advantages" : [False, True],
        "n_epochs" : [3, 10],
        "clip_range" : 0.2
    }

    SACParams = {
        "algorithm" : "SAC",

        "buffer_size" : 1000000,
        "batch_size" : 10,
        "gamma" : 0.99,
        "eps" : 0.3,
        "off_policy" : True,
        "gradient_steps" : 1, #[-1, 1],

        "hidden_layers" : None,
        "n_critics" : 2,
        
        "optimizer" : optim.Adam,
        "lr" : 0.0003,
        "entropy_coef" : 1,
        "tau": 0.005
    }

    TD3Params = {
        "algorithm" : "TD3",

        "buffer_size" : 1000000,
        "batch_size" : 100,
        "gamma" : 0.99,
        "eps" : 0.3,
        "off_policy" : True,
        "gradient_steps" : [-1, 1],

        "hidden_layers" : None,
        "n_critics" : 2,
        
        "optimizer" : optim.Adam,
        "lr" : 0.001,
        "entropy_coef" : 1,
        "tau": 0.005,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "policy_delay": 2
    }

    params = Params()
    params.add(DQNParams)
    params.add(DQNRainbowParams)
    params.add(QNetworkParams)
    params.add(AACParams)
    params.add(PPOParams)
    params.add(SACParams)
    params.add(TD3Params)

    Policies = {
        "DQN": DQNPolicy,
        "DQNRainbow": DQNPolicy,
        "QNetwork": QNetworkPolicy,
        "AAC": AACPolicy,
        "PPO": PPOPolicy,
        "SAC": SACPolicy,
        "TD3": TD3Policy,
    }
    Trainers = {
        "DQN": DQNTrainer,
        "DQNRainbow": DQNRainbowTrainer,
        "QNetwork": QNetworkTrainer,
        "AAC": AACTrainer,
        "PPO": PPOTrainer,
        "SAC": SACTrainer,
        "TD3": TD3Trainer,
    }
    ########################
    
    ######################## choosing algorithm
    assert len(sys.argv)==2, f"Number of arguments: {len(sys.argv)} arguments."
    alg = (int)(sys.argv[-1])
    num_of_algorithms = len(params.all_parameters)
    assert 1<=alg<=num_of_algorithms, f"Parameter ({alg}) should be in [1,{num_of_algorithms}]."
    ########################

    chosen_params = params.all_parameters[alg-1]
    algorithm = chosen_params["algorithm"]
    if not eval:
        rospy.loginfo(f"Training with {algorithm}")
    else:
        rospy.loginfo(f"Evaluating with {algorithm}")
    Policy = Policies[algorithm]
    Trainer = Trainers[algorithm]

    policy = Policy(in_features = num_observations, out_actions = num_actions, **chosen_params)
    trainer = Trainer(policy = policy, num_actions = num_actions, device=device, **chosen_params)

    agent = Agent_rl(
        policy, 
        env, 
        device=device)

    # batch_size = size of experiences
    if not eval:
        agent.train(
            trainer, 
            save_every=save_every, 
            folder=f"model_{alg}",
            from_episode=-1,
            **chosen_params)
    else:
        agent.eval(
            folder=f"model_{alg}",
            from_episode=-1)