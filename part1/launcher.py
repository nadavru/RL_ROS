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
    
    ######################## choosing algorithm
    assert len(sys.argv)>=2, f"Number of arguments: {len(sys.argv)} arguments."
    alg = (int)(sys.argv[1])
    ########################

    ######################## global params
    save_every = 25
    device="cuda"
    verbose = False
    eval = 0 # 0 for train, 1 for eval, 2 for eval with domain network
    if len(sys.argv)>=3:
        eval = (int)(sys.argv[2])
    ########################
    
    ######################## local params
    x = 0.07
    y = 0.02
    num_of_bins = 16
    random_init = True
    velocity_handler = SimpleTransformer2(x=x, y=y)
    rewarder = SimpleRightHand(init_dest=70)
    ########################
    
    ######################## environment initialization
    if eval!=2:
        env = FirstEnv(
            velocity_handler, 
            rewarder, 
            num_of_bins=num_of_bins, 
            random_init=random_init,
            verbose=verbose)
    else:
        env = SecondEnv(
            velocity_handler, 
            rewarder, 
            num_of_bins=num_of_bins, 
            random_init=random_init,
            verbose=verbose,
            domain_net_path="domain_net/model",
            device=device)

    num_actions = env.total_bins
    num_observations = env.total_observations
    ########################

    #TODO add exploration for predict
    #TODO add noise
    #TODO normalize states (only for train?)

    ######################## defining parameters
    DQNParams = {
        "algorithm" : "DQN",
        "hidden_layers" : [[64,64],[400,400]],
        "lr" : [0.0001, 0.001],
        "buffer_size" : 1000000,
        "batch_size" : 32,
        "gamma" : 0.99,
        "tau" : 1.0,
        "eps" : 1.0,
        "entropy_coef" : [0,1],
        "target_update" : 1000,
        "max_grad_norm" : 10,
        "off_policy" : True,
        "optimizer" : optim.Adam,
        "learning_starts" : 50000,
        "eps_gamma" : 0.1
    }

    DQNRainbowParams = {
        "algorithm" : "DQNRainbow",
        "hidden_layers" : [[64,64],[400,400]],
        "lr" : [0.0001, 0.001],
        "buffer_size" : 1000000,
        "batch_size" : 32,
        "gamma" : 0.99,
        "tau" : 1.0,
        "eps" : 1.0,
        "entropy_coef" : [0,1],
        "target_update" : 1000,
        "max_grad_norm" : 10,
        "off_policy" : True,
        "optimizer" : optim.Adam,
        "learning_starts" : 50000,
        "eps_gamma" : 0.1
    }

    QNetworkParams = {
        "algorithm" : "QNetwork",
        "hidden_layers" : [[64,64],[400,400]],
        "lr" : [0.0001, 0.001],
        "buffer_size" : 1000000,
        "batch_size" : 32,
        "gamma" : 0.99,
        "eps" : 1.0,
        "entropy_coef" : [0,1],
        "off_policy" : True,
        "optimizer" : optim.Adam,
        "learning_starts" : 50000,
        "eps_gamma" : 0.1
    }
    
    # n_steps = 5
    AACParams = {
        "algorithm" : "AAC",
        "hidden_layers" : [[64,64],[400,400]],
        "lr" : [0.0007, 0.007],
        "buffer_size" : None,
        "batch_size" : -1,
        "gamma" : 0.99,
        "eps" : 0,
        "entropy_coef" : [0,1],
        "max_grad_norm" : 0.5,
        "off_policy" : False,
        "optimizer" : [optim.Adam, optim.RMSprop],
        "ortho_init" : True,
        "value_coef" : 0.5,
        "gae_coef" : 1.0,
        "normalize_advantages" : [False, True],
        "learning_starts" : 1,
        "eps_gamma" : 0.997, 
        "n_episodes" : 1
    }
    
    # n_steps = 2048
    PPOParams = {
        "algorithm" : "PPO",
        "hidden_layers" : [[64,64],[400,400]],
        "lr" : [0.0003, 0.003],
        "buffer_size" : None,
        "batch_size" : -1,
        "gamma" : 0.99,
        "eps" : 0,
        "entropy_coef" : [0,1] ,
        "max_grad_norm" : 0.5,
        "off_policy" : False,
        "optimizer" : optim.Adam,
        "ortho_init" : True,
        "value_coef" : 0.5,
        "gae_coef" : 0.95,
        "normalize_advantages" : [False, True],
        "n_epochs" : 10,
        "clip_range" : 0.2,
        "learning_starts" : 1,
        "eps_gamma" : 0.1, 
        "n_episodes" : 200
    }

    SACParams = {
        "algorithm" : "SAC",

        "buffer_size" : 1000000,
        "batch_size" : 256,
        "gamma" : 0.99,
        "eps" : 0,
        "off_policy" : True,
        "gradient_steps" : 1,
        "learning_starts" : 100,
        "eps_gamma" : 0.997,

        "hidden_layers" : [[256,256],[400,400]],
        "n_critics" : [2,4],
        
        "optimizer" : optim.Adam,
        "lr" : 0.0003,
        "entropy_coef" : [0,1],
        "tau": 0.005
    }

    TD3Params = {
        "algorithm" : "TD3",

        "buffer_size" : 1000000,
        "batch_size" : 100,
        "gamma" : 0.99,
        "eps" : [0,0.3],
        "off_policy" : True,
        "gradient_steps" : [-1, 1],
        "learning_starts" : 100,
        "eps_gamma" : 0.997,

        "hidden_layers" : [[256,256],[400,300]],
        "n_critics" : [2,4],
        
        "optimizer" : optim.Adam,
        "lr" : 0.001,
        "entropy_coef" : [0,1],
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
    
    num_of_algorithms = len(params.all_parameters)
    assert 1<=alg<=num_of_algorithms, f"Parameter ({alg}) should be in [1,{num_of_algorithms}]."

    chosen_params = params.all_parameters[alg-1]
    print(chosen_params)
    algorithm = chosen_params["algorithm"]

    
    Policy = Policies[algorithm]
    policy = Policy(in_features = num_observations, out_actions = num_actions, **chosen_params)
    agent = Agent_rl(
        policy, 
        env, 
        device=device)
    
    if eval==0:
        print(f"Training with {algorithm}")

        Trainer = Trainers[algorithm]
        trainer = Trainer(policy = policy, num_actions = num_actions, device=device, **chosen_params)
        agent.train(
            trainer, 
            save_every=save_every, 
            folder=f"model_{alg}",
            from_episode=-1,
            **chosen_params)
    elif eval==1:
        print(f"Evaluating with {algorithm}")

        agent.eval(
            folder=f"model_{alg}",
            from_episode=-1)
    else:
        print(f"Evaluating with {algorithm} (with domain network)")

        agent.eval(
            folder=f"model_{alg}",
            from_episode=-1)
