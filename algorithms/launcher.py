#!/usr/bin/env python3

from rl_agent import Agent_rl
from train import *
from policies import *
from environment import *
import torch.optim as optim
import sys

if __name__ == '__main__':

    '''velocity_handler = VelocityHandler(linear_step=0.1, angular_step=0.1, max_linear=1, max_angular=1)
    rewarder = RightHand()'''
    '''velocity_handler = SimpleVelocityHandler()
    rewarder = SimpleRightHand()
    env = Env(velocity_handler, rewarder, random_init=True, verbose=True)'''
    #velocity_handler = SimpleTransformer(x=0.1, y=0.05)
    '''points = [[-0.8,-2],[-1.14,-1.36],[-1.5,-1.4],[-2.3,0],[-1.5,1.4],[-1.14,1.36],[-0.8,2],
    [0.8,2],[1.14,1.36],[1.5,1.4],[2.05,0.5],[1.8,0],[2.05,-0.5],[1.5,-1.4],[1.14,-1.36],[0.8,-2]]'''
    #loss = BaselineEntropyPolicyGradientLoss(num_actions, beta=0.5)#VanillaPolicyGradientLoss()


    '''print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))'''
    
    ######################## local params
    x = 0.07
    y = 0.02
    num_of_bins = 18
    random_init = True
    velocity_handler = SimpleTransformer2(x=x, y=y)
    rewarder = SimpleRightHand(init_dest=70)
    ########################
    
    
    env = SimpleEnvTrain(
        velocity_handler, 
        rewarder, 
        num_of_bins=num_of_bins, 
        random_init=random_init)#, verbose=True)

    num_actions = env.total_bins
    num_observations = env.total_observations

    device="cuda"
    
    #TODO add exploration for predict
    #TODO add noise
    #TODO normalize states (only for train?)

    ######################## choosing algorithm
    alg = "aac"

    if alg=="dqn":

        hidden_layers = None # [64,64]
        lr = 0.0001
        buffer_size = 1000
        batch_size = 32
        gamma = 0.99
        entropy_coef = 1 #
        target_update = 100 #
        max_grad_norm = 10
        with_baseline = False #
        save_every = 25
        off_policy = True
        optimizer = optim.Adam

        policy = DQNPolicy(num_observations, num_actions, hidden_layers)
        
        trainer = DQNTrainer(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            max_grad_norm=max_grad_norm, 
            entropy_coef=entropy_coef, 
            gamma=gamma, 
            target_update=target_update, 
            with_baseline=with_baseline, 
            device=device)
    
    elif alg=="qnet":
        
        hidden_layers = None # [64,64]
        lr = 0.0001
        buffer_size = 1000
        batch_size = 32
        gamma = 0.99
        entropy_coef = 1 #
        with_baseline = False #
        save_every = 25
        off_policy = True
        optimizer = optim.Adam

        policy = QNetworkPolicy(num_observations, num_actions, hidden_layers)
        
        trainer = QNetworkTrainer(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef=entropy_coef, 
            gamma=gamma, 
            with_baseline=with_baseline, 
            device=device)
    
    elif alg=="aac":

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 #
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 1.0 # 0.0
        normalize_advantages = True #

        policy = AACPolicy(num_observations, num_actions, hidden_layers, ortho_init)
        
        trainer = AACTrainer(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            max_grad_norm=max_grad_norm, 
            entropy_coef=entropy_coef, 
            value_coef=value_coef, 
            gae_coef=gae_coef, 
            gamma=gamma, 
            with_baseline=with_baseline, 
            normalize_advantages=normalize_advantages, 
            device=device)
    
    elif alg=="ppo":

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 #
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 0.95 # 0.0
        normalize_advantages = True #
        n_epochs = 10
        clip_range = 0.2

        policy = PPOPolicy(num_observations, num_actions, hidden_layers, ortho_init)
        
        trainer = PPOTrainer(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            max_grad_norm=max_grad_norm, 
            entropy_coef=entropy_coef, 
            value_coef=value_coef, 
            gae_coef=gae_coef, 
            gamma=gamma, 
            with_baseline=with_baseline, 
            normalize_advantages=normalize_advantages, 
            n_epochs=n_epochs, 
            clip_range=clip_range, 
            device=device)
    else:
        print(f"Algorithm {alg} is not defined. Abort.")
        exit()
    ########################

    agent = Agent_rl(
        policy, 
        env, 
        device=device)

    # batch_size = size of experiences
    agent.train(
        trainer, 
        capacity=buffer_size, 
        batch_size=batch_size, 
        gamma=gamma, 
        off_policy=off_policy, 
        save_every=save_every, 
        folder="model_aac", from_episode=-1)
