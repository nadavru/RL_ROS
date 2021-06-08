#!/usr/bin/env python3

from rl_agent import Agent_rl
from train import *
from policies import *
from environment import *
import torch.optim as optim
import sys

if __name__ == '__main__':

    num_of_algorithms = 30
    assert len(sys.argv)==2, f"Number of arguments: {len(sys.argv)} arguments."
    alg = (int)(sys.argv[-1])
    assert 1<=alg<=num_of_algorithms, f"Parameter ({alg}) should be in [1,{num_of_algorithms}]."
    
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

    if alg==1:

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
    
    elif alg==2:

        hidden_layers = None # [64,64]
        lr = 0.0001
        buffer_size = 1000
        batch_size = 32
        gamma = 0.99
        entropy_coef = 0.3 #
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
    
    elif alg==3:

        hidden_layers = None # [64,64]
        lr = 0.0001
        buffer_size = 1000
        batch_size = 32
        gamma = 0.99
        entropy_coef = 0.3 #
        target_update = 10 #
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
    
    elif alg==4:

        hidden_layers = None # [64,64]
        lr = 0.0001
        buffer_size = 1000
        batch_size = 32
        gamma = 0.99
        entropy_coef = 1 #
        target_update = 10 #
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
    
    elif alg==5:
        
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
    
    elif alg==6:
        
        hidden_layers = None # [64,64]
        lr = 0.0001
        buffer_size = 1000
        batch_size = 32
        gamma = 0.99
        entropy_coef = 0.3 #
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
    
    elif alg==7:

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
    
    elif alg==8:

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
        value_coef = 0.1 #
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
    
    elif alg==9:

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
        gae_coef = 0.0
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
    
    elif alg==10:

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
        value_coef = 0.1 #
        gae_coef = 0.0
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
    
    elif alg==11:

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
        normalize_advantages = False #

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
    
    elif alg==12:

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
        value_coef = 0.1 #
        gae_coef = 1.0 # 0.0
        normalize_advantages = False #

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
    
    elif alg==13:

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
        gae_coef = 0.0
        normalize_advantages = False #

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
    
    elif alg==14:

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
        value_coef = 0.1 #
        gae_coef = 0.0
        normalize_advantages = False #

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
    
    elif alg==15:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 0.95 # 0.0
        normalize_advantages = True #
        n_epochs = 10 #
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
    
    elif alg==16:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 0.95 # 0.0
        normalize_advantages = True #
        n_epochs = 3 #
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
    
    elif alg==17:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 0.95 # 0.0
        normalize_advantages = False #
        n_epochs = 10 #
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
    
    elif alg==18:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 0.95 # 0.0
        normalize_advantages = False #
        n_epochs = 3 #
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
    
    elif alg==19:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 0.0
        normalize_advantages = True #
        n_epochs = 10 #
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
    
    elif alg==20:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 0.0
        normalize_advantages = True #
        n_epochs = 3 #
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
    
    elif alg==21:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 0.0
        normalize_advantages = False #
        n_epochs = 10 #
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
    
    elif alg==22:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.5 #
        gae_coef = 0.0
        normalize_advantages = False #
        n_epochs = 3 #
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
    
    elif alg==23:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.1 #
        gae_coef = 0.95 # 0.0
        normalize_advantages = True #
        n_epochs = 10 #
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
    
    elif alg==24:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.1 #
        gae_coef = 0.95 # 0.0
        normalize_advantages = True #
        n_epochs = 3 #
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
    
    elif alg==25:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.1 #
        gae_coef = 0.95 # 0.0
        normalize_advantages = False #
        n_epochs = 10 #
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
    
    elif alg==26:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.1 #
        gae_coef = 0.95 # 0.0
        normalize_advantages = False #
        n_epochs = 3 #
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
    
    elif alg==27:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.1 #
        gae_coef = 0.0
        normalize_advantages = True #
        n_epochs = 10 #
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
    
    elif alg==28:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.1 #
        gae_coef = 0.0
        normalize_advantages = True #
        n_epochs = 3 #
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
    
    elif alg==29:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.1 #
        gae_coef = 0.0
        normalize_advantages = False #
        n_epochs = 10 #
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
    
    elif alg==30:

        hidden_layers = None # [64,64]
        lr = 0.0007
        buffer_size = None
        batch_size = -1
        gamma = 0.99
        entropy_coef = 1 
        max_grad_norm = 0.5
        with_baseline = False
        save_every = 25
        off_policy = False
        optimizer = optim.Adam
        ortho_init = True
        value_coef = 0.1 #
        gae_coef = 0.0
        normalize_advantages = False #
        n_epochs = 3 #
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
        folder=f"model_{alg}")
