import torch
import torch.nn as nn
import torch.optim as optim
from data import Experience, TrainBatch
from typing import List
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: optim.Optimizer,
        lr: float,
        num_actions: int,
        entropy_coef: float = 0.3, #for entropy
        gamma: float = 0.99,
        device = "cpu",
        **kw,
    ):
        self.model = policy
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        if isinstance(device, str):
            self.device = torch.device("cuda" if device=="cuda" and torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.current_episode = 0

        self.model.to(self.device)
        self.max_entropy = -(1/num_actions) * torch.log(torch.empty(1, device=self.device).fill_(1/num_actions)) * 3
    
    def entropy_loss(self, action_scores: torch.Tensor):
        state_entropy = -(action_scores.softmax(dim=1) * action_scores.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
        return -state_entropy/self.max_entropy #min loss_e -> max entropy

    def to(self, device):
        self.model.to(device)
    
    def train(self, batch: List[Experience]):

        self.current_episode += 1

        batch = TrainBatch(*zip(*batch), device=self.device)

        return self.train_batch(batch)

    @abstractmethod
    def train_batch(self, batch: TrainBatch):

        """
        Train the policy according to batch.
        Return tuple of loss, and a dictionary of all the losses
        """
        pass


class QNetworkTrainer(BaseTrainer):        

    def train_batch(self, batch: TrainBatch):

        next_state_values = self.model(batch.next_states).max(dim=1)[0].detach() * (~batch.is_dones)
        # next_state_values[batch.is_dones] = torch.zeros(torch.sum(batch.is_dones).item(), device=self.device)

        expected_state_action_values = (next_state_values * self.gamma) + batch.rewards

        action_scores = self.model(batch.states)
        loss_e = self.entropy_loss(action_scores)
        state_action_values = action_scores.gather(1, batch.actions[..., None])

        criterion = nn.SmoothL1Loss()
        loss_p = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        loss = loss_p + self.entropy_coef*loss_e

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss = loss.to("cpu").item()
        print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p.to("cpu").item(), \
            loss_e=loss_e.to("cpu").item())


class DQNTrainer(BaseTrainer):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: optim.Optimizer,
        lr: float,
        num_actions: int,
        max_grad_norm: float,
        entropy_coef: float = 0.3, #for entropy
        gamma: float = 0.99,
        tau: float = 1.0,
        target_update: int = 10,
        device = "cpu",
        **kw,
    ):
        super(DQNTrainer, self).__init__(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef, 
            gamma,  
            device,
            **kw,
        )
        
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.target_update = target_update

    def train_batch(self, batch: TrainBatch):

        next_state_values = self.model.target_net(batch.next_states).max(dim=1)[0].detach() * (~batch.is_dones)
        # next_state_values[batch.is_dones] = torch.zeros(torch.sum(batch.is_dones).item(), device=self.device)

        expected_state_action_values = (next_state_values * self.gamma) + batch.rewards

        action_scores = self.model(batch.states)
        loss_e = self.entropy_loss(action_scores)
        state_action_values = action_scores.gather(1, batch.actions[..., None])

        criterion = nn.SmoothL1Loss()
        loss_p = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        loss = loss_p + self.entropy_coef*loss_e

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.model.clip_grads(self.model.policy_net, -self.max_grad_norm, self.max_grad_norm)
        self.optimizer.step()
        
        if self.current_episode % self.target_update == 0:
            self.model.soft_update(self.model.target_net, self.model.policy_net, tau=self.tau)
        
        loss = loss.to("cpu").item()
        print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p.to("cpu").item(), \
            loss_e=loss_e.to("cpu").item())


class AACTrainer(BaseTrainer):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: optim.Optimizer,
        lr: float,
        num_actions: int,
        max_grad_norm: float,
        entropy_coef: float = 0.3, #for entropy
        value_coef: float = 0.5,
        gae_coef: float = 1.0,
        gamma: float = 0.99,
        normalize_advantages: bool = False,
        device = "cpu",
        **kw,
    ):
        super(AACTrainer, self).__init__(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef, 
            gamma, 
            device,
            **kw,
        )
        
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.gae_coef = gae_coef
        self.normalize_advantages = normalize_advantages

    def train_batch(self, batch: TrainBatch):

        action_scores = self.model(batch.states)
        state_values = self.model.state_net(batch.states).squeeze(dim=1)
        
        loss_e = self.entropy_loss(action_scores)

        loss_v = nn.MSELoss()(batch.qvals, state_values)
        
        with torch.no_grad():
            # basically, no need of max since shape is [N,1]
            next_state_values = self.model.state_net(batch.next_states).max(dim=1)[0] * (~batch.is_dones)
            advantages = next_state_values * self.gamma + batch.rewards - state_values
            next_gae = 0
            for i in range(advantages.shape[0]-1, -1, -1):
                next_gae = advantages[i] + self.gamma * self.gae_coef * next_gae
                advantages[i] = next_gae
            # advantages = batch.qvals - state_values
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean())
                if batch.states.shape[0]>1:
                    advantages /= (advantages.std() + 1e-8)
        
        log_action_proba = nn.functional.log_softmax(action_scores, dim=1)
        selected_action_log_proba = log_action_proba.gather(dim=1, index=batch.actions[..., None]).squeeze(dim=1)
        weighted_avg_experience_rewards = selected_action_log_proba * advantages
        loss_p = -weighted_avg_experience_rewards.mean()

        loss = loss_p + self.entropy_coef*loss_e + self.value_coef*loss_v

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.model.clip_grads(self.model.policy_net, -self.max_grad_norm, self.max_grad_norm)
        self.model.clip_grads(self.model.state_net, -self.max_grad_norm, self.max_grad_norm)
        self.optimizer.step()
        
        loss = loss.to("cpu").item()
        print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p.to("cpu").item(), \
            loss_e=loss_e.to("cpu").item(), \
            loss_v=loss_v.to("cpu").item())


class PPOTrainer(BaseTrainer):
    def __init__(
        self,
        policy: nn.Module,
        optimizer: optim.Optimizer,
        lr: float,
        num_actions: int,
        max_grad_norm: float,
        entropy_coef: float = 0.3, #for entropy
        value_coef: float = 0.5,
        gae_coef: float = 1.0,
        gamma: float = 0.99,
        normalize_advantages: bool = False,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        device = "cpu",
        **kw,
    ):
        super(PPOTrainer, self).__init__(
            policy, 
            optimizer, 
            lr, 
            num_actions, 
            entropy_coef, 
            gamma, 
            device,
            **kw,
        )
        
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.gae_coef = gae_coef
        self.normalize_advantages = normalize_advantages
        self.n_epochs = n_epochs
        self.clip_range = clip_range

    def train_batch(self, batch: TrainBatch):

        old_action_scores = self.model(batch.states).detach()
        state_values = self.model.state_net(batch.states).squeeze(dim=1).detach()
        
        with torch.no_grad():
            # basically, no need of max since shape is [N,1]
            next_state_values = self.model.state_net(batch.next_states).max(dim=1)[0] * (~batch.is_dones)
            advantages = next_state_values * self.gamma + batch.rewards - state_values
            next_gae = 0
            for i in range(advantages.shape[0]-1, -1, -1):
                next_gae = advantages[i] + self.gamma * self.gae_coef * next_gae
                advantages[i] = next_gae
            # advantages = batch.qvals - state_values
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean())
                if batch.states.shape[0]>1:
                    advantages /= (advantages.std() + 1e-8)
        
            old_log_action_proba = nn.functional.log_softmax(old_action_scores, dim=1)
            old_selected_action_log_proba = old_log_action_proba.gather(dim=1, index=batch.actions[..., None]).squeeze(dim=1)

        losses = []
        losses_p, losses_e, losses_v = [], [], []
        for _ in range(self.n_epochs):

            state_values = self.model.state_net(batch.states).squeeze(dim=1)
            loss_v = nn.MSELoss()(batch.qvals, state_values)
            
            action_scores = self.model(batch.states)
            log_action_proba = nn.functional.log_softmax(action_scores, dim=1)
            selected_action_log_proba = log_action_proba.gather(dim=1, index=batch.actions[..., None]).squeeze(dim=1)

            ratio = torch.exp(selected_action_log_proba - old_selected_action_log_proba)

            loss_p1 = advantages * ratio
            loss_p2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            loss_p = -torch.min(loss_p1, loss_p2).mean()
        
            loss_e = self.entropy_loss(action_scores)

            loss = loss_p + self.entropy_coef*loss_e + self.value_coef*loss_v
            
            # logging
            losses.append(loss.to("cpu").item())
            losses_p.append(loss_p.to("cpu").item())
            losses_e.append(loss_e.to("cpu").item())
            losses_v.append(loss_v.to("cpu").item())

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.model.clip_grads(self.model.policy_net, -self.max_grad_norm, self.max_grad_norm)
            self.model.clip_grads(self.model.state_net, -self.max_grad_norm, self.max_grad_norm)
            self.optimizer.step()
        
        def mean(l):
            return sum(l)/len(l)

        loss = mean(losses)
        loss_p = mean(losses_p)
        loss_e = mean(losses_e)
        loss_v = mean(losses_v)

        print("total_loss = ", loss)
        return loss, dict(
            loss=loss, \
            loss_p=loss_p, \
            loss_e=loss_e, \
            loss_v=loss_v)
