import torch
import torch.nn as nn
from functools import partial


class QNetwork(nn.Module):
    def __init__(self, in_features, out_actions, h_dims=None, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()

        # ====== YOUR CODE: ======
        layers = []
        activation = nn.ReLU

        if h_dims is None:
            h_dims = [64,64]

        in_dim = in_features
        for h_dim in h_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            #layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation())
            #layers.append(nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=out_actions))
        self.layers = nn.Sequential(*layers)
        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "h_dims": h_dims}
        # ========================

    def forward(self, x):
        # ====== YOUR CODE: ======
        action_scores = self.layers(x)
        # ========================
        return action_scores
    
    def predict(self, state):

        with torch.no_grad():
            actions_prob = torch.squeeze(self(state))
            selected_action = actions_prob.argmax(dim=0)
        
        return selected_action


class BaseNetwork(nn.Module):
    def forward(self, x):

        action_scores = self.policy_net(x)
        return action_scores
    
    def predict(self, state):

        with torch.no_grad():
            actions_prob = torch.squeeze(self(state))
        
        return actions_prob.cpu()
    
    @staticmethod
    def soft_update(target_model: nn.Module, from_model: nn.Module, tau):

        for target_param, input_param in zip(target_model.parameters(), from_model.parameters()):
            target_param.data.copy_(tau*input_param.data + (1.0 - tau)*target_param.data)
    
    @staticmethod
    def clip_grads(model: nn.Module, min, max):

        for param in model.parameters():
            param.grad.data.clamp_(-min, max)

class QNetworkPolicy(BaseNetwork):
    def __init__(self, in_features, out_actions, h_dims=None, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()

        # ====== YOUR CODE: ======

        activation = nn.ReLU

        if h_dims is None:
            h_dims = [64,64]

        layers = []
        in_dim = in_features
        for h_dim in h_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=out_actions))
        self.policy_net = nn.Sequential(*layers)

        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "h_dims": h_dims}
        # ========================


class DQNPolicy(BaseNetwork):
    def __init__(self, in_features, out_actions, h_dims=None, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()

        # ====== YOUR CODE: ======

        activation = nn.ReLU

        if h_dims is None:
            h_dims = [64,64]

        layers = []
        in_dim = in_features
        for h_dim in h_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=out_actions))
        self.policy_net = nn.Sequential(*layers)
        
        layers = []
        in_dim = in_features
        for h_dim in h_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=out_actions))
        self.target_net = nn.Sequential(*layers)

        self.soft_update(self.target_net, self.policy_net, tau=1.0)
        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "h_dims": h_dims}
        # ========================


class AACPolicy(BaseNetwork):
    def __init__(self, in_features, out_actions, h_dims=None, ortho_init=True, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()

        # ====== YOUR CODE: ======

        activation = nn.Tanh

        if h_dims is None:
            h_dims = [64,64]

        value_layers = []
        in_dim = in_features
        for h_dim in h_dims:
            value_layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            value_layers.append(activation())
            in_dim = h_dim
        value_final = nn.Linear(in_features=in_dim, out_features=1)
        
        policy_layers = []
        in_dim = in_features
        for h_dim in h_dims:
            policy_layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            policy_layers.append(activation())
            in_dim = h_dim
        policy_final = nn.Linear(in_features=in_dim, out_features=out_actions)

        if ortho_init:
            map(partial(self.init_weights, gain=2**0.5), value_layers)
            self.init_weights(value_final, gain=1)
            map(partial(self.init_weights, gain=2**0.5), policy_layers)
            self.init_weights(policy_final, gain=0.01)

        value_layers.append(value_final)
        self.state_net = nn.Sequential(*value_layers)

        policy_layers.append(policy_final)
        self.policy_net = nn.Sequential(*policy_layers)
        
        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "h_dims": h_dims}
        # ========================

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1):

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

PPOPolicy = AACPolicy

class SACPolicy(nn.Module):
    def __init__(self, in_features, out_actions, max_grad_norm, h_dims=None, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()

        # ====== YOUR CODE: ======
        self.max_grad_norm = max_grad_norm
        
        activation = nn.ReLU

        if h_dims is None:
            h_dims = [64,64]

        layers = []
        in_dim = in_features
        for h_dim in h_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=out_actions))
        self.q_net_policy = nn.Sequential(*layers)
        
        layers = []
        in_dim = in_features
        for h_dim in h_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_features=in_dim, out_features=out_actions))
        self.q_net_target = nn.Sequential(*layers)

        self.update_target()
        self.params = {"class": self.__class__, "in_features": in_features, "out_actions": out_actions, "h_dims": h_dims}
        # ========================

    def update_target(self):

        self.q_net_target.load_state_dict(self.q_net_policy.state_dict())

    def forward(self, x):
        # ====== YOUR CODE: ======
        action_scores = self.q_net_policy(x)
        # ========================
        return action_scores
    
    def clip_policy_grads(self):
        for param in self.q_net_policy.parameters():
            param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
        # nn.utils.clip_grad_norm_(self.q_net_policy.parameters(), self.max_grad_norm)
    
    def predict(self, state):

        with torch.no_grad():
            actions_prob = torch.squeeze(self(state))
            selected_action = actions_prob.argmax(dim=0)
        
        return selected_action.item()