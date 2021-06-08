import torch
import torch.utils.data
import itertools
from typing import List, Tuple, Union, Callable, Iterable, Iterator, NamedTuple
from collections import deque
import random
from typing import List


class Experience(NamedTuple):
    """
    Represents one experience tuple for the Agent.
    """

    state: torch.FloatTensor
    next_state: torch.FloatTensor
    action: int # categorial
    reward: float
    qval: float
    is_done: bool


class TrainBatch(object):
    
    def __init__(
        # all in shape [batch]
        self,
        states: torch.FloatTensor, #[batch,state]
        next_states: torch.FloatTensor, #[batch,state]
        actions: torch.LongTensor, #[batch]
        rewards: torch.FloatTensor, #[batch]
        qvals: torch.FloatTensor, #[batch]
        is_dones: torch.BoolTensor, #[batch]
        device,
    ):

        states = torch.stack(list(states), dim=0)
        next_states = torch.stack(list(next_states), dim=0)
        actions = torch.LongTensor(list(actions))
        rewards = torch.FloatTensor(list(rewards))
        qvals = torch.FloatTensor(list(qvals))
        is_dones = torch.torch.BoolTensor(list(is_dones))
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == next_states.shape[0] == is_dones.shape[0]

        self.states = states.to(device)
        self.next_states = next_states.to(device)
        self.actions = actions.to(device)
        self.rewards = rewards.to(device)
        self.qvals = qvals.to(device)
        self.is_dones = is_dones.to(device)

'''class Episode(object):
    """
    Represents an entire sequence of experiences until a terminal state was
    reached.
    """

    def __init__(self, total_reward, experiences):
        self.total_reward = total_reward
        self.experiences = experiences

class TrainBatch(object):
    """
    Holds a batch of data to train on.
    """

    def __init__(
        # all in shape [exps]
        self,
        states,#: torch.FloatTensor, [experiences,state]
        actions,#: torch.LongTensor, [experiences]
        rewards,#: torch.FloatTensor, [experiences]
        next_states,#: torch.FloatTensor, [experiences,state]
        is_dones,#: torch.FloatTensor, [experiences]
        total_rewards,#: torch.FloatTensor, [episodes]
    ):

        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == next_states.shape[0]

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.total_rewards = total_rewards

    ''''''def __iter__(self):
        return iter([self.states, self.actions, self.q_vals, self.total_rewards])''''''

    @classmethod
    def from_episodes(cls, episodes, device=None):
        """
        Constructs a TrainBatch from a list of Episodes by extracting all
        experiences from all episodes.
        :param episodes: List of episodes to create the TrainBatch from.
        :param gamma: Discount factor for q-vals calculation
        """
        train_batch = None

        #   - Extract states, actions and total rewards from episodes.
        #   - Calculate the q-values for states in each experience.
        #   - Construct a TrainBatch instance.
        # ====== YOUR CODE: ======
        device = torch.device("cuda" if orch.cuda.is_available() else "cpu") if device is None else device

        experiences = list(itertools.chain(*[ep.experiences for ep in episodes]))

        states = torch.cat([exp.state.unsqueeze(dim=0) for exp in experiences], dim=0).to(device)

        actions = torch.LongTensor([exp.action for exp in experiences]).to(device)

        rewards = torch.LongTensor([exp.reward for exp in experiences]).to(device)
        
        is_dones = torch.LongTensor([exp.is_done for exp in experiences]).to(device)

        next_states = states.detach()
        # delete first line, add zeros line at the end
        next_states = torch.cat((next_states[1:], torch.zeros(next_states.shape[1], device=device)[None,...]), dim=0)
        
        total_rewards = torch.FloatTensor([ep.total_reward for ep in episodes]).to(device)

        train_batch = cls(states, actions, rewards, next_states, is_dones, total_rewards=total_rewards)
        # ========================
        return train_batch

    @property
    def num_episodes(self):
        return torch.numel(self.total_rewards)

    ''''''def __repr__(self):
        return (
            f"TrainBatch(states: {self.states.shape}, "
            f"actions: {self.actions.shape}, "
            f"q_vals: {self.q_vals.shape}), "
            f"num_episodes: {self.num_episodes})"
        )''''''

    def __len__(self):
        # number of experiences
        return self.states.shape[0]
'''

class ReplayMemory(object):

    def __init__(self, capacity, batch_size, gamma=0.99, off_policy=True):
        self.memory = deque([],maxlen=capacity) if off_policy else []
        self.batch_size = batch_size
        self.gamma = gamma
        self.off_policy = off_policy
        self.experiences = []

    def push_experience(self, exp: Experience):
        self.experiences.append(exp)
    
    def push_episode(self):
        next_qval = 0
        for i in range(len(self.experiences)-1, -1, -1):
            exp = self.experiences[i]
            next_qval = exp.reward + self.gamma * next_qval
            self.experiences[i] = exp._replace(qval = next_qval)
        for exp in self.experiences:
            self.memory.append(exp)
        self.experiences = []

    def get_batch(self) ->List[Experience]:
        size = len(self.memory)
        if self.off_policy:
            if size < self.batch_size:
                raise Exception(f"Memory is too small for sampling ({size})")
            else:
                yield random.sample(self.memory, self.batch_size)
        elif size>0:
            if self.batch_size==-1:
                yield self.memory
            else:
                for ind in range(size//self.batch_size):
                    yield self.memory[ind*self.batch_size:(ind+1)*self.batch_size]
                if size%self.batch_size > 0:
                    yield self.memory[-(size%self.batch_size):]
            self.memory = []

