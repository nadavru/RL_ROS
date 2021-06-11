import torch
import torch.utils.data
from typing import List, NamedTuple
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

