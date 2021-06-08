import torch
import torch.nn as nn
from data import TrainBatch

class VanillaPolicyGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: TrainBatch, action_scores: torch.Tensor, **kw):
        """
        Calculates the policy gradient loss function.
        :param batch: A TrainBatch of experiences, shape (N,).
        :param action_scores: The scores (not probabilities) for all possible
        actions at each experience in the batch, shape (N, A).
        :return: A tuple of the loss and a dict for printing.
        """
        #  Calculate the loss.
        #  Use the helper methods in this class to first calculate the weights
        #  and then the loss using the weights and action scores.
        # ====== YOUR CODE: ======
        loss_p = self._policy_loss(
            batch=batch,
            action_scores=action_scores,
            policy_weight=batch.q_vals
        )
        # ========================
        return loss_p

    def _policy_loss(self, batch, action_scores, policy_weight):
        #   - Calculate log-probabilities of the actions.
        #   - Select only the log-proba of actions that were actually taken.
        #   - Calculate the weighted average using the given weights.
        #   - Helpful methods: log_softmax() and gather().
        #   Note that our batch is "flat" i.e. it doesn't separate between
        #   multiple episodes, but simply stores (s,a,r,) experiences from
        #   different episodes. So, here we'll simply average over the number
        #   of total experiences in our batch.
        # ====== YOUR CODE: ======
        log_action_proba = nn.functional.log_softmax(action_scores, dim=1)
        selected_action_log_proba = log_action_proba.gather(dim=1, index=batch.actions.unsqueeze(dim=1)).squeeze(dim=1)
        weighted_avg_experience_rewards = selected_action_log_proba * policy_weight
        loss_p = -weighted_avg_experience_rewards.mean()
        # ========================
        return loss_p


class BaselinePolicyGradientLoss(VanillaPolicyGradientLoss):
    def forward(self, batch: TrainBatch, action_scores: torch.Tensor, **kw):
        """
        Calculates the baseline policy gradient loss function.
        :param batch: A TrainBatch of experiences, shape (N,).
        :param action_scores: The scores (not probabilities) for all possible
        actions at each experience in the batch, shape (N, A).
        :return: A tuple of the loss and a dict for printing.
        """
        #  Calculate the loss and baseline.
        #  Use the helper methods in this class as before.
        # ====== YOUR CODE: ======
        log_action_proba = nn.functional.log_softmax(action_scores, dim=1)
        selected_action_log_proba = log_action_proba.gather(dim=1, index=batch.actions.unsqueeze(dim=1)).squeeze(dim=1)
        policy_weight, baseline = self._policy_weight(batch=batch)
        weighted_avg_experience_rewards = selected_action_log_proba * policy_weight
        loss_p = -weighted_avg_experience_rewards.mean()
        # ========================
        return loss_p, dict(loss_p=loss_p.item(), baseline=baseline.item())

    def _policy_weight(self, batch: TrainBatch):
        #  Calculate both the policy weight term and the baseline value for
        #  the PG loss with baseline.
        # ====== YOUR CODE: ======
        policy_weight = batch.q_vals.clone()
        baseline = policy_weight.mean()
        policy_weight -= baseline
        # ========================
        return policy_weight, baseline


class ActionEntropyLoss(nn.Module):
    def __init__(self, n_actions, beta=1.0):
        """
        :param n_actions: Number of possible actions.
        :param beta: Factor to apply to the loss (a hyperparameter).
        """
        super().__init__()
        self.max_entropy = self.calc_max_entropy(n_actions)
        self.beta = beta
    
    def to(self, device):
        self.max_entropy = self.max_entropy.to(device)

    @staticmethod
    def calc_max_entropy(n_actions):
        """
        Calculates the maximal possible entropy value for a given number of
        possible actions.
        """
        max_entropy = None
        # ====== YOUR CODE: ======
        def entropy(prob: int):
            return -prob * torch.log(torch.Tensor([prob])) * n_actions

        max_entropy = entropy(prob=1/n_actions)
        # ========================
        return max_entropy

    def forward(self, batch: TrainBatch, action_scores, **kw):
        """
        Calculates the entropy loss.
        :param batch: A TrainBatch containing N experiences.
        :param action_scores: The scores for each of A possible actions
        at each experience in the batch, shape (N, A).
        :return: A tuple of the loss and a dict for printing.
        """
        if isinstance(action_scores, tuple):
            # handle case of multiple return values from model; we assume
            # scores are the first element in this case.
            action_scores, _ = action_scores

        #   Notes:
        #   - Use self.max_entropy to normalize the entropy to [0,1].
        #   - Notice that we want to maximize entropy, not minimize it.
        #     Make sure minimizing your returned loss with SGD will maximize
        #     the entropy.
        #   - Use pytorch built-in softmax and log_softmax.
        #   - Calculate loss per experience and average over all of them.
        # ====== YOUR CODE: ======
        def entropy(action_scores: torch.Tensor): # returns tensor with shape (N,)
            return (action_scores.softmax(dim=1) * action_scores.log_softmax(dim=1)).sum(dim=1)

        model_entropies = entropy(action_scores=action_scores)
        mean_entropy = (model_entropies / self.max_entropy).mean(dim=0)
        loss_e = mean_entropy
        # ========================

        loss_e *= self.beta
        return loss_e, dict(loss_e=loss_e.item())


class BaselineEntropyPolicyGradientLoss(nn.Module):
    def __init__(self, n_actions, beta=1.0):
        """
        :param n_actions: Number of possible actions.
        :param beta: Factor to apply to the loss (a hyperparameter).
        """
        super().__init__()
        self.entropy_loss = ActionEntropyLoss(n_actions, beta)
        self.baseline_loss = BaselinePolicyGradientLoss()
    
    def to(self, device):
        self.entropy_loss.to(device)
    
    def forward(self, batch: TrainBatch, action_scores: torch.Tensor, **kw):
        """
        Calculates the baseline policy gradient loss function.
        :param batch: A TrainBatch of experiences, shape (N,).
        :param action_scores: The scores (not probabilities) for all possible
        actions at each experience in the batch, shape (N, A).
        :return: A tuple of the loss and a dict for printing.
        """
        #  Calculate the loss and baseline.
        #  Use the helper methods in this class as before.
        # ====== YOUR CODE: ======
        
        loss_e, _ = self.entropy_loss(batch, action_scores, **kw)
        loss_p, dictionary = self.baseline_loss(batch, action_scores, **kw)
        dictionary["loss_e"]=loss_e.item()
        loss = loss_e + loss_p

        return loss, dictionary