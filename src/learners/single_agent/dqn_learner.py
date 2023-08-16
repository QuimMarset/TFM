import numpy as np
import torch as th
from components.standarize_stream import RunningMeanStd
from learners.base_classes.base_q_learner import BaseQLearner



class DQNLearner(BaseQLearner):

    """
        We are dealing with a single-agent method that uses the environment state as input
        Thus, we do not use recurrent layers, and we can sample random transitions
    """

    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        self._create_standarizers(args)


    def _create_standarizers(self, args):
        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardize_returns:
            self.return_standarizer = RunningMeanStd(shape=(1,), device=device)
        if self.args.standardize_rewards:
            self.reward_standarizer = RunningMeanStd(shape=(1,), device=device)
    

    def compute_agent_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask_elems = mask.sum().item()

        if self.args.standardize_rewards:
            self.reward_standarizer.update(rewards)
            rewards = (rewards - self.reward_standarizer.mean) / th.sqrt(self.reward_standarizer.var)

        chosen_action_qs = self._compute_qs(batch)
        # (b, 1, n_discrete_actions ** n_agents)
        target_qs = self.target_agent.forward(batch, 1)
        max_target_qs = self._compute_max_target_qs(batch, target_qs)

        if self.args.standardize_returns:
            max_target_qs = (max_target_qs - self.return_standarizer.mean) / th.sqrt(self.return_standarizer.var)

        targets = rewards + self.args.gamma * (1 - terminated) * max_target_qs.detach()

        if self.args.standardize_returns:
            self.return_standarizer.update(targets)
            targets = (targets - self.return_standarizer.mean) / th.sqrt(self.return_standarizer.var)

        td_error = (chosen_action_qs - targets.detach())
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask_elems

        agent_metrics = {
            'loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'q_taken_mean' : (chosen_action_qs * mask).sum().item() / mask_elems,
            'target_mean' : (targets * mask).sum().item() / mask_elems
        }
        return loss, agent_metrics
    

    def _compute_qs(self, batch):
        self.agent.init_hidden(batch.batch_size)
        qs = []
        # (b, 1, n_agents)
        binary_actions = batch["actions"][:, :-1].squeeze(-1)

        # (b, 1, n_discrete_actions ** n_agents)
        qs = self.agent.forward(batch, 0)

        # (b, 1, 1)
        decimal_actions = self._binary_to_decimal(binary_actions).detach()
        # (b, 1, 1)
        chosen_action_qs = th.gather(qs, dim=-1, index=decimal_actions)
        return chosen_action_qs
    

    def _compute_max_target_qs(self, batch, target_qs):
        if self.args.double_q:
            # Compute max using the current network, but evaluate using the target one
            # (b, 1, 1)
            max_actions = self.agent.forward(batch, 1).detach().max(dim=-1, keepdim=True)[1]
            # (b, 1, 1)
            target_max_qs = th.gather(target_qs, -1, index=max_actions)
        else:
            # (b, 1, 1)
            target_max_qs = target_qs.max(dim=-1, keepdim=True)[0]
        
        return target_max_qs.detach()


    def _shifting(self, bitlist):
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit
        return out


    def _binary_to_decimal(self, binary_actions):        
        if self.args.use_cuda:
            binary_actions = binary_actions.cpu()

        decimal_actions = np.apply_along_axis(self._shifting, -1, binary_actions.numpy())
        decimal_actions = np.expand_dims(decimal_actions, axis=-1)
        return th.tensor(decimal_actions, dtype=int, device=self.args.device)
    