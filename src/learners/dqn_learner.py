import numpy as np
import torch as th
from learners.base_classes.base_q_learner import BaseQLearner



class DQNLearner(BaseQLearner):
    

    def compute_agent_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_elems = mask.sum().item()

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        chosen_action_qs, qs = self._compute_qs(batch)
        target_max_qs = self._compute_target_qs(batch, qs)
        
        if self.args.standardise_returns:
            target_max_qs = (target_max_qs - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qs.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = (chosen_action_qs - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        agent_metrics = {
            'loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'q_taken_mean' : (chosen_action_qs * mask).sum().item() / (mask_elems * self.args.n_agents),
            'target_mean' : (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
        }
        return loss, agent_metrics
    

    def _compute_qs(self, batch):
        qs = []
        self.agent.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            qs_t = self.agent.forward(batch, t=t)
            qs.append(qs_t)
        # (b, t, n_discrete_actions ** n_agents)
        qs = th.stack(qs, dim=1)

        # Action for each agent (b, t, n_agents)
        actions = batch["actions"][:, :-1].squeeze(-1)
        # Index of those actions (b, t, 1)
        fold_actions = self._get_action_index(actions)
        # (b, t, 1)
        chosen_action_qs = th.gather(qs[:, :-1], dim=-1, index=fold_actions)
        return chosen_action_qs, qs
    

    def _compute_target_qs(self, batch, qs):
        target_qs = []
        self.target_agent.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_qs_t = self.target_agent.forward(batch, t=t)
            target_qs.append(target_qs_t)
        # (b, t, n_discrete_actions ** n_agents)
        target_qs = th.stack(target_qs[1:], dim=1)

        if self.args.double_q:
            # Compute max using the current network, but evaluate using the target one
            qs_detach = qs.clone().detach()
            # (b, t, 1)
            max_actions = qs_detach[:, 1:].max(dim=-1, keepdim=True)[1]
            # (b, t, 1)
            target_max_qs = th.gather(target_qs, -1, max_actions)
        else:
            # (b, t, 1)
            target_max_qs = target_qs.max(dim=-1, keepdim=True)[0]
        
        return target_max_qs


    def _shifting(self, bitlist):
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit
        return out


    def _get_action_index(self, actions):
        if self.args.use_cuda:
            actions = actions.cpu()

        int_actions = np.apply_along_axis(self._shifting, -1, actions)
        int_actions = np.expand_dims(int_actions, axis=-1)
        return th.tensor(int_actions, dtype=int, device=self.args.device)
    