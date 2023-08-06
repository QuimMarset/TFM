import copy
import torch as th
from modules.mixers import mixer_factory
from learners.base_classes.base_q_learner import BaseQLearner



class ContinuousQLearner(BaseQLearner):

    def _create_mixers(self, args):
        kwargs = {
            'args' : args,
        }
        self.mixer = mixer_factory.build(args.mixer, **kwargs)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.agent_params += list(self.mixer.parameters())


    def compute_agent_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_elems = mask.sum().item()

        qs = self._compute_qs(batch)
        target_actions = self._compute_target_actions(batch, t_env)
        target_qs = self._compute_target_qs(batch, target_actions)

        targets = rewards.expand_as(target_qs) + self.args.gamma * (1 - terminated.expand_as(target_qs)) * target_qs

        td_error = (qs - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        agent_metrics = {
            'agent_loss' : loss.item(),
            'weight_norm' : (th.sum(th.cat([th.sum(p**2).unsqueeze(0) for p in self.agent_params]))**0.5).item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems, 
            'q_taken_mean' : (qs * mask).sum().item() / (mask_elems * self.args.n_agents),
            'target_mean' : (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
            'td_error_abs' : masked_td_error.abs().sum().item() / batch.batch_size
        }
        return loss, agent_metrics
    

    def _compute_qs(self, batch):
        ind_qs = []
        self.agent.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            qs_t = self.agent.forward(batch, t, actions=batch["actions"][:, t].detach())
            ind_qs.append(qs_t)
        ind_qs = th.stack(ind_qs[:-1], dim=1)
        # (b, t, n_agents, 1) -> (b, t, 1, 1)
        joint_qs = self.mixer(ind_qs.view(-1, self.args.n_agents, 1), batch["state"][:, :-1])
        return joint_qs


    def _compute_target_actions(self, batch, t_env):
        target_actions = []
        self.target_agent.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_actions_t = self.target_agent.select_actions(batch, t, t_env, test_mode=True)
            target_actions.append(target_actions_t)
        target_actions = th.stack(target_actions, dim=1)
        return target_actions


    def _compute_target_qs(self, batch, target_actions):
        ind_target_qs = []
        self.target_agent.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_qs_t = self.target_agent.forward(batch, t, actions=target_actions[:, t].detach())
            ind_target_qs.append(target_qs_t)
        ind_target_qs = th.stack(ind_target_qs[1:], dim=1)
        # (b, t, n_agents, 1) -> (b, t, 1, 1)
        joint_target_qs = self.mixer(ind_target_qs.view(-1, self.args.n_agents, 1), batch["state"][:, :-1])
        return joint_target_qs
    