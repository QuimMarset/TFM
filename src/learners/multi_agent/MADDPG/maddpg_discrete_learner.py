import torch as th
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner
from components.td_lambda import build_td_lambda_targets



class MADDPGDiscreteLearner(BaseActorCriticLearner):


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        actions_buffer = batch["actions_onehot"]
        loss = 0

        actions_policy = self._compute_actions(batch)

        for i in range(self.args.n_agents):
            actions = self._combine_actions(actions_buffer, actions_policy, i)
            qs_i = self.critic.forward(batch, actions)[:, :-1].squeeze(-1)

            mask_elems = mask.sum().item()

            loss += - (qs_i * mask).sum() / mask_elems

        loss /= self.args.n_agents

        actor_metrics = {
            'actor_loss' : loss.item(),
        }
        return loss, actor_metrics
    

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"]
        terminated = batch["terminated"].float()
        actions = batch["actions_onehot"]
        mask = batch["filled"].float()

        target_actions = self._compute_target_actions(batch)
        target_qs = self.target_critic.forward(batch, target_actions.detach()).squeeze(-1)
        qs = self.critic.forward(batch, actions.detach()).squeeze(-1)

        if self.args.use_td_lambda:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_qs, self.args.gamma,
                                              self.args.td_lambda)
        else:
            targets = rewards[:, :-1] + self.args.gamma * (1 - terminated[:, :-1]) * target_qs[:, 1:].detach()

        mask_elems = mask[:, :-1].sum().item()

        td_error = qs[:, :-1] - targets.detach()
        masked_td_error = td_error * mask[:, :-1]
        loss = (masked_td_error ** 2).sum() / mask_elems

        critic_metrics = {
            'critic_loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'qs_mean' : (qs[:, :-1] * mask[:, :-1]).sum().item() / mask_elems,
            'target_mean' : (targets * mask[:, :-1]).sum().item() / mask_elems
        }
        return loss, critic_metrics
    

    def _compute_target_actions(self, batch):
        self.target_actor.init_hidden(batch.batch_size)
        target_actions = []

        for t in range(batch.max_seq_length):
            target_actions_t = self.target_actor.select_target_actions(batch, t)
            target_actions.append(target_actions_t)
        
        # (b, num_transitions + 1, n_agents, n_discrete_actions)
        target_actions = th.stack(target_actions, dim=1)
        return target_actions
    

    def _compute_actions(self, batch):
        self.actor.init_hidden(batch.batch_size)
        actions = []

        for t in range(batch.max_seq_length):
            actions_t = self.actor.select_train_actions(batch, t)
            actions.append(actions_t)
        
        # (b, num_transitions + 1, n_agents, n_discrete_actions)
        actions = th.stack(actions, dim=1)
        return actions
    

    def _combine_actions(self, actions_buffer, actions_policy, agent_index):
        # Both actions -> (b, num_transitions + 1, n_agents, n_discrete_actions)
        actions = actions_buffer.detach().clone()
        actions[:, :, agent_index] = actions_policy[:, :, agent_index]
        # (b, num_transitions + 1, n_agents, n_discrete_actions)
        return actions
