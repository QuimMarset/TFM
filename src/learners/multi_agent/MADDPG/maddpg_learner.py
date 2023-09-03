import torch as th
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner



class MADDPGLearner(BaseActorCriticLearner):

    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        self.n_agents = args.n_agents
        self.action_shape = args.action_shape


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        actions_buffer = batch["actions"]
        loss_q_term = 0

        actions_policy = self._compute_actions(batch)

        for i in range(self.n_agents):
            actions = self._combine_actions(actions_buffer, actions_policy, i)
            qs_i = self.critic.forward(batch, actions).squeeze(-1)[:, :-1]

            mask = mask.expand_as(qs_i)
            mask_elems = mask.sum().item()

            loss_q_term += - (qs_i * mask).sum() / mask_elems

        regularization_weight = 1e-3
        if not self.args.actions_regularization:
            regularization_weight = 0

        regularization_term = regularization_weight * (actions_policy[:, :-1]**2).mean()

        loss = loss_q_term / self.n_agents + regularization_term

        actor_metrics = {
            'actor_loss' : loss.item(),
        }
        return loss, actor_metrics
    

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        actions = batch["actions"]
        
        target_actions = self._compute_target_actions(batch)
        target_qs = self.target_critic.forward(batch, target_actions.detach()).squeeze(-1)
        qs = self.critic.forward(batch, actions.detach()).squeeze(-1)

        targets = rewards + self.args.gamma * (1 - terminated) * target_qs[:, 1:].detach()

        mask = mask.expand_as(targets)
        mask_elems = mask.sum().item()

        td_error = (qs[:, :-1] - targets.detach())
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask_elems

        critic_metrics = {
            'critic_loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'qs_mean' : (qs[:, :-1] * mask).sum().item() / mask_elems,
            'target_mean' : (targets * mask).sum().item() / mask_elems,
        }
        return loss, critic_metrics
    

    def _compute_target_actions(self, batch):
        self.target_actor.init_hidden(batch.batch_size)
        target_actions = []

        for t in range(batch.max_seq_length):
            target_actions_t = self.target_actor.forward(batch, t)
            target_actions.append(target_actions_t)
        
        # (b, num_transitions + 1, n_agents, n_discrete_actions)
        target_actions = th.stack(target_actions, dim=1)
        return target_actions
    

    def _compute_actions(self, batch):
        self.actor.init_hidden(batch.batch_size)
        actions = []

        for t in range(batch.max_seq_length):
            actions_t = self.actor.forward(batch, t)
            actions.append(actions_t)
        
        # (b, num_transitions + 1, n_agents, n_discrete_actions)
        actions = th.stack(actions, dim=1)
        return actions
    

    def _combine_actions(self, actions_buffer, actions_policy, agent_index):
        # Both actions -> (b, num_transitions + 1, n_agents, n_discrete_actions)
        actions = []

        for i in range(self.args.n_agents):

            if i != agent_index:
                actions_i = actions_buffer[:, :, i:i+1].detach().clone()
            else:
                actions_i = actions_policy[:, :, i:i+1]

            actions.append(actions_i)
        
        # (b, num_transitions + 1, n_agents, n_discrete_actions)
        actions = th.cat(actions, dim=2)
        return actions