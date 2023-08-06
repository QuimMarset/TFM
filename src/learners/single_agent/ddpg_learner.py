import torch as th
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner



class DDPGLearner(BaseActorCriticLearner):

    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        self.n_agents = args.n_agents
        self.action_shape = args.action_shape


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        mask_elems = mask.sum().item()
        batch_size = batch.batch_size

        self.actor.init_hidden(batch_size)
        self.critic.init_hidden(batch.batch_size)
        
        qs = []

        for t in range(batch.max_seq_length - 1):
            # (b, 1, action_shape * n_agents)
            actions_t = self.actor.select_actions_train(batch, t_ep=t, t_env=t_env)
            # (b, 1, 1)
            qs_t = self.critic.forward(batch, t, actions_t)
            qs.append(qs_t)

        # (b, t, 1, 1)
        qs = th.stack(qs, dim=1)

        loss = - (qs.reshape(-1, 1) * mask.reshape(-1, 1)).sum() / mask_elems

        actor_metrics = {
            'actor_loss' : loss.item(),
        }

        return loss, actor_metrics
    

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask_elems = mask.sum().item()

        target_actions = self._compute_target_actions(batch, t_env)
        target_qs = self._compute_target_qs(batch, target_actions).squeeze(-1)
        qs = self._compute_qs(batch).squeeze(-1)

        targets = rewards + self.args.gamma * (1 - terminated) * target_qs
        
        td_error = (targets.detach() - qs)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask_elems

        critic_metrics = {
            'critic_loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'target_mean' : (targets * mask).sum().item() / mask_elems
        }

        return loss, critic_metrics


    def _compute_target_actions(self, batch, t_env):
        target_actions = []
        self.target_actor.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            # (b, 1, n_agents * action_shape)
            target_actions_t = self.target_actor.select_actions_train(batch, t_ep=t, t_env=t_env)
            target_actions.append(target_actions_t)

        # (b, t, 1, n_agents * action_shape)
        target_actions = th.stack(target_actions, dim=1)
        return target_actions


    def _compute_target_qs(self, batch, target_actions):
        target_qs = []
        self.target_critic.init_hidden(batch.batch_size)

        for t in range(1, batch.max_seq_length):
            # (b, 1, 1)
            target_qs_t = self.target_critic.forward(batch, t, target_actions[:, t].detach())
            target_qs.append(target_qs_t)

        # (b, t, 1, 1)
        return th.stack(target_qs, dim=1)


    def _compute_qs(self, batch):
        qs = []
        self.critic.init_hidden(batch.batch_size)

        # (b, t, n_agents, action_shape)
        actions = batch["actions"][:, :-1]
        # (b, t, 1, n_agents * action_shape)
        actions = actions.view(batch.batch_size, -1, 1, self.args.action_shape * self.n_agents)
        
        for t in range(batch.max_seq_length - 1):
            # (b, 1, 1)
            qs_t = self.critic.forward(batch, t, actions[:, t].detach())
            qs.append(qs_t)
        
        # (b, t, 1, 1)
        return th.stack(qs, dim=1)
    