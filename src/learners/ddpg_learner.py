import torch as th
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner



class DDPGLearner(BaseActorCriticLearner):

    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        self.n_agents = args.n_agents
        self.action_shape = args.action_shape


    def compute_actor_loss(self, batch, t_env):
        terminated = batch["terminated"][:, :-1].float()
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        mask = 1 - terminated
        batch_size = batch.batch_size

        self.actor.init_hidden(batch_size)
        self.critic.init_hidden(batch.batch_size)
        
        qs = []

        for t in range(batch.max_seq_length - 1):
            # (b, n_agents, action_shape)
            actions_t = self.actor.select_actions(batch, t_ep=t, t_env=t_env)
            # (b, 1, n_agents * action_shape)
            actions_t = actions_t.view(batch_size, 1, self.n_agents, self.args.action_shape)
            # (b, 1, 1)
            qs_t = self.critic.forward(batch, t, actions_t)
            qs.append(qs_t)

        qs = th.stack(qs, dim=1)

        loss = - (qs.reshape(-1, 1) * mask.reshape(-1, 1)).mean()

        actor_metrics = {
            'actor_loss' : loss.item(),
        }

        return loss, actor_metrics
    

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_elems = mask.sum().item()

        target_actions = self._compute_target_actions(batch, t_env)
        target_qs = self._compute_target_qs(batch, target_actions).squeeze(-1)
        qs = self._compute_qs(batch).squeeze(-1)

        targets = rewards.expand_as(target_qs) + self.args.gamma * (1 - terminated.expand_as(target_qs)) * target_qs
        td_error = (targets.detach() - qs)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()

        critic_metrics = {
            'critic_loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / batch.batch_size,
            'target_mean' : targets.sum().item() / mask_elems
        }

        return loss, critic_metrics


    def _compute_target_actions(self, batch, t_env):
        target_actions = []
        self.target_actor.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            target_actions_t = self.target_actor.select_actions(batch, t_ep=t, t_env=t_env)
            target_actions.append(target_actions_t)

        target_actions = th.stack(target_actions, dim=1)
        # (b, t, n_agents, action_shape) -> (b, t, 1, n_agents * action_shape)
        target_actions = target_actions.view(batch.batch_size, batch.max_seq_len, 
                                             1, self.args.action_shape * self.n_agents)
        return target_actions


    def _compute_target_qs(self, batch, target_actions):
        target_qs = []
        self.target_critic.init_hidden(batch.batch_size)

        for t in range(1, batch.max_seq_length):
            ind_target_qs = self.target_critic.forward(batch, t, target_actions[:, t].detach())
            joint_target_qs = self.target_mixer(ind_target_qs, batch["state"][:, t:t+1])
            target_qs.append(joint_target_qs)

        return th.stack(target_qs, dim=1)


    def _compute_qs(self, batch):
        qs = []
        actions = batch["actions"][:, :-1]
        actions = actions.view(batch.batch_size, batch.max_seq_len, 
                               1, self.args.action_shape * self.n_agents)
        self.critic.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length - 1):
            ind_qs = self.critic.forward(batch, t, actions[:, t].detach())
            joint_qs = self.mixer(ind_qs, batch["state"][:, t:t + 1])
            qs.append(joint_qs)
        
        return th.stack(qs, dim=1)
    