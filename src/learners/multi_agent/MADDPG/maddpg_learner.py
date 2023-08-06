import torch as th
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner



class MADDPGLearner(BaseActorCriticLearner):

    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        self.n_agents = args.n_agents
        self.action_shape = args.action_shape


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        actions_buffer = batch["actions"][:, 0]
        loss = 0

        actions_policy = self.actor.forward(batch, 0)

        for i in range(self.n_agents):
            actions = self._combine_actions(actions_buffer, actions_policy, i)
            qs_i = self.critic.forward(batch, 0, actions)

            mask = mask.expand_as(qs_i)
            mask_elems = mask.sum().item()

            loss += - (qs_i * mask).sum() / mask_elems

        loss /= self.n_agents

        actor_metrics = {
            'actor_loss' : loss.item(),
        }
        return loss, actor_metrics
    

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        actions = batch["actions"][:, 0]
        
        target_actions = self.actor.forward(batch, -1)
        target_qs = self.target_critic.forward(batch, -1, target_actions.detach())
        qs = self.critic.forward(batch, 0, actions.detach())

        targets = rewards + self.args.gamma * (1 - terminated) * target_qs.detach()

        mask = mask.expand_as(targets)
        mask_elems = mask.sum().item()

        td_error = (qs - targets.detach())
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask_elems

        critic_metrics = {
            'critic_loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'qs_mean' : (qs * mask).sum().item() / mask_elems,
            'target_mean' : (targets * mask).sum().item() / mask_elems,
        }
        return loss, critic_metrics
    

    def _combine_actions(self, actions_buffer, actions_policy, agent_index):
        # Both actions -> (b, n_agents, *action_shape)
        actions = []
        for i in range(self.n_agents):

            if i != agent_index:
                actions_i = actions_buffer[:, i:i+1].detach().clone()
            else:
                actions_i = actions_policy[:, i:i+1]

            actions.append(actions_i)
            
        actions = th.cat(actions, dim=1)
        return actions