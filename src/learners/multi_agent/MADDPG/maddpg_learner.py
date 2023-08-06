import torch as th
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner



class MADDPGLearner(BaseActorCriticLearner):

    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        self.n_agents = args.n_agents
        self.action_shape = args.action_shape


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        
        actions = self.actor.forward(batch, 0)
        actions = self._replicate_actions(actions)
        actions = self._detach_other_agent_actions(actions)

        qs = self.critic.forward(batch, 0, actions)

        mask = mask.expand_as(qs)
        mask_elems = mask.sum().item()

        loss = - (qs * mask).sum() / mask_elems

        actor_metrics = {
            'actor_loss' : loss.item(),
        }
        return loss, actor_metrics
    

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()

        actions = self._replicate_actions(actions)
        
        target_actions = self.actor.forward(batch, 1)
        target_actions = self._replicate_actions(target_actions)

        target_qs = self.target_critic.forward(batch, 1, target_actions.detach())
        qs = self.critic.forward(batch, 0, actions.detach())

        targets = rewards.expand_as(target_qs) + self.args.gamma * (1 - terminated.expand_as(target_qs)) * target_qs.detach()

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
    
    
    def _replicate_actions(self, actions):
        # (b, n_agents, action_shape) -> (b, n_agents, n_agents * action_shape)
        batch_size = actions.shape[0]
        actions = actions.view(batch_size, 1, self.n_agents * self.action_shape)
        return actions.expand(-1, self.n_agents, -1)
    

    def _detach_other_agent_actions(self, actions):
        detached_actions = []
        
        for i in range(self.n_agents):

            temp_action = th.split(actions[:, i, :], self.action_shape, dim=1)
            actions_i = []
            
            for j in range(self.n_agents):
                if i == j:
                    actions_i.append(temp_action[j])
                else:
                    actions_i.append(temp_action[j].detach())
            
            actions_i = th.cat(actions_i, dim=-1)
            detached_actions.append(actions_i.unsqueeze(1))
        
        return th.cat(detached_actions, dim=1)