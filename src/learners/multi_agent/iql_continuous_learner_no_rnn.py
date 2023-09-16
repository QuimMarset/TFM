import copy
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner




class IQLContinuousLearnerNoRNN(BaseActorCriticLearner):

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        actions = batch['actions'][:, 0]

        target_actions = self.target_actor.forward(batch, 1)
        target_qs = self.target_critic.forward(batch, 1, target_actions.detach())
        qs = self.critic.forward(batch, 0, actions)

        rewards = rewards.expand_as(target_qs)
        terminated = terminated.expand_as(target_qs)

        targets = rewards + self.args.gamma * (1 - terminated) * target_qs.detach()
        
        mask = mask.expand_as(targets)
        mask_elems = mask.sum().item()

        td_error = targets.detach() - qs
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask_elems

        critic_metrics = {
            'critic_loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'target_mean' : (targets * mask).sum().item() / mask_elems,
            'qs_mean' : (qs * mask).sum().item() / mask_elems
        }

        return loss, critic_metrics


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        mask_elems = mask.sum().item()

        actions = self.actor.forward(batch, 0)

        qs = self.critic.forward(batch, 0, actions)
            
        actions_regularization_weight = 1e-3
        if not self.args.actions_regularization:
            actions_regularization_weight = 0

        mask = mask.expand_as(qs)
        mask_elems = mask.sum().item()

        loss_q_term = (qs * mask).sum() / mask_elems
        loss_action_term = (actions**2).mean() * actions_regularization_weight

        loss = - loss_q_term + loss_action_term

        actor_metrics = {
            'actor_loss' : loss.item(),
            'actor_actions_mean' : actions.mean().item(),
            'actor_qs_mean' : loss_q_term.item()
        }

        return loss, actor_metrics
    