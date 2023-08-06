import torch as th
import copy
from learners.multi_agent.FACMAC.facmac_learner import FACMACLearner



class FACMACDistributionLearner(FACMACLearner):


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        mask_elems = mask.sum().item()

        actions = self.actor.forward_train(batch, 0)
        ind_qs = self.critic.forward(batch, 0, actions)
        joint_qs = self.mixer(ind_qs, batch["state"][:, 0:1])
       
        loss_q_term = (joint_qs * mask).sum() / mask_elems
        
        #loss_entropy_term = (entropy * entropy_mask).sum() / entropy_mask.sum()
        #entropy_ratio = self.args.entropy_coef / loss_entropy_term.item()

        loss = - loss_q_term

        actor_metrics = {
            'actor_loss' : loss.item(),
            'actor_qs_mean' : loss_q_term.item(),
            #'actor_entropy_mean' : loss_entropy_term.item()
        }
        return loss, actor_metrics
    

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()

        target_actions = self.target_actor.forward_train(batch, 1)
        target_qs = self._compute_target_qs(batch, target_actions)
        qs = self._compute_qs(batch)

        targets = rewards + self.args.gamma * (1 - terminated) * target_qs.detach()
        
        mask_elems = mask.sum().item()

        td_error = targets.detach() - qs
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask_elems

        critic_metrics = {
            'critic_loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'target_mean' : (targets * mask).sum().item() / mask_elems,
        }

        return loss, critic_metrics


    def _compute_target_qs(self, batch, target_actions):
        ind_target_qs = self.target_critic.forward(batch, 1, target_actions.detach())
        joint_target_qs = self.target_mixer(ind_target_qs, batch['state'][:, 1:2])
        return joint_target_qs.view(batch.batch_size, 1, 1)


    def _compute_qs(self, batch):
        actions = batch['actions'][:, 0]
        ind_qs = self.critic.forward(batch, 0, actions)
        joint_qs = self.mixer(ind_qs, batch["state"][:, 0:1])
        return joint_qs.view(batch.batch_size, 1, 1)
