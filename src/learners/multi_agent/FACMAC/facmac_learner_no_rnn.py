import copy
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner
from modules.mixers import mixer_factory



class FACMACLearnerNoRNN(BaseActorCriticLearner):


    def create_mixers(self, args):
        self.mixer = mixer_factory.build(args.mixer, **{'args' : args})
        self.critic_params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)


    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()

        target_actions = self.target_actor.forward(batch, 1)
        target_qs = self._compute_target_qs(batch, target_actions)
        qs = self._compute_qs(batch)

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


    def _compute_target_qs(self, batch, target_actions):
        ind_target_qs = self.target_critic.forward(batch, 1, target_actions.detach())
        joint_target_qs = self.target_mixer(ind_target_qs, batch['state'][:, 1:2])
        return joint_target_qs.view(batch.batch_size, 1, 1)


    def _compute_qs(self, batch):
        actions = batch['actions'][:, 0]
        ind_qs = self.critic.forward(batch, 0, actions)
        joint_qs = self.mixer(ind_qs, batch["state"][:, 0:1])
        return joint_qs.view(batch.batch_size, 1, 1)


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        mask_elems = mask.sum().item()

        actions = self.actor.forward(batch, 0)

        ind_qs = self.critic.forward(batch, 0, actions)

        if self.args.update_actor_with_joint_qs:
            qs = self.mixer(ind_qs, batch["state"][:, 0:1])
        else:
            qs = ind_qs
           
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
    