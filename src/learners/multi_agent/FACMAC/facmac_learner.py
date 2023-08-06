import torch as th
import copy
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner
from modules.mixers import mixer_factory



class FACMACLearner(BaseActorCriticLearner):

    def create_mixers(self, args):
        self.mixer = mixer_factory.build(args.mixer, **{'args' : args})
        self.critic_params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)


    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()

        target_actions = self._compute_target_actions(batch, t_env)
        target_qs = self._compute_target_qs(batch, target_actions).squeeze(-1)
        qs = self._compute_qs(batch).squeeze(-1)

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
        }

        return loss, critic_metrics


    def _compute_target_actions(self, batch, t_env):
        target_actions = []
        self.target_actor.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            # test_mode to avoid adding noise
            target_actions_t = self.target_actor.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
            target_actions.append(target_actions_t)
        
        return th.stack(target_actions, dim=1)


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
        self.critic.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length - 1):
            ind_qs = self.critic.forward(batch, t, actions[:, t].detach())
            joint_qs = self.mixer(ind_qs, batch["state"][:, t:t + 1])
            qs.append(joint_qs)
        
        return th.stack(qs, dim=1)


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        mask_elems = mask.sum().item()

        actions = []
        qs = []

        self.actor.init_hidden(batch.batch_size)
        self.critic.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            actions_t = self.actor.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
            ind_qs = self.critic.forward(batch, t, actions_t)
            joint_qs = self.mixer(ind_qs, batch["state"][:, t:t+1])
            
            actions.append(actions_t)
            qs.append(joint_qs)
        
        actions = th.stack(actions[:-1], dim=1)
        qs = th.stack(qs[:-1], dim=1)

        actions_regularization_weight = 1e-3
        if not self.args.actions_regularization:
            actions_regularization_weight = 0

        loss_q_term = (qs.reshape(-1, 1) * mask.reshape(-1, 1)).sum() / mask_elems

        loss = - loss_q_term + (actions**2).mean() * actions_regularization_weight

        actor_metrics = {
            'actor_loss' : loss.item(),
            'actor_actions_mean' : actions.mean().item(),
            'actor_qs_mean' : - loss_q_term.item()
        }

        return loss, actor_metrics
    