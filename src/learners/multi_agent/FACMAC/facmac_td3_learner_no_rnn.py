import torch as th
import copy
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner
from modules.mixers import mixer_factory



class FACMACTD3LearnerNoRNN(BaseActorCriticLearner):


    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        self.update_actor_targets_freq = args.update_actor_targets_freq


    def create_mixers(self, args):
        self.mixer_1 = mixer_factory.build(args.mixer, **{'args' : args})
        self.mixer_2 = mixer_factory.build(args.mixer, **{'args' : args})
        self.critic_params += list(self.mixer_1.parameters())
        self.critic_params += list(self.mixer_2.parameters())
        self.target_mixer_1 = copy.deepcopy(self.mixer_1)
        self.target_mixer_2 = copy.deepcopy(self.mixer_2)


    def train(self, batch, t_env):
        critic_metrics = self.train_critic(batch, t_env)

        actor_metrics = {}
        if self.training_steps % self.update_actor_targets_freq == 0:
            actor_metrics = self.train_actor(batch, t_env)
            self.update_targets()

        self.training_steps += 1

        if self.is_time_to_log(t_env):
            metrics = {**critic_metrics, **actor_metrics}
            self.log_training_stats(metrics, t_env)
            self.last_log_step = t_env


    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()

        if self.args.use_training_steps_to_compute_target_noise:
            step = self.training_steps
        else:
            step = t_env

        target_actions = self.target_actor.select_target_actions(batch, 1, step)
        target_qs = self._compute_target_qs(batch, target_actions)
        qs_1, qs_2 = self._compute_qs(batch)

        targets = rewards + self.args.gamma * (1 - terminated) * target_qs.detach()

        mask = mask.expand_as(targets)
        mask_elems = mask.sum().item()

        td_error_1 = targets.detach() - qs_1
        masked_td_error_1 = td_error_1 * mask
        loss_1 = (masked_td_error_1 ** 2).sum() / mask_elems

        td_error_2 = targets.detach() - qs_2
        masked_td_error_2 = td_error_2 * mask
        loss_2 = (masked_td_error_2 ** 2).sum() / mask_elems
        
        loss = loss_1 + loss_2

        critic_metrics = {
            'critic_loss' : loss.item(),
            'td_error_1_abs' : masked_td_error_1.abs().sum().item() / mask_elems,
            'td_error_2_abs' : masked_td_error_2.abs().sum().item() / mask_elems,
            'target_mean' : (targets * mask).sum().item() / mask_elems,
        }

        return loss, critic_metrics


    def _compute_target_qs(self, batch, target_actions):
        target_1_qs, target_2_qs = self.target_critic.forward(batch, 1, target_actions.detach())
        joint_targets_1 = self.target_mixer_1(target_1_qs, batch["state"][:, 1:2])
        joint_targets_2 = self.target_mixer_2(target_2_qs, batch["state"][:, 1:2])
        min_targets = th.min(joint_targets_1, joint_targets_2)
        return min_targets


    def _compute_qs(self, batch):
        actions = batch["actions"][:, 0]
        ind_qs_1, ind_qs_2 = self.critic.forward(batch, 0, actions.detach())    
        joint_qs_1 = self.mixer_1(ind_qs_1, batch["state"][:, 0:1])
        joint_qs_2 = self.mixer_2(ind_qs_2, batch["state"][:, 0:1])
        return joint_qs_1, joint_qs_2


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()

        actions = self.actor.forward(batch, 0)

        ind_qs_1 = self.critic.forward_first(batch, 0, actions)

        if self.args.update_actor_with_joint_qs:
            qs_1 = self.mixer_1(ind_qs_1, batch["state"][:, 0:1])
        else:
            qs_1 = ind_qs_1

        mask = mask.expand_as(qs_1)
        mask_elems = mask.sum().item()

        actions_regularization_weight = 1e-3
        if not self.args.actions_regularization:
            actions_regularization_weight = 0

        loss_q_term = (qs_1 * mask).sum() / mask_elems

        loss = - loss_q_term + (actions**2).mean() * actions_regularization_weight

        actor_metrics = {
            'actor_loss' : loss.item(),
            'actor_actions_mean' : actions.mean().item(),
            'actor_qs_mean' : (ind_qs_1 * mask).sum().item() / mask_elems
        }

        return loss, actor_metrics
    

    def update_targets_hard(self):
        self.target_actor.load_state(self.actor)
        self.target_critic.load_state(self.critic)
        self.target_mixer_1.load_state_dict(self.mixer_1.state_dict())
        self.target_mixer_2.load_state_dict(self.mixer_2.state_dict())


    def update_targets_soft(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_mixer_1.parameters(), self.mixer_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_mixer_2.parameters(), self.mixer_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def cuda(self):
        self.actor.cuda()
        self.target_actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        self.mixer_1.cuda()
        self.target_mixer_1.cuda()
        self.mixer_2.cuda()
        self.target_mixer_2.cuda()


    def save_models(self, path):
        self.actor.save_models(path)
        self.target_actor.save_models(path, is_target=True)
        self.critic.save_models(path)
        self.target_critic.save_models(path, is_target=True)
        th.save(self.mixer_1.state_dict(), f"{path}/mixer_1.th")
        th.save(self.mixer_2.state_dict(), f"{path}/mixer_2.th")
        th.save(self.actor_optimiser.state_dict(), f"{path}/actor_opt.th")
        th.save(self.critic_optimiser.state_dict(), f"{path}/critic_opt.th")


    def load_models(self, path):
        self.actor.load_models(path)
        if not self.args.evaluate:
            self.target_actor.load_models(path, is_target=False)
            self.critic.load_models(path)
            self.target_critic.load_models(path, is_target=False)
            #self.mixer_1.load_state_dict(th.load(f"{path}/mixer_1.th", map_location=lambda storage, loc: storage))
            #self.mixer_2.load_state_dict(th.load(f"{path}/mixer_2.th", map_location=lambda storage, loc: storage))
            #self.actor_optimiser.load_state_dict(th.load(f"{path}/actor_opt.th", map_location=lambda storage, loc: storage))
            #self.critic_optimiser.load_state_dict(th.load(f"{path}/critic_opt.th", map_location=lambda storage, loc: storage))
