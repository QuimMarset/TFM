import torch as th
import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from controllers import REGISTRY as actor_REGISTRY
from critic_controllers import REGISTRY as critic_REGISTRY



class BaseActorCriticLearner:

    def __init__(self, scheme, logger, args):
        self.args = args
        self.scheme = scheme
        self.logger = logger

        self.create_actors(scheme, args)
        self.create_critics(scheme, args)
        self.create_mixers(args)
        self._create_optimizers(args)

        self.last_log_step = -self.args.learner_log_interval - 1
        self.last_hard_update_step = -self.args.hard_update_interval - 1
        self.training_steps = 0


    def create_actors(self, scheme, args):
        self.actor = actor_REGISTRY[args.mac](scheme, args)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_params = list(self.actor.parameters())


    def create_critics(self, scheme, args):
        self.critic = critic_REGISTRY[args.critic_controller](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())


    def create_mixers(self, args):
        self.mixer = None
        self.target_mixer = None


    def _create_optimizers(self, args):
        self.actor_optimiser = Adam(self.actor_params, args.lr, weight_decay=args.l2_reg_coef, eps=args.optimizer_epsilon)
        self.critic_optimiser = Adam(self.critic_params, args.critic_lr, weight_decay=args.l2_reg_coef, eps=args.optimizer_epsilon)

        if args.lr_decay_actor:
            self.actor_scheduler = StepLR(self.actor_optimiser, args.lr_decay_episodes, gamma=args.lr_decay_gamma)
        if args.lr_decay_critic:
            self.critic_shceduler = StepLR(self.critic_optimiser, args.lr_decay_episodes, gamma=args.lr_decay_gamma)


    def train(self, batch, t_env):
        critic_metrics = self.train_critic(batch, t_env)
        actor_metrics = self.train_actor(batch, t_env)
        self.update_targets()
        self.training_steps += 1
        if self.is_time_to_log(t_env):
            metrics = {**critic_metrics, **actor_metrics}
            self.log_training_stats(metrics, t_env)
            self.last_log_step = t_env


    def train_critic(self, batch, t_env):
        critic_loss, critic_metrics = self.compute_critic_loss(batch, t_env)
        self.critic_optimiser.zero_grad()
        critic_loss.backward()

        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip_critic)
        critic_metrics['critic_grad_norm'] = critic_grad_norm.item()

        self.critic_optimiser.step()
        if self.args.lr_decay_critic:
            self.critic_shceduler.step()

        return critic_metrics
    

    def compute_critic_loss(self, batch, t_env):
        raise NotImplementedError()


    def train_actor(self, batch, t_env):
        actor_loss, actor_metrics = self.compute_actor_loss(batch, t_env)
        self.actor_optimiser.zero_grad()
        actor_loss.backward()

        actor_grad_norm = th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip_actor)
        actor_metrics['actor_grad_norm'] = actor_grad_norm.item()
        
        self.actor_optimiser.step()
        if self.args.lr_decay_actor:
            self.actor_scheduler.step()

        return actor_metrics
    

    def compute_actor_loss(self, batch, t_env):
        raise NotImplementedError()
    

    def is_time_to_log(self, t_env):
        return t_env - self.last_log_step >= self.args.learner_log_interval
        

    def log_training_stats(self, metrics, t_env):
        for metric_name in metrics:
            self.logger.log_stat(metric_name, metrics[metric_name], t_env)


    def update_targets(self):
        if self.args.target_update_mode == "hard":
            if self.is_time_to_hard_update():
                self.update_targets_hard()
                self.last_hard_update_step = self.training_steps
        else:
            self.update_targets_soft(self.args.target_update_tau)


    def is_time_to_hard_update(self):
        return self.training_steps - self.last_hard_update_step >= self.args.hard_update_interval


    def update_targets_hard(self):
        self.target_actor.load_state(self.actor)
        self.target_critic.load_state(self.critic)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())


    def update_targets_soft(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def cuda(self):
        self.actor.cuda()
        self.target_actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()


    def save_models(self, path):
        self.actor.save_models(path)
        self.critic.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.actor_optimiser.state_dict(), f"{path}/actor_opt.th")
        th.save(self.critic_optimiser.state_dict(), f"{path}/critic_opt.th")


    def load_models(self, path):
        self.actor.load_models(path)

        if not self.args.evaluate:
            self.critic.load_models(path)
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
            self.actor_optimiser.load_state_dict(th.load(f"{path}/actor_opt.th", map_location=lambda storage, loc: storage))
            self.critic_optimiser.load_state_dict(th.load(f"{path}/critic_opt.th", map_location=lambda storage, loc: storage))
