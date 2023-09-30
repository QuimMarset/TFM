import torch as th
from learners.single_agent.ddpg_learner import DDPGLearner



class TD3Learner(DDPGLearner):


    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        self.update_actor_targets_freq = args.update_actor_targets_freq


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


    def compute_actor_loss(self, batch, t_env):
        mask = batch["filled"][:, :-1].float()
        mask_elems = mask.sum().item()
        
        actions = self.actor.select_actions_train(batch, 0)

        qs = self.critic.forward_first(batch, 0, actions)

        loss = - (qs * mask).sum() / mask_elems

        actor_metrics = {
            'actor_loss' : loss.item(),
        }

        return loss, actor_metrics
    

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask_elems = mask.sum().item()

        if self.args.use_training_steps_to_compute_target_noise:
            step = self.training_steps
        else:
            step = t_env
        
        target_actions = self.target_actor.select_target_actions(batch, 1, step)
        target_1_qs, target_2_qs = self.target_critic.forward(batch, 1, target_actions.detach())

        actions = batch["actions"][:, 0]
        actions = actions.view(batch.batch_size, 1, self.action_shape * self.n_agents)
        
        qs_1, qs_2 = self.critic.forward(batch, 0, actions.detach())
        
        min_targets = th.min(target_1_qs, target_2_qs)
        targets = rewards + self.args.gamma * (1 - terminated) * min_targets.detach()

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
            'target_mean' : (targets * mask).sum().item() / mask_elems
        }

        return loss, critic_metrics
    