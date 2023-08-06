import copy
import torch as th
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner
from modules.mixers import mixer_factory
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets



class ContinuousTransformerLearner(BaseActorCriticLearner):


    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        
        # In this method, the critic and mixer are handled together
        # Thus, the critic objects maintain the mixer+critic
        # And the mixer objects are set to None

    
    def create_critics(self, scheme, args):
        kwargs = {
            'args' : args,
        }
        self.critic = mixer_factory.build(args.mixer, **kwargs)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())


    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # Set network to train mode
        self.actor.agent.train()

        qvals = self._compute_critic_joined_qvalues(batch)
        target_qvals = self._compute_critic_target_joined_qvalues(batch)

        targets = rewards + self.args.gamma * (1 - terminated) * target_qvals
        td_error = targets.detach() - qvals
        
        mask = mask.expand_as(td_error)
        mask_elems = mask.sum().item()
        
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        metrics = {
            'critic_loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'target_mean' : targets.sum().item() / mask_elems
        }
        return loss, metrics            


    def _compute_critic_joined_qvalues(self, batch):
        actions = batch["actions"][:, :-1]
        self.actor.init_hidden(batch.batch_size)
        hyper_weights = self.critic.init_hidden(batch.batch_size)
        qvals = []

        for t in range(batch.max_seq_length - 1):

            _, actor_hidden_states = self.actor.forward(batch, t=t, return_hs=True)
            mixer_out, hyper_weights = self.critic(
                actions[:, t].detach(),
                actor_hidden_states.detach(),
                hyper_weights,
                batch["state"][:, t],
                batch["obs"][:, t])
            qvals.append(mixer_out)

        qvals = th.stack(qvals, dim=1)
        return qvals.view(batch.batch_size, -1, 1)


    def _compute_critic_target_joined_qvalues(self, batch):
        self.target_actor.init_hidden(batch.batch_size)
        hyper_weights = self.target_critic.init_hidden(batch.batch_size)
        target_qvals = []

        for t in range(1, batch.max_seq_length):
            target_actions, target_actor_hidden_states = \
                self.target_actor.forward(batch, t=t, return_hs=True)
            
            target_mixer_out, hyper_weights = self.target_critic(
                target_actions.detach(),
                target_actor_hidden_states.detach(),
                hyper_weights,
                batch["state"][:, t],
                batch["obs"][:, t])
            
            target_qvals.append(target_mixer_out)
        
        target_qvals = th.stack(target_qvals, dim=1)
        return target_qvals.view(batch.batch_size, -1, 1)


    def compute_actor_loss(self, batch, t_env):
        self.actor.init_hidden(batch.batch_size)
        hyper_weights = self.critic.init_hidden(batch.batch_size)
        chosen_action_qvals = []
        actions = []
        
        for t in range(batch.max_seq_length):
            actions_t, actor_hidden_states = self.actor.forward(batch, t=t, return_hs=True)
            mixer_out, hyper_weights = self.critic(
                actions_t,
                actor_hidden_states.detach(),
                hyper_weights,
                batch["state"][:, t],
                batch["obs"][:, t])
            
            chosen_action_qvals.append(mixer_out)
            actions.append(actions_t)

        actions = th.stack(actions[:-1], dim=1)
        chosen_action_qvals = th.stack(chosen_action_qvals[:-1], dim=1)

        actions_regularization_weight = 1e-3
        if not self.args.actions_regularization:
            actions_regularization_weight = 0

        loss = -chosen_action_qvals.mean() + (actions**2).mean() * actions_regularization_weight

        actor_metrics = {
            'actor_loss' : loss.item(),
            'actor_actions_mean' : actions.mean().item(),
            'actor_qs_mean' : chosen_action_qvals.mean().item()
        }
        return loss, actor_metrics
    