import copy
from .facmac_learner import FACMACLearner
from components.episode_buffer import EpisodeBatch
from controllers_critics import REGISTRY as critic_REGISTRY
import torch as th
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import StepLR
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_ablations import VDNState, QMixerNonmonotonic


class FACMACLearnerRE(FACMACLearner):

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self._create_critics(scheme, args)
        self._create_optimizers(args)
        
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_step = 0


    def _create_critics(self, scheme, args):
        self.critic = critic_REGISTRY[args.critic_controller](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1:  # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "vdn-s":
                self.mixer = VDNState(args)
            elif args.mixer == "qmix-nonmonotonic":
                self.mixer = QMixerNonmonotonic(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)


    def _create_optimizers(self, args):
        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps, weight_decay=args.l2_reg_coef)
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps, weight_decay=args.l2_reg_coef)
        
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8), weight_decay=args.l2_reg_coef)
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8), weight_decay=args.l2_reg_coef)
        
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if args.lr_decay_actor:
            self.agent_scheduler = StepLR(self.agent_optimiser, step_size=args.lr_decay_steps, gamma=args.lr_decay_gamma)
        if args.lr_decay_critic:
            self.critic_shceduler = StepLR(self.critic_optimiser, step_size=args.lr_decay_steps, gamma=args.lr_decay_gamma)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        critic_loss, critic_grad_norm, mask, targets = self.train_critic(batch)
        actor_loss, agent_grad_norm = self.train_actor(batch, t_env)
        self.update_targets()
        self.training_steps += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("target_mean", targets.sum().item() / mask_elems, t_env)
            self.logger.log_stat("pg_loss", actor_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
            self.log_stats_t = t_env


    def train_critic(self, batch):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        target_actions = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=None, test_mode=True)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time

        q_taken = []
        self.critic.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            critic_out = self.critic.forward(batch, t=t)
            if self.mixer is not None:
                critic_out = self.mixer(critic_out.view(batch.batch_size, -1, 1), batch["state"][:, t:t + 1])
            q_taken.append(critic_out)
        q_taken = th.stack(q_taken, dim=1)

        target_vals = []
        self.target_critic.init_hidden(batch.batch_size)
        for t in range(1, batch.max_seq_length):
            # Pass t=t+1 to ensure selecting the action in time-step t (A t-1 in critic's forward function)
            target_critic_out = self.target_critic.forward(batch, t=t, actions=target_actions)
            if self.mixer is not None:
                target_critic_out = self.target_mixer(target_critic_out.view(batch.batch_size, -1, 1),
                    batch["state"][:, t:t+1])
            target_vals.append(target_critic_out)
        target_vals = th.stack(target_vals, dim=1)

        if self.mixer is not None:
            q_taken = q_taken.view(batch.batch_size, -1, 1)
            target_vals = target_vals.view(batch.batch_size, -1, 1)
        else:
            q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
            target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

        targets = rewards.expand_as(target_vals) + self.args.gamma * (1 - terminated.expand_as(target_vals)) * target_vals
        td_error = (targets.detach() - q_taken)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        if self.args.lr_decay_critic:
            self.critic_shceduler.step()

        return loss, critic_grad_norm, mask, targets


    def train_actor(self, batch, t_env):
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        mac_out = []
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        self.critic.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)["actions"].view(batch.batch_size,
                self.n_agents, self.n_actions)
        
            q = self.critic.forward(batch, t=t, actions=agent_outs)
            if self.mixer is not None:
                q = self.mixer(q.view(batch.batch_size, -1, 1), batch["state"][:, t:t+1])
        
            mac_out.append(agent_outs)
            chosen_action_qvals.append(q)
        mac_out = th.stack(mac_out[:-1], dim=1)
        chosen_action_qvals = th.stack(chosen_action_qvals[:-1], dim=1)
        pi = mac_out
        
        # Compute the actor loss
        actor_weight = self.args.actor_weight
        pg_loss = -chosen_action_qvals.mean() * actor_weight + (pi**2).mean() * 1e-3

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()
        if self.args.lr_decay_actor:
            self.agent_scheduler.step()

        return pg_loss, agent_grad_norm


    def update_targets(self):
        if getattr(self.args, "target_update_mode", "hard") == "hard":
            if (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
                self._update_targets()
                self.last_target_update_step = self.training_steps
        
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
            
        else:
            raise Exception(
                "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))
        

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state(self.critic)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())