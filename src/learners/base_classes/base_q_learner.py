import copy
import torch as th
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from components.standarize_stream import RunningMeanStd
from controllers import REGISTRY as agent_registry
from modules.mixers import mixer_factory




class BaseQLearner:
    
    def __init__(self, scheme, logger, args):
        self.scheme = scheme
        self.args = args
        self.logger = logger
        self._create_agents(scheme, args)
        self._create_mixers(args)
        self._create_optimizer(args)
        self._create_standarizers(args)
        self.last_log_step = -self.args.learner_log_interval - 1
        self.last_hard_update_step = -self.args.hard_update_interval - 1
        self.training_steps = 0


    def _create_agents(self, scheme, args):
        self.agent = agent_registry[args.mac](scheme, args)
        self.target_agent = copy.deepcopy(self.agent)
        self.agent_params = list(self.agent.parameters())


    def _create_mixers(self, args):
        self.mixer = None
        self.target_mixer = None


    def _create_optimizer(self, args):
        self.optimizer = Adam(params=self.agent_params, lr=args.lr, weight_decay=args.l2_reg_coef, eps=args.optimizer_epsilon)
        if args.lr_decay:
            self.agent_scheduler = StepLR(self.optimizer, step_size=args.lr_decay_episodes, gamma=args.lr_decay_gamma)


    def _create_standarizers(self, args):
        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(args.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)


    def train(self, batch, t_env):
        agent_metrics = self.train_agent(batch, t_env)
        self.update_targets()
        self.training_steps += 1
        if self.is_time_to_log(t_env):
            self.log_training_stats(agent_metrics, t_env)
            self.last_log_step = t_env


    def train_agent(self, batch, t_env):
        agent_loss, agent_metrics = self.compute_agent_loss(batch, t_env)
        self.optimizer.zero_grad()
        agent_loss.backward()
        
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        agent_metrics['agent_grad_norm'] = agent_grad_norm.item()
        
        self.optimizer.step()
        if self.args.lr_decay:
            self.agent_scheduler.step()

        return agent_metrics
    

    def compute_agent_loss(self, batch, t_env):
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
        self.target_agent.load_state(self.agent)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())


    def update_targets_soft(self, tau):
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def cuda(self):
        self.agent.cuda()
        self.target_agent.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()


    def save_models(self, path):
        self.agent.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.optimizer.state_dict(), f"{path}/opt.th")


    def load_models(self, path):
        self.agent.load_models(path)
        if not self.args.evaluate:
            self.target_agent.load_models(path)
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
            self.optimizer.load_state_dict(th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
