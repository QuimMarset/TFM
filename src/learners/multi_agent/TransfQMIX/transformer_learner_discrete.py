import copy
import torch as th
from learners.base_classes.base_q_learner import BaseQLearner
from modules.mixers import mixer_factory
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets



class DiscreteTransformerLearner(BaseQLearner):


    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        
    
    def _create_mixers(self, args):
        kwargs = {
            'args' : args,
        }
        self.mixer = mixer_factory.build(args.mixer, **kwargs)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.agent_params += list(self.mixer.parameters())


    def compute_agent_loss(self, batch, t_env):
        rewards = batch["reward"]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask_elems = mask[:, :-1].sum().item()
        
        # Set network to train mode
        self.agent.agent.train()

        qs, hidden_states = self._compute_qs(batch)
        # (b, episode_length, n_agents)
        chosen_action_qs = th.gather(qs[:, :-1], dim=3, index=actions).squeeze(3)
        # (b, episode_length, 1)
        joined_chosen_action_qs = self._compute_joined_chosen_qs(batch, chosen_action_qs, hidden_states)

        target_qs, target_hidden_states = self._compute_target_qs(batch)
        # (b, episode_length + 1, n_agents)
        max_target_qs = self._compute_max_target_qs(qs, target_qs)
        # (b, episode_length + 1, 1)
        joined_max_target_qs = self._compute_joined_max_target_qs(batch, max_target_qs, target_hidden_states)
        
        # (b, episode_length, 1)
        targets = build_td_lambda_targets(rewards, terminated, mask, joined_max_target_qs, 
                                          self.args.n_agents, self.args.gamma, self.args.td_lambda)

        td_error = (joined_chosen_action_qs - targets.detach())
        masked_td_error = td_error * mask[:, :-1]
        loss = (masked_td_error ** 2).sum() / mask_elems

        metrics = {
            'loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'q_taken_mean' : (joined_chosen_action_qs * mask[:, :-1]).sum().item() / mask_elems,
            'target_mean' : (targets * mask[:, :-1]).sum().item() / mask_elems
        }
        return loss, metrics            


    def _compute_qs(self, batch):
        self.agent.init_hidden(batch.batch_size)
        qs = []
        hidden_states = []

        for t in range(batch.max_seq_length):
            qs_t, hidden_states_t = self.agent.forward(batch, t=t, return_hs=True)
            qs.append(qs_t)
            hidden_states.append(hidden_states_t)

        # (b, episode_length + 1, n_agents, n_discrete_actions), (b, episode_length + 1, n_agents, emb_dim)
        return th.stack(qs, dim=1), th.stack(hidden_states, dim=1)
    

    def _compute_target_qs(self, batch):
        self.target_agent.init_hidden(batch.batch_size)
        target_qs = []
        target_hidden_states = []

        for t in range(batch.max_seq_length):
            target_qs_t, target_hidden_states_t = self.target_agent.forward(batch, t=t, return_hs=True)
            target_qs.append(target_qs_t)
            target_hidden_states.append(target_hidden_states_t)

        # (b, episode_length + 1, n_agents, n_discrete_actions), (b, episode_length + 1, n_agents, emb_dim)
        return th.stack(target_qs, dim=1), th.stack(target_hidden_states, dim=1)
    

    def _compute_max_target_qs(self, qvals, target_qs):
        # Max over target Q-Values with Double Q-learning
        qvals_detach = qvals.clone().detach()
        # (b, episode_length + 1, n_agents, 1)
        max_actions = qvals_detach.max(dim=3, keepdim=True)[1]
        # (b, episode_length + 1, n_agents)
        target_max_qvals = th.gather(target_qs, dim=-1, index=max_actions).squeeze(3)
        return target_max_qvals
    

    def _compute_joined_max_target_qs(self, batch, target_max_qs, target_hidden_states):
        # (b, 3, emb_dim)
        hyper_weights = self.target_mixer.init_hidden().expand(batch.batch_size, -1, -1)
        
        joined_target_max_qs = []

        for t in range(batch.max_seq_length):
            # (b, 1, 1), (b, 1, emb_dim)
            target_mixer_out, hyper_weights = self.target_mixer(
                # (b, 1, n_agents)
                target_max_qs[:, t].view(-1, 1, self.args.n_agents), 
                target_hidden_states[:, t],
                hyper_weights,
                batch["state"][:, t],
                batch["obs"][:, t]
            )
            joined_target_max_qs.append(target_mixer_out.squeeze(-1))
        
        # (b, episode_length + 1, 1)
        return th.stack(joined_target_max_qs, dim=1)
    

    def _compute_joined_chosen_qs(self, batch, chosen_qs, hidden_states):
        # (b, 3, emb_dim)
        hyper_weights = self.mixer.init_hidden().expand(batch.batch_size, -1, -1)

        joined_chosen_qs = []

        for t in range(batch.max_seq_length - 1):
            # (b, 1, 1), (b, 1, emb_dim)
            mixer_out, hyper_weights = self.mixer(
                # (b, 1, n_agents)
                chosen_qs[:, t].view(-1, 1, self.args.n_agents),
                hidden_states[:, t,].detach(),
                hyper_weights,
                batch["state"][:, t],
                batch["obs"][:, t]
            )
            
            joined_chosen_qs.append(mixer_out.squeeze(-1))
        
        # (b, episode_length, 1)
        return th.stack(joined_chosen_qs, dim=1)