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
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # Set network to train mode
        self.agent.agent.train()

        qvals, hidden_states = self._compute_qvalues(batch)
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(qvals[:, :-1], dim=3, index=actions).squeeze(3)

        target_qvals, target_hidden_states = self._compute_target_qvalues(batch)

        target_max_qvals = self._compute_max_target_qvalues(qvals, target_qvals)
        joined_target_max_qvals = self._compute_joined_target_max_qvalues(batch, target_max_qvals, 
                                                                         target_hidden_states)
            
        if getattr(self.args, 'q_lambda', False):
            qvals = th.gather(target_qvals, 3, batch["actions"]).squeeze(3)
            qvals = self.target_mixer(qvals, batch["state"], batch["obs"])

            targets = build_q_lambda_targets(rewards, terminated, mask, joined_target_max_qvals, 
                                             qvals, self.args.gamma, self.args.td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, joined_target_max_qvals, 
                                              self.args.n_agents, self.args.gamma, self.args.td_lambda)

        joined_chosen_action_qvals = self._compute_joined_chosen_qvalues(batch, chosen_action_qvals, hidden_states)

        td_error = (joined_chosen_action_qvals - targets.detach())
        td_error = 0.5 * td_error.pow(2)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        loss = masked_td_error.sum() / mask.sum()

        mask_elems = mask.sum().item()
        metrics = {
            'loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'q_taken_mean' : (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
            'target_mean' : (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
        }
        return loss, metrics            


    def _compute_qvalues(self, batch):
        self.agent.init_hidden(batch.batch_size)
        qvals = []
        hidden_states = []

        for t in range(batch.max_seq_length):
            # (batch_size, n_agents, n_actions)
            qvals_t, hidden_states_t = self.agent.forward(batch, t=t, return_hs=True)
            qvals.append(qvals_t)
            hidden_states.append(hidden_states_t)

        qvals = th.stack(qvals, dim=1)
        hidden_states = th.stack(hidden_states, dim=1)
        return qvals, hidden_states
    

    def _compute_target_qvalues(self, batch):
        self.target_agent.init_hidden(batch.batch_size)
        target_qvals = []
        target_hidden_states = []

        for t in range(batch.max_seq_length):
            target_qvals_t, target_hidden_states_t = self.target_agent.forward(batch, t=t, return_hs=True)
            target_qvals.append(target_qvals_t)
            target_hidden_states.append(target_hidden_states_t)

        target_qvals = th.stack(target_qvals, dim=1)
        target_hidden_states = th.stack(target_hidden_states, dim=1)
        return target_qvals, target_hidden_states
    

    def _compute_max_target_qvalues(self, qvals, target_qvals):
        # Max over target Q-Values with Double q learning
        qvals_detach = qvals.clone().detach()
        cur_max_actions = qvals_detach.max(dim=3, keepdim=True)[1]
         # (batch_size, max_seq_length, n_agents)
        target_max_qvals = th.gather(target_qvals, 3, cur_max_actions).squeeze(3)
        return target_max_qvals
    

    def _compute_joined_target_max_qvalues(self, batch, target_max_qvals, target_hidden_states):
        hyper_weights = self.target_mixer.init_hidden().expand(batch.batch_size, -1, -1)
        
        joined_target_max_qvals = []

        for t in range(batch.max_seq_length):

            target_mixer_out, hyper_weights = self.target_mixer(
                target_max_qvals[:, t].view(-1, 1, self.args.n_agents), # (batch, 1, n_agents)
                target_hidden_states[:, t],
                hyper_weights,
                batch["state"][:, t],
                batch["obs"][:, t]
            )
            joined_target_max_qvals.append(target_mixer_out.squeeze(-1))
        
        joined_target_max_qvals = th.stack(joined_target_max_qvals, dim=1)
        return joined_target_max_qvals
    

    def _compute_joined_chosen_qvalues(self, batch, chosen_qvals, hidden_states):
        hyper_weights = self.mixer.init_hidden().expand(batch.batch_size, -1, -1)

        joined_chosen_qvals = []

        for t in range(batch.max_seq_length - 1):

            mixer_out, hyper_weights = self.mixer(
                chosen_qvals[:, t].view(-1, 1, self.args.n_agents),
                hidden_states[:, t,].detach(),
                hyper_weights,
                batch["state"][:, t],
                batch["obs"][:, t]
            )
            
            joined_chosen_qvals.append(mixer_out.squeeze(-1))
        
        joined_chosen_qvals = th.stack(joined_chosen_qvals, dim=1)
        return joined_chosen_qvals