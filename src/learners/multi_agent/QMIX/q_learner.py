import copy
import torch as th
from modules.mixers import mixer_factory
from learners.base_classes.base_q_learner import BaseQLearner



class QLearner(BaseQLearner):
    
    def _create_mixers(self, args):
        if args.mixer is not None:
            kwargs = {
                'args' : args,
            }
            self.mixer = mixer_factory.build(args.mixer, **kwargs)
            self.target_mixer = copy.deepcopy(self.mixer)
            self.agent_params += list(self.mixer.parameters())
        else:
            # IQL
            self.mixer = None
            self.target_mixer = None
        

    def compute_agent_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()

        # (b, episode_length + 1, n_agents, n_discrete_actions) and
        # (b, episode_length, n_agents)
        ind_qs_all_actions, ind_qs = self._compute_ind_qs(batch)
        # (b, episode_length, n_agents, n_discrete_actions)
        ind_target_qs = self._compute_ind_target_qs(batch)
        # (b, episode_length, n_agents)
        ind_max_target_qs = self._compute_ind_max_target_qs(ind_qs_all_actions.clone().detach(), ind_target_qs)

        if self.mixer is not None:
            # Both (b, episode_length, 1)
            qs = self.mixer(ind_qs, batch["state"][:, :-1])
            max_target_qs = self.target_mixer(ind_max_target_qs, batch["state"][:, 1:])
        else:
            # Both (b, episode_length, n_agents)
            qs = ind_qs
            max_target_qs = ind_max_target_qs
            rewards = rewards.expand_as(max_target_qs)
            terminated = terminated.expand_as(max_target_qs)

        # (b, episode_length, n_agents) or (b, episode_length, 1)
        targets = rewards + self.args.gamma * (1 - terminated) * max_target_qs.detach()

        mask = mask.expand_as(targets)
        mask_elems = mask.sum().item()

        td_error = qs - targets.detach()
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask_elems

        metrics = {
            'loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'q_taken_mean' : (qs * mask).sum().item() / mask_elems,
            'target_mean' : (targets * mask).sum().item() / mask_elems
        }
        return loss, metrics


    def _compute_ind_target_qs(self, batch):
        ind_target_qs = []
        self.target_agent.init_hidden(batch.batch_size)
        
        for t in range(1, batch.max_seq_length):
            ind_target_qs_t = self.target_agent.forward(batch, t=t)
            ind_target_qs.append(ind_target_qs_t)
        
        # (b, episode_length, n_agents, n_discrete_actions)
        return th.stack(ind_target_qs, dim=1)


    def _compute_ind_qs(self, batch):
        ind_qs = []
        # (b, episode_length, n_agents, 1)
        actions = batch["actions"][:, :-1]
        self.agent.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            # (b, n_agents, n_discrete_actions)
            ind_qs_t = self.agent.forward(batch, t=t)
            ind_qs.append(ind_qs_t)
        
        # (b, episode_length + 1, n_agents, n_discrete_actions)
        ind_qs = th.stack(ind_qs, dim=1)
        # (b, episode_length, n_agents)
        chosen_ind_qs = th.gather(ind_qs[:, :-1], dim=3, index=actions).squeeze(3)
        return ind_qs, chosen_ind_qs
    

    def _compute_ind_max_target_qs(self, ind_qs_all_actions, ind_target_qs):
        
        if self.args.double_q:
            # max returns also the indices that maximize the Qs, hence the max actions
            max_actions = ind_qs_all_actions[:, 1:].max(dim=3, keepdim=True)[1]
            # (b, episode_length, n_agents)
            ind_target_max_qs = th.gather(ind_target_qs, 3, max_actions).squeeze(3)
        else:
            # (b, episode_length, n_agents)
            ind_target_max_qs = ind_target_qs.max(dim=3)[0]

        return ind_target_max_qs
    