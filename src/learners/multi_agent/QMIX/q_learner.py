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
        agent_qs_all_actions, agent_qs = self._compute_agent_qs(batch)
        # (b, episode_length, n_agents, n_discrete_actions)
        agent_target_qs = self._compute_agent_target_qs(batch)
        # (b, episode_length, n_agents)
        agent_max_target_qs = self._compute_agent_max_target_qs(agent_qs_all_actions.clone().detach(), agent_target_qs)

        if self.mixer is not None:
            # Both (b, episode_length, 1)
            qs = self.mixer(agent_qs, batch["state"][:, :-1])
            max_target_qs = self.target_mixer(agent_max_target_qs, batch["state"][:, 1:])
        else:
            # Both (b, episode_length, n_agents)
            qs = agent_qs
            max_target_qs = agent_max_target_qs
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


    def _compute_agent_target_qs(self, batch):
        agent_target_qs = []
        self.target_agent.init_hidden(batch.batch_size)
        
        for t in range(1, batch.max_seq_length):
            agent_target_qs_t = self.target_agent.forward(batch, t=t)
            agent_target_qs.append(agent_target_qs_t)
        
        # (b, episode_length, n_agents, n_discrete_actions)
        return th.stack(agent_target_qs, dim=1)


    def _compute_agent_qs(self, batch):
        agent_qs = []
        # (b, episode_length, n_agents, 1)
        actions = batch["actions"][:, :-1]
        self.agent.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            # (b, n_agents, n_discrete_actions)
            agent_qs_t = self.agent.forward(batch, t=t)
            agent_qs.append(agent_qs_t)
        
        # (b, episode_length + 1, n_agents, n_discrete_actions)
        agent_qs = th.stack(agent_qs, dim=1)
        # (b, episode_length, n_agents)
        chosen_agent_qs = th.gather(agent_qs[:, :-1], dim=3, index=actions).squeeze(3)
        return agent_qs, chosen_agent_qs
    

    def _compute_agent_max_target_qs(self, agent_qs_all_actions, agent_target_qs):
        
        if self.args.double_q:
            # Compute max using the current network, but evaluate using the target one
            # max returns also the agent indices that maximize the Qs, hence the max actions
            max_actions = agent_qs_all_actions[:, 1:].max(dim=3, keepdim=True)[1]
            # (b, episode_length, n_agents)
            agent_target_max_qs = th.gather(agent_target_qs, 3, max_actions).squeeze(3)
        else:
            # (b, episode_length, n_agents)
            agent_target_max_qs = agent_target_qs.max(dim=3)[0]

        return agent_target_max_qs
    