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
            self.mixer = None
            self.target_mixer = None
        

    def compute_agent_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        history_qs = []
        self.agent.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            history_qs_t = self.agent.forward(batch, t=t)
            history_qs.append(history_qs_t)
        history_qs = th.stack(history_qs, dim=1)
        chosen_action_qvals = th.gather(history_qs[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_history_qs = []
        self.target_agent.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_history_qs_t = self.target_agent.forward(batch, t=t)
            target_history_qs.append(target_history_qs_t)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_history_qs = th.stack(target_history_qs[1:], dim=1)

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = history_qs.clone().detach()
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_history_qs, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_history_qs.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        mask_elems = mask.sum().item()
        metrics = {
            'loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'q_taken_mean' : (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
            'target_mean' : (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
        }
        return loss, metrics