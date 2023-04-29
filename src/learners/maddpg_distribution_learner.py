import torch as th
from learners.maddpg_learner import MADDPGLearner



class MADDPGDistributionLearner(MADDPGLearner):


    def compute_actor_loss(self, batch, t_env):
        terminated = batch["terminated"][:, :-1].float()
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        mask = 1 - terminated
        batch_size = batch.batch_size
        self.actor.init_hidden(batch_size)
        # No init hidden in the critic as it is not supposed to use RNN (but it could)
        
        actions = []
        log_probs = []
        for t in range(batch.max_seq_length - 1):
            actions_t, log_probs_t = self.actor.select_actions_with_log_probs(batch, t_ep=t)
            actions.append(actions_t)
            log_probs.append(log_probs_t)

        actions = th.stack(actions, dim=1)
        # (b, t, n_agents, -1) -> (b, t, n_agents, n_agents * -1)
        actions = actions.view(batch_size, -1, 1, self.n_agents * self.args.action_shape).expand(-1, -1, self.n_agents, -1)
        log_probs = th.stack(log_probs, dim=1).reshape(-1, 1)

        # Same actions but detaching the gradient of those belonging to a different agent
        new_actions = []
        for i in range(self.n_agents):
            temp_action = th.split(actions[:, :, i, :], self.args.action_shape, dim=2)
            actions_i = []
            for j in range(self.n_agents):
                if i == j:
                    actions_i.append(temp_action[j])
                else:
                    actions_i.append(temp_action[j].detach())
            actions_i = th.cat(actions_i, dim=-1)
            new_actions.append(actions_i.unsqueeze(2))
        new_actions = th.cat(new_actions, dim=2)

        qs = self.critic.forward(batch, new_actions).reshape(-1, 1)
        loss = - (qs * mask.reshape(-1, 1)).mean() + 0.2 * log_probs.mean()

        actor_metrics = {
            'actor_loss' : loss.item(),
        }
        return loss, actor_metrics
