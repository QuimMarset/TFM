import torch as th
from components.standarize_stream import RunningMeanStd
from learners.base_classes.base_actor_critic_learner import BaseActorCriticLearner



class MADDPGLearner(BaseActorCriticLearner):

    def __init__(self, scheme, logger, args):
        super().__init__(scheme, logger, args)
        self.n_agents = args.n_agents
        self.action_shape = args.action_shape
        self._create_standarizers(args)


    def _create_standarizers(self, args):
        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)


    def compute_actor_loss(self, batch, t_env):
        terminated = batch["terminated"][:, :-1].float()
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        mask = 1 - terminated
        batch_size = batch.batch_size
        self.actor.init_hidden(batch_size)
        
        actions = []
        for t in range(batch.max_seq_length - 1):
            actions_t = self.actor.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
            actions.append(actions_t)

        actions = th.stack(actions, dim=1)
        # (b, t, n_agents, -1) -> (b, t, n_agents, n_agents * -1)
        actions = actions.view(batch_size, -1, 1, self.n_agents * self.action_shape).expand(-1, -1, self.n_agents, -1)

        # Same actions but detaching the gradient of those belonging to a different agent
        new_actions = []
        for i in range(self.n_agents):
            temp_action = th.split(actions[:, :, i, :], self.action_shape, dim=2)
            actions_i = []
            for j in range(self.n_agents):
                if i == j:
                    actions_i.append(temp_action[j])
                else:
                    actions_i.append(temp_action[j].detach())
            actions_i = th.cat(actions_i, dim=-1)
            new_actions.append(actions_i.unsqueeze(2))
        new_actions = th.cat(new_actions, dim=2)

        q = self.critic.forward(batch, new_actions).reshape(-1, 1)
        loss = - (q * mask.reshape(-1, 1)).mean()

        actor_metrics = {
            'actor_loss' : loss.item(),
        }
        return loss, actor_metrics
    

    def compute_critic_loss(self, batch, t_env):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"]
        terminated = batch["terminated"][:, :-1].float()
        rewards = rewards.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        mask = 1 - terminated
        mask_elems = mask.sum().item()
        batch_size = batch.batch_size

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        actions = actions.view(batch_size, -1, 1, self.n_agents * self.action_shape).expand(-1, -1, self.n_agents, -1)
        q_taken = self.critic.forward(batch, actions[:, :-1].detach())

        # Use the target actor and target critic network to compute the target q
        target_actions = self._compute_target_actions(batch, t_env)
        target_vals = self.target_critic.forward(batch, target_actions.detach())

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = rewards.reshape(-1, 1) + self.args.gamma * (1 - terminated.reshape(-1, 1)) * target_vals.reshape(-1, 1).detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = (q_taken.view(-1, 1) - targets.detach())
        masked_td_error = td_error * mask.reshape(-1, 1)
        loss = (masked_td_error ** 2).mean()

        critic_metrics = {
            'critic_loss' : loss.item(),
            'td_error_abs' : masked_td_error.abs().sum().item() / mask_elems,
            'q_taken_mean' : q_taken.sum().item() / mask_elems,
            'target_mean' : targets.sum().item() / mask_elems
        }
        return loss, critic_metrics
    

    def _compute_target_actions(self, batch, t_env):
        self.target_actor.init_hidden(batch.batch_size)
        target_actions = []
        for t in range(1, batch.max_seq_length):
            agent_target_outs = self.target_actor.select_actions(batch, t, t_env, test_mode=True)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time

        target_actions = target_actions.view(batch.batch_size, -1, 1, 
                                             self.n_agents * self.action_shape).expand(-1, -1, self.n_agents, -1)
        return target_actions
