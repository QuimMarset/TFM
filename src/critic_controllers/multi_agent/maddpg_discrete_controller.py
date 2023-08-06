from critic_controllers.multi_agent.maddpg_controller import MADDPGCriticController



class MADDPGDiscreteCriticController(MADDPGCriticController):
    

    def _get_action_shape(self, scheme):
        return scheme["actions_onehot"]["vshape"][0] * self.n_agents


    def _build_actions(self, batch, actions):
        batch_size = batch.batch_size
        max_t = batch.max_seq_length - 1
        if actions is None:
            actions = batch['actions_onehot']
            # (b, t, n_agents, -1) -> (b, t, 1, n_agents * n_discrete_actions) -> (b, t, n_agents, n_agents * n_discrete_actions)
            actions = actions.view(batch_size, -1, 1, self.n_agents * self.args.n_discrete_actions).expand(-1, -1, self.n_agents, -1)
        return actions.reshape(batch_size * max_t, self.n_agents, -1)
    