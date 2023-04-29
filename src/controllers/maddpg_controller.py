from modules.agents import agent_factory
import torch as th
import numpy as np



class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*x.shape)
        self.state = x + dx
        return self.state * self.scale



class MADDPGMAC:

    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        
        shape = (self.n_agents, scheme["actions"]["vshape"][0])
        self.exploration = OUNoise(shape)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        actions = self.forward(ep_batch, t_ep)
        
        if not test_mode and t_env < self.args.stop_noise_step:
            noise = th.tensor(self.exploration.noise(), dtype=th.float32, device=ep_batch.device)
            noise = noise.unsqueeze(0).expand(ep_batch.batch_size, -1, -1)
            actions += noise
            actions = self._clamp_actions(actions)

        return actions[bs]
    

    def _clamp_actions(self, actions):
        for index in range(self.n_agents):
            action_space = self.args.action_spaces[index]
            for dimension_num in range(action_space.shape[0]):
                min_action = action_space.low[dimension_num].item()
                max_action = action_space.high[dimension_num].item()
                actions[:, index, dimension_num].clamp_(min_action, max_action)
        return actions


    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)


    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            # (1, h) -> (b, 1, h)
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, -1, -1)
            # If the MAC does not share parameters: (b, 1, h) -> (b, n_agents, h)
            if self.hidden_states.shape[1] != self.n_agents:
                self.hidden_states = self.hidden_states.expand(-1, self.n_agents, -1)


    def parameters(self):
        return self.agent.parameters()


    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())


    def cuda(self):
        self.agent.cuda()


    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))


    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))


    def _build_agents(self, input_shape):
        kwargs = {
            'input_shape' : input_shape,
            'args' : self.args
        }
        self.agent = agent_factory.build(self.args.agent, **kwargs)


    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t-1])
        
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
