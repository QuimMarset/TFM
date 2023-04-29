from modules.agents import agent_factory
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th



class BasicMAC:

    # This multi-agent controller works for discrete actions

    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        if args.action_selector is not None:
            self.action_selector = action_REGISTRY[args.action_selector](args)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], t_env,
                                                            test_mode=test_mode)
        # (b, n_agents, 1)
        chosen_actions = chosen_actions.unsqueeze(-1)
        return chosen_actions


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


    def named_parameters(self):
        return self.agent.named_parameters()


    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())


    def load_state_from_state_dict(self, state_dict):
        self.agent.load_state_dict(state_dict)


    def cuda(self, device="cuda"):
        self.agent.cuda(device=device)


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
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape


    def save_models(self, path):
        th.save(self.agent.state_dict(), f'{path}/agent.th')


    def load_models(self, path):
        self.agent.load_state_dict(th.load(f'{path}/agent.th', 
                                           map_location=lambda storage, 
                                           loc: storage))
        