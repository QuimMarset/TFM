from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class NonSharedMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs = []
        hidden_states = []
        for i, agent in enumerate(self.agents):
            agent_out, hidden_state = agent(agent_inputs[:, i], self.hidden_states[i])
            agent_outs.append(agent_out)
            hidden_states.append(hidden_state)
        self.hidden_states = hidden_states
        agent_outs = th.stack(agent_outs).squeeze(1)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = [agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1) for agent in self.agents]  # bav

    def parameters(self):
        parameters = []
        for agent in self.agents:
            parameters.extend(agent.parameters())
        return parameters

    def load_state(self, other_mac):
        for agent, other_agent in zip(self.agents, other_mac.agents):
            agent.load_state_dict(other_agent.state_dict())
        

    def cuda(self):
        for agent in self.agents:
            agent.cuda()

    def save_models(self, path):
        for i, agent in enumerate(self.agents):
            th.save(agent.state_dict(), f'{path}/agent_{i}.th')

    def load_models(self, path):
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(th.load(f'{path}/agent_{i}.th', map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agents = [agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(self.n_agents)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape