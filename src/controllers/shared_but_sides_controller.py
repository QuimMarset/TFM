from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class SharedButSidesMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_shared = self.n_agents - 2
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        if args.action_selector is not None:
            self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions


    def forward(self, ep_batch, t, test_mode=False):
        shared_inputs, side_1_inputs, side_2_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        shared_outs, self.shared_hidden_states = self.shared_agent(shared_inputs, self.shared_hidden_states)
        side_1_outs, self.side_1_hidden_states = self.side_agent_1(side_1_inputs, self.side_1_hidden_states)
        side_2_outs, self.side_2_hidden_states = self.side_agent_2(side_2_inputs, self.side_2_hidden_states)

        if ep_batch.batch_size > 1:
            shared_outs = shared_outs.reshape((ep_batch.batch_size, self.n_shared, -1))
            side_1_outs = side_1_outs.unsqueeze(dim=1)
            side_2_outs = side_2_outs.unsqueeze(dim=1)
            agent_outs = th.cat([side_1_outs, shared_outs, side_2_outs], dim=1)
            agent_outs = agent_outs.view(ep_batch.batch_size * self.n_agents, -1)
        else:
            agent_outs = th.cat([side_1_outs, shared_outs, side_2_outs], dim=0)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)


    def init_hidden(self, batch_size):
        self.side_1_hidden_states = self.side_agent_1.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)
        self.shared_hidden_states = self.shared_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_shared, -1)
        self.side_2_hidden_states = self.side_agent_2.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)


    def parameters(self):
        parameters = []
        for agent in [self.side_agent_1, self.shared_agent, self.side_agent_2]:
            parameters.extend(agent.parameters())
        return parameters


    def load_state(self, other_mac):
        self.shared_agent.load_state_dict(other_mac.shared_agent.state_dict())
        self.side_agent_1.load_state_dict(other_mac.side_agent_1.state_dict())
        self.side_agent_2.load_state_dict(other_mac.side_agent_2.state_dict())


    def cuda(self, device='cuda'):
        self.shared_agent.cuda(device=device)
        self.side_agent_1.cuda(device=device)
        self.side_agent_2.cuda(device=device)


    def save_models(self, path):
        th.save(self.shared_agent.state_dict(), f'{path}/shared_agent.th')
        th.save(self.side_agent_1.state_dict(), f'{path}/side_agent_1.th')
        th.save(self.side_agent_2.state_dict(), f'{path}/side_agent_2.th')


    def load_models(self, path):
        self.shared_agent.load_state_dict(th.load(f'{path}/shared_agent.th', map_location=lambda storage, loc: storage))
        self.side_agent_1.load_state_dict(th.load(f'{path}/side_agent_1.th', map_location=lambda storage, loc: storage))
        self.side_agent_2.load_state_dict(th.load(f'{path}/side_agent_2.th', map_location=lambda storage, loc: storage))


    def _build_agents(self, input_shape):
        self.shared_agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        if self.args.obs_agent_id:
            input_shape -= self.n_shared
        self.side_agent_1 = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.side_agent_2 = agent_REGISTRY[self.args.agent](input_shape, self.args)


    def build_inputs_common(self, batch, t, start_index, end_index, is_shared):
        batch_size = batch.batch_size
        inputs = []
        inputs.append(batch['obs'][:, t, start_index:end_index])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch['actions_onehot'][:, t, start_index:end_index]))
            else:
                inputs.append(batch['actions_onehot'][:, t-1, start_index:end_index])

        if self.args.obs_agent_id and is_shared:
            agent_ids = th.eye(self.n_shared, device=batch.device).unsqueeze(0).expand(batch_size, -1, -1)
            inputs.append(agent_ids)

        if is_shared:
            reshape_dim = [batch_size * self.n_shared, -1]
        else:
            reshape_dim = [batch_size, -1]
        inputs = th.cat([x.reshape(*reshape_dim) for x in inputs], dim=-1)
        return inputs


    def _build_inputs_shared(self, batch, t):
        return self.build_inputs_common(batch, t, 1, -1, is_shared=True)


    def _build_inputs_side_1(self, batch, t):
        return self.build_inputs_common(batch, t, 0, 1, is_shared=False)


    def _build_inputs_side_2(self, batch, t):
        return self.build_inputs_common(batch, t, -1, self.n_agents, is_shared=False)


    def _build_inputs(self, batch, t):
        shared_inputs = self._build_inputs_shared(batch, t)
        side_1_inputs = self._build_inputs_side_1(batch, t)
        side_2_inputs = self._build_inputs_side_2(batch, t)
        return shared_inputs, side_1_inputs, side_2_inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_shared
        return input_shape