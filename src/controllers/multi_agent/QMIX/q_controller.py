import torch as th
from components.action_selectors import REGISTRY as action_selector_registry
from controllers.base_classes.base_controller import BaseController



class QController(BaseController):

    # This multi-agent controller works for discrete actions -> QMIX and VDN

    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.action_selector = action_selector_registry[args.action_selector](args)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], t_env,
                                                            test_mode=test_mode)
        # (b, n_agents, 1)
        chosen_actions = chosen_actions.unsqueeze(-1)
        return chosen_actions


    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        # (b, n_agents, n_discrete_actions), (b, n_agents, hidden_dim)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs


    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.add_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])

        if self.args.add_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        if self.args.add_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if self.args.add_agent_id:
            input_shape += self.n_agents

        return input_shape
        