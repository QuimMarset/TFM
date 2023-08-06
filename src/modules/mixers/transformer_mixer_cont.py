import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.transformer import Transformer
from modules.mixers.transformer_mixer import orthogonal_init_



class TransformerMixerContinuous(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.action_shape = args.action_shape
        # Get the number of entities for the mixer if specified, otherwise use n_entities
        self.n_entities = getattr(self.args, "n_entities_state", self.args.n_entities)
        self.feat_dim = args.state_entity_feats
        self.emb_dim = args.mixer_emb_dim
        self.use_entity_embeddings = getattr(args, "use_entity_embeddings", False)
        self.entity_emb_dim = getattr(args, "entity_emb_dim", 2)

        if self.use_entity_embeddings:
            # Assuming max of 100 entities
            self.entity_embedder = nn.Embedding(100, self.entity_emb_dim)

        # Critic
        input_dim = self.feat_dim + self.action_shape
        input_dim += self.entity_emb_dim if self.use_entity_embeddings else 0
        self.critic = nn.Sequential(
            nn.Linear(input_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
        )

        # mixer network
        self.transformer_mixer = Transformer(
            self.emb_dim,
            args.mixer_heads,
            args.mixer_depth,
            args.ff_hidden_mult,
            args.dropout,
            getattr(args,"mixer_save_memory",False)
        )
        self.hyper_b2 = nn.Linear(self.emb_dim, 1)

        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        self.custom_space = args.env_args.get("state_entity_mode", True)
        
        # the final projection over q_agents and q_tot
        self.q_agents = nn.Linear(self.emb_dim, 1)

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)


    def init_hidden(self, batch_size):
        # Hidden state used in the 3 recurrent vectors to compute the hyper-weights
        return th.zeros(batch_size, 3, self.emb_dim).to(self.args.device)
    

    def forward(self, actions, hidden_states, hyper_weights, states, obs):
        
        inputs = self._prepare_inputs(actions, states)

        # Critic
        critic_embs = self.critic(inputs[:, :self.n_agents, :])
        qvals = self.q_agents(critic_embs).view(-1, 1, self.n_agents)

        # Mixer 
        # embs = self.feat_embedding(inputs)
        x = th.cat((critic_embs.detach(), hidden_states.detach(), hyper_weights), 1)
        embs = self.transformer_mixer.forward(x, x)

        w1 = embs[:, :self.n_agents, :]
        # First bias matrix (batch_size, 1, emb) -> the first hyper_weight
        b1 = embs[:, -3, :].view(-1, 1, self.emb_dim)
        
        # Second weight matrix (batch_size, emb, 1) -> second hyper_weight
        w2 = embs[:, -2, :].view(-1, self.emb_dim, 1)
        # Second bias (scalar) (batch_size, 1, 1) -> third hyper_weight @ hyper_b2
        b2 = F.relu(self.hyper_b2(embs[:, -1, :])).view(-1, 1, 1)
        
        w1 = self.pos_func(w1)
        w2 = self.pos_func(w2)

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1) # (b, 1, emb)
        y = th.matmul(hidden, w2) + b2 # (b, 1, 1)
        
        return y, embs[:, -3:, :]


    def _prepare_inputs(self, actions, states):
        b, a, u = actions.size() 

        inputs = states.reshape(b, self.n_entities, self.feat_dim)

        # Adding agent-specific embeddings
        if self.use_entity_embeddings:
            entity_ids = th.arange(a).to(actions.device)
            entity_embs = self.entity_embedder(entity_ids).unsqueeze(0).repeat(b, 1, 1)
            inputs = th.cat((inputs, entity_embs), dim=-1)

        # Create zero tensor for non-agent entities and concatenate actions and zero_actions along the agent/entity dimension
        if self.n_entities > a:
            non_agent_entities = self.n_entities - a
            zero_actions = th.zeros(b, non_agent_entities, u, device=actions.device, dtype=actions.dtype)
            extended_actions = th.cat((actions, zero_actions), dim=1)
        else:
            extended_actions = actions

        # cat the actions
        inputs = th.cat((inputs, extended_actions), dim=-1)

        return inputs
    

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        elif self.qmix_pos_func == "abs":
            return th.abs(x)
        else:
            return x
        

    def save_models(self, path):
        th.save(self.state_dict(), f'{path}/transformer_mixer.th')