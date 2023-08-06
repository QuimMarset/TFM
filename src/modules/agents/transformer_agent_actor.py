import torch.nn as nn
import torch as th
from modules.layers.transformer import Transformer


class TransformerActorAgent(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_entities = getattr(self.args, "n_entities_obs", self.args.n_entities) 
        
        self.feat_dim = args.obs_entity_feats
        self.emb_dim = args.emb_dim
        self.use_entity_embeddings = getattr(args, "use_entity_embeddings", False)
        self.entity_emb_dim = getattr(args, "entity_emb_dim", 2)

        self.feat_embedding = nn.Linear(
            self.feat_dim + (self.entity_emb_dim if self.use_entity_embeddings else 0),
            self.emb_dim
        )

        # Adding agent-specific embeddings
        if self.use_entity_embeddings:
            # Assuming max of 100 entities
            self.entity_embedder = nn.Embedding(100,  self.entity_emb_dim)

        self.transformer = Transformer(args.emb_dim, args.num_heads, args.num_blocks, args.ff_hidden_mult, 
                                       args.dropout, getattr(args, "agent_save_memory", False))

        # Output shape is the shape of the action
        self.actions_layer = nn.Linear(args.emb_dim, args.n_net_outputs)


    def init_hidden(self):
        # Hidden state is used as the [CLS] token in BERT
        return th.zeros(1, self.emb_dim).to(self.args.device)


    def forward(self, inputs, hidden_states):
        batch_size = inputs.size(0)
        n_agents = inputs.size(1)

        # (b, n_agents, n_entities * feat_dim) -> (b * n_agents, n_entities, feat_dim)
        inputs  = inputs.view(-1, self.n_entities, self.feat_dim)
        # (b, n_agents, emb_dim) -> (b * n_agents, 1, emb_dim)
        hidden_states = hidden_states.view(-1, 1, self.emb_dim)

        if self.use_entity_embeddings:
            entity_ids = th.arange(n_agents).to(inputs.device)
            entity_embs = self.entity_embedder(entity_ids).unsqueeze(1).repeat(1, self.n_entities, 1)
            entity_embs_expanded = entity_embs.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            entity_embs_expanded = entity_embs_expanded.view(batch_size * n_agents, self.n_entities, self.entity_emb_dim)
            inputs = th.cat((inputs, entity_embs_expanded), dim=-1)

        # Project the embeddings
        embs = self.feat_embedding(inputs)

        # The transformer queries and keys are the input embeddings plus the hidden state
        x = th.cat((hidden_states, embs), 1)

        # Get the transformer embeddings
        embs = self.transformer.forward(x, x)

        # Extract the current hidden state
        h = embs[:, 0:1, :]

        # Get the actions (assuming in the range [-1, 1])
        actions = th.tanh(self.actions_layer(h))

        return actions.view(batch_size, n_agents, -1), h.view(batch_size, n_agents, -1)
