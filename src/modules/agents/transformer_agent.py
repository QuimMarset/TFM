import torch.nn as nn
import torch as th
from modules.layers.transformer import Transformer



class TransformerAgent(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        # get the number of entities for the agent if specified, otherwise use n_entities
        self.n_entities = getattr(args, "n_entities_obs", self.args.n_entities)
        self.feat_dim = args.obs_entity_feats
        self.emb_dim = args.emb_dim

        # Embedding of the entity features
        self.feat_embedding = nn.Linear(self.feat_dim, self.emb_dim)

        # transformer block
        self.transformer = Transformer(self.emb_dim, args.num_heads, 
                                       args.num_blocks, args.ff_hidden_mult, args.dropout)

        self.q_basic = nn.Linear(self.emb_dim, args.n_net_outputs)


    def init_hidden(self):
        # Hidden state is used as the [CLS] token in BERT
        return th.zeros(1, self.emb_dim).to(self.args.device)


    def forward(self, inputs, hidden_state):
        batch_size = inputs.size(0)
        n_agents = inputs.size(1)
        
        # inputs has 3 dimensions (b, n_agents, n_entities * feat_dim)
        inputs  = inputs.view(-1, self.n_entities, self.feat_dim)
        # hidden_state has 3 dimensions (b, n_agents, emb_dim)
        hidden_state = hidden_state.view(-1, 1, self.emb_dim)

        # Project the embeddings
        embs = self.feat_embedding(inputs)

        # The transformer queries and keys are the input embeddings plus the hidden state
        x = th.cat((hidden_state, embs), dim=1)

        # Get the transformer embeddings
        embs = self.transformer.forward(x, x)

        # Extract the current hidden state
        h = embs[:, 0:1, :]

        # Get the q values
        q = self.q_basic(h)

        return q.view(batch_size, n_agents, -1), h.view(batch_size, n_agents, -1)
