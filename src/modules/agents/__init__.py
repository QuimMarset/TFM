REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .linear_agent import LinearAgent
from .mlp_agent_actor import MLPActorAgent
from .rnn_agent_actor import RNNActorAgent
from .comix_agent import CEMAgent, CEMRecurrentAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY['linear'] = LinearAgent
REGISTRY['rnn_actor'] = RNNActorAgent
REGISTRY['mlp_actor'] = MLPActorAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["cemrnn"] = CEMRecurrentAgent