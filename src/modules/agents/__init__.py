from modules.common.factory import Factory
from modules.agents.mlp_agent import MLPAgent
from modules.agents.mlp_agent_actor import MLPActorAgent
from modules.agents.rnn_agent import RNNAgent
from modules.agents.rnn_agent_actor import RNNActorAgent
from modules.agents.rnn_mean_log_std_agent import RNNMeanLogStdAgent
from modules.agents.mlp_mean_log_std_agent import MLPMeanLogStdAgent
from modules.agents.shared_but_sides_agent import SharedButSidesAgent
from modules.agents.non_shared_agent import NonSharedAgent
from modules.agents.transformer_agent import TransformerAgent
from modules.agents.transformer_agent_actor import TransformerActorAgent
from modules.agents.shared_but_first_agent import SharedButFirstAgent
from modules.agents.cnn_agent_actor import CNNActorAgent
from modules.agents.mlp_agent_actor_ao import MLPActorAgentAdaptiveOptics



def build_mlp(input_shape, args, **ignored):
    return MLPAgent(input_shape, args)

def build_sides_mlp(input_shape, args, **ignored):
    return SharedButSidesAgent(input_shape, args, MLPAgent)

def build_non_shared_mlp(input_shape, args, **ignored):
    return NonSharedAgent(input_shape, args, MLPAgent)

def build_actor_mlp(input_shape, args, **ignored):
    return MLPActorAgent(input_shape, args)

def build_sides_actor_mlp(input_shape, args, **ignored):
    return SharedButSidesAgent(input_shape, args, MLPActorAgent)

def build_non_shared_actor_mlp(input_shape, args, **ignored):
    return NonSharedAgent(input_shape, args, MLPActorAgent)

def build_rnn(input_shape, args, **ignored):
    return RNNAgent(input_shape, args)

def build_sides_rnn(input_shape, args, **ignored):
    return SharedButSidesAgent(input_shape, args, RNNAgent)

def build_non_shared_rnn(input_shape, args, **ignored):
    return NonSharedAgent(input_shape, args, RNNAgent)

def build_actor_rnn(input_shape, args, **ignored):
    return RNNActorAgent(input_shape, args)

def build_sides_actor_rnn(input_shape, args, **ignored):
    return SharedButSidesAgent(input_shape, args, RNNActorAgent)

def build_non_shared_actor_rnn(input_shape, args, **ignored):
    return NonSharedAgent(input_shape, args, RNNActorAgent)

def build_mean_log_std_rnn(input_shape, args, **ignored):
    return RNNMeanLogStdAgent(input_shape, args)

def build_non_shared_mean_log_std_rnn(input_shape, args, **ignored):
    return SharedButSidesAgent(input_shape, args, RNNMeanLogStdAgent)

def build_sides_mean_log_std_rnn(input_shape, args, **ignored):
    return NonSharedAgent(input_shape, args, RNNMeanLogStdAgent)

def build_mean_log_std_mlp(input_shape, args, **ignored):
    return MLPMeanLogStdAgent(input_shape, args)

def build_transformer(args, **ignored):
    return TransformerAgent(args)

def build_transformer_actor(args, **ignored):
    return TransformerActorAgent(args)

def build_shared_but_first_actor_mlp(input_shape, args, **ignored):
    return SharedButFirstAgent(input_shape, args, MLPActorAgent)

def build_cnn_actor(input_shape, args, **ignored):
    return CNNActorAgent(input_shape, args)

def build_mlp_actor_ao(input_shape, args, **ignored):
    return MLPActorAgentAdaptiveOptics(input_shape, args) 


agent_factory = Factory()

# Value-based methods

agent_factory.register_builder("mlp", build_mlp)
agent_factory.register_builder("shared_but_sides_mlp", build_sides_mlp)
agent_factory.register_builder("non_shared_mlp", build_non_shared_mlp)

agent_factory.register_builder("rnn", build_rnn)
agent_factory.register_builder("shared_but_sides_rnn", build_sides_rnn)
agent_factory.register_builder("non_shared_rnn", build_non_shared_rnn)

agent_factory.register_builder("transformer", build_transformer)

# Policy gradient-based methods

agent_factory.register_builder("mlp_actor", build_actor_mlp)
agent_factory.register_builder("shared_but_sides_actor_mlp", build_sides_mlp)
agent_factory.register_builder("non_shared_actor_mlp", build_non_shared_actor_mlp)
agent_factory.register_builder('shared_but_first_actor_mlp', build_shared_but_first_actor_mlp)

agent_factory.register_builder('mlp_actor_ao', build_mlp_actor_ao)

agent_factory.register_builder("rnn_actor", build_actor_rnn)
agent_factory.register_builder("shared_but_sides_actor_rnn", build_sides_actor_rnn)
agent_factory.register_builder("non_shared_actor_rnn", build_non_shared_actor_rnn)

agent_factory.register_builder("cnn_actor", build_cnn_actor)

agent_factory.register_builder("rnn_mean_log_std", build_mean_log_std_rnn)
agent_factory.register_builder("mlp_mean_log_std", build_mean_log_std_mlp)
agent_factory.register_builder("non_shared_mean_log_std_rnn", build_non_shared_mean_log_std_rnn)
agent_factory.register_builder("sides_mean_log_std_rnn", build_sides_mean_log_std_rnn)

agent_factory.register_builder("transformer_actor", build_transformer_actor)
