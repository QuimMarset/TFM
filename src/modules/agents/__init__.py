from modules.common.factory import Factory
from modules.agents.mlp_agent import MLPAgent
from modules.agents.mlp_agent_actor import MLPActorAgent
from modules.agents.rnn_agent import RNNAgent
from modules.agents.rnn_agent_actor import RNNActorAgent
from modules.agents.rnn_mean_log_std_agent import RNNMeanLogStdAgent
from modules.agents.shared_but_sides_agent import SharedButSidesAgent
from modules.agents.non_shared_agent import NonSharedAgent


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


agent_factory = Factory()

# Value-based methods
agent_factory.register_builder("mlp", build_mlp)
agent_factory.register_builder("sides_mlp", build_sides_mlp)
agent_factory.register_builder("non_shared_mlp", build_non_shared_mlp)
agent_factory.register_builder("rnn", build_rnn)
agent_factory.register_builder("sides_rnn", build_sides_rnn)
agent_factory.register_builder("non_shared_rnn", build_non_shared_rnn)

# Policy gradient-based methods
agent_factory.register_builder("mlp_actor", build_mlp)
agent_factory.register_builder("sides_actor_mlp", build_sides_mlp)
agent_factory.register_builder("non_shared_actor_mlp", build_non_shared_mlp)
agent_factory.register_builder("rnn_actor", build_actor_rnn)
agent_factory.register_builder("sides_actor_rnn", build_sides_actor_rnn)
agent_factory.register_builder("non_shared_actor_rnn", build_non_shared_actor_rnn)
agent_factory.register_builder("rnn_mean_log_std", build_mean_log_std_rnn)
agent_factory.register_builder("non_shared_mean_log_std_rnn", build_non_shared_mean_log_std_rnn)
agent_factory.register_builder("sides_mean_log_std_rnn", build_sides_mean_log_std_rnn)
