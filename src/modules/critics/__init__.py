from modules.common.factory import Factory
from modules.critics.mlp_critic import MLPCritic
from modules.critics.rnn_critic import RNNCritic
from modules.critics.non_shared_critic import NonSharedCritic
from modules.critics.shared_but_sides_critic import SharedButSidesCritic
from modules.critics.double_critic import DoubleCritic
from modules.critics.non_shared_double_critic import NonSharedDoubleCritic
from modules.critics.shared_but_first_critic import SharedButFirstCritic
from modules.critics.shared_but_first_double_critic import SharedButFirstDoubleCritic
from modules.critics.cnn_critic import CNNCritic
from modules.critics.centralized_mlp_critic import CentralizedMLPCritic
from modules.critics.centralized_cnn_critic import CentralizedCNNCritic
from modules.critics.mlp_critic_ao import MLPCriticAdaptiveOptics


critic_factory = Factory()

def build_mlp(input_shape, action_shape, args, **ignored):
    return MLPCritic(input_shape, action_shape, args)

def build_rnn(input_shape, action_shape, args, **ignored):
    return RNNCritic(input_shape, action_shape, args)

def build_shared_but_sides_mlp(input_shape, action_shape, args, **ignored):
    return SharedButSidesCritic(input_shape, action_shape, args, MLPCritic)

def build_shared_but_sides_rnn(input_shape, action_shape, args, **ignored):
    return SharedButSidesCritic(input_shape, action_shape, args, RNNCritic)

def build_non_shared_mlp(input_shape, action_shape, args, **ignored):
    return NonSharedCritic(input_shape, action_shape, args, MLPCritic)

def build_non_shared_rnn(input_shape, action_shape, args, **ignored):
    return NonSharedCritic(input_shape, action_shape, args, RNNCritic)

def build_double_critic(input_shape, action_shape, args, **ignored):
    return DoubleCritic(input_shape, action_shape, args, MLPCritic)

def build_double_critic_rnn(input_shape, action_shape, args, **ignored):
    return DoubleCritic(input_shape, action_shape, args, RNNCritic)

def build_non_shared_double_critic(input_shape, action_shape, args, **ignored):
    return NonSharedDoubleCritic(input_shape, action_shape, args, MLPCritic)

def build_shared_but_first_critic(input_shape, action_shape, args, **ignored):
    return SharedButFirstCritic(input_shape, action_shape, args, MLPCritic)

def build_shared_but_first_double_critic(input_shape, action_shape, args, **ignored):
    return SharedButFirstDoubleCritic(input_shape, action_shape, args, MLPCritic)

def build_cnn_critic(input_shape, action_shape, args, **ignored):
    return CNNCritic(input_shape, action_shape, args)

def build_double_critic_cnn(input_shape, action_shape, args, **ignored):
    return DoubleCritic(input_shape, action_shape, args, CNNCritic)

def build_centralized_mlp_critic(input_shape, action_shape, args, **ignored):
    return CentralizedMLPCritic(input_shape, action_shape, args)

def build_centralized_cnn_critic(input_shape, action_shape, args, **ignored):
    return CentralizedCNNCritic(input_shape, action_shape, args)

def build_mlp_critic_ao(input_shape, action_shape, args, **ignored):
    return MLPCriticAdaptiveOptics(input_shape, action_shape, args)

def build_double_critic_ao(input_shape, action_shape, args, **ignored):
    return DoubleCritic(input_shape, action_shape, args, MLPCriticAdaptiveOptics)


critic_factory.register_builder('mlp', build_mlp)
critic_factory.register_builder('sides_mlp', build_shared_but_sides_mlp)
critic_factory.register_builder('non_shared_mlp', build_non_shared_mlp)
critic_factory.register_builder('shared_but_first_mlp', build_shared_but_first_critic)

critic_factory.register_builder('mlp_ao', build_mlp_critic_ao)
critic_factory.register_builder('double_critic_ao', build_double_critic_ao)

critic_factory.register_builder('rnn', build_rnn)
critic_factory.register_builder('sides_rnn', build_shared_but_sides_rnn)
critic_factory.register_builder('non_shared_rnn', build_non_shared_rnn)

critic_factory.register_builder('double_critic', build_double_critic)
critic_factory.register_builder('non_shared_double_critic', build_non_shared_double_critic)
critic_factory.register_builder('shared_but_first_double_critic', build_shared_but_first_double_critic)

critic_factory.register_builder('double_critic_rnn', build_double_critic_rnn)

critic_factory.register_builder('cnn', build_cnn_critic)
critic_factory.register_builder('double_critic_cnn', build_double_critic_cnn)

critic_factory.register_builder('centralized_mlp', build_centralized_mlp_critic)
critic_factory.register_builder('centralized_cnn', build_centralized_cnn_critic)