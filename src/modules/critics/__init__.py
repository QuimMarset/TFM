from modules.critics.mlp_critic import MLPCritic
from modules.critics.rnn_critic import RNNCritic
from modules.critics.non_shared_critic import NonSharedCritic
from modules.critics.shared_but_sides_critic import SharedButSidesCritic
from modules.common.factory import Factory


critic_factory = Factory()

def build_mlp(history_shape, action_shape, args, **ignored):
    return MLPCritic(history_shape, action_shape, args)

def build_rnn(history_shape, action_shape, args, **ignored):
    return RNNCritic(history_shape, action_shape, args)

def build_shared_but_sides_mlp(history_shape, action_shape, args, **ignored):
    return SharedButSidesCritic(history_shape, action_shape, args, MLPCritic)

def build_shared_but_sides_rnn(history_shape, action_shape, args, **ignored):
    return SharedButSidesCritic(history_shape, action_shape, args, RNNCritic)

def build_non_shared_mlp(history_shape, action_shape, args, **ignored):
    return NonSharedCritic(history_shape, action_shape, args, MLPCritic)

def build_non_shared_rnn(history_shape, action_shape, args, **ignored):
    return NonSharedCritic(history_shape, action_shape, args, RNNCritic)


critic_factory.register_builder('mlp', build_mlp)
critic_factory.register_builder('rnn', build_rnn)
critic_factory.register_builder('sides_mlp', build_shared_but_sides_mlp)
critic_factory.register_builder('sides_rnn', build_shared_but_sides_rnn)
critic_factory.register_builder('non_shared_mlp', build_non_shared_mlp)
critic_factory.register_builder('non_shared_rnn', build_non_shared_rnn)
