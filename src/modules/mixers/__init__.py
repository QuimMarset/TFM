from modules.common.factory import Factory
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.transformer_mixer import TransformerMixer
from modules.mixers.transformer_mixer_cont import TransformerMixerContinuous
from modules.mixers.custom_mixer import CustomMixer



def build_qmix_mixer(args, **ignored):
    return QMixer(args)


def build_vdn_mixer(args, **ignored):
    return VDNMixer(args)


def build_transformer_mixer(args, abs=False, **ignored):
    return TransformerMixer(args, abs)


def build_continuous_transformer_mixer(args, **ignored):
    return TransformerMixerContinuous(args)


def build_custom_mixer(args, **ignored):
    return CustomMixer(args)



mixer_factory = Factory()

mixer_factory.register_builder('qmix', build_qmix_mixer)
mixer_factory.register_builder('vdn', build_vdn_mixer)
mixer_factory.register_builder('transformer', build_transformer_mixer)
mixer_factory.register_builder('transformer_continuous', build_continuous_transformer_mixer)
mixer_factory.register_builder('custom_mixer', build_custom_mixer)