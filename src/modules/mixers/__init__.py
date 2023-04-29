from modules.common.factory import Factory
from modules.mixers.qmix import QMixer



def build_qmix_mixer(args, **ignored):
    return QMixer(args)



mixer_factory = Factory()

mixer_factory.register_builder('qmix', build_qmix_mixer)
