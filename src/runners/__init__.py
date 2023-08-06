from runners.episode_runner import EpisodeRunner
from runners.parallel_runner import ParallelRunner
from runners.episode_runner_ao import EpisodeRunnerAdaptiveOptics


REGISTRY = {}


REGISTRY["episode"] = EpisodeRunner
REGISTRY["parallel"] = ParallelRunner
REGISTRY['episode_ao'] = EpisodeRunnerAdaptiveOptics
