from runners.episode_runner import EpisodeRunner
from runners.episode_runner_ao import EpisodeRunnerAdaptiveOptics
from runners.parallel_runner_one_thread import ParallelRunnerOneThread


REGISTRY = {}


REGISTRY["episode"] = EpisodeRunner
REGISTRY['episode_ao'] = EpisodeRunnerAdaptiveOptics
REGISTRY['parallel'] = ParallelRunnerOneThread
