from runners.episode_runner import EpisodeRunner
from runners.parallel_runner import ParallelRunner
from runners.episode_runner_ao import EpisodeRunnerAdaptiveOptics
from runners.parallel_runner_one_thread import ParallelRunnerOne


REGISTRY = {}


REGISTRY["episode"] = EpisodeRunner
REGISTRY["parallel"] = ParallelRunner
REGISTRY['episode_ao'] = EpisodeRunnerAdaptiveOptics
REGISTRY['parallel_one'] = ParallelRunnerOne
