from .episode_runner import EpisodeRunner
from .parallel_runner import ParallelRunner


REGISTRY = {}


REGISTRY["episode"] = EpisodeRunner
REGISTRY["parallel"] = ParallelRunner
