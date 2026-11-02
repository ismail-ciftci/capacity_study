"""Random streams. Every stream is a numpy SeedSequence child of the study root seed.

Member key = (phase code, recipe_id, stream index). Tuning trials of one recipe share stream index 0, so they share
initialization and data order (paired trials). Final seeds use stream index = seed index. Streams of different
recipes or phases share nothing. Data order for epoch e is derived from (key, e), so resuming needs no RNG state.
"""
import numpy as np

PHASE_CODE = {"calibration": 0, "tuning": 1, "final": 2}
_INIT, _ORDER = 0, 1


def member_key(phase: str, recipe_id: int, stream_index: int) -> tuple[int, int, int]:
    return (PHASE_CODE[phase], recipe_id, stream_index)


def _seed(root_seed: int, spawn_key: tuple[int, ...]) -> int:
    words = np.random.SeedSequence(root_seed, spawn_key=spawn_key).generate_state(2, dtype=np.uint32)
    return int(words[0]) | (int(words[1] & 0x7FFFFFFF) << 32)


def init_seed(root_seed: int, key: tuple[int, int, int]) -> int:
    return _seed(root_seed, (*key, _INIT))


def order_seed(root_seed: int, key: tuple[int, int, int], epoch: int) -> int:
    return _seed(root_seed, (*key, _ORDER, epoch))
