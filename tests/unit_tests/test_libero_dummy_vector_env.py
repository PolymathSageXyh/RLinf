import numpy as np
import pytest


pytest.importorskip("libero")

from rlinf.envs.libero.venv import ReconfigureDummyEnv


class _FakeLiberoEnv:
    def __init__(self, value: int) -> None:
        self.value = value
        self.closed = False

    def step(self, action):
        return np.array([self.value]), float(action), False, {}

    def reset(self, **kwargs):
        return np.array([self.value])

    def seed(self, seed=None):
        return [seed]

    def close(self) -> None:
        self.closed = True


def test_reconfigure_dummy_env_replaces_selected_env_in_process() -> None:
    old_envs = [_FakeLiberoEnv(1), _FakeLiberoEnv(2)]
    vector_env = ReconfigureDummyEnv(
        [lambda: old_envs[0], lambda: old_envs[1]]
    )

    vector_env.reconfigure_env_fns([lambda: _FakeLiberoEnv(9)], id=[1])
    observations, rewards, _, infos = vector_env.step(np.array([3.0, 4.0]))

    np.testing.assert_array_equal(observations, np.array([[1], [9]]))
    np.testing.assert_array_equal(rewards, np.array([3.0, 4.0]))
    assert [info["env_id"] for info in infos] == [0, 1]
    assert not old_envs[0].closed
    assert old_envs[1].closed

    vector_env.close()
    assert old_envs[0].closed


def test_reconfigure_dummy_env_validates_factory_count() -> None:
    vector_env = ReconfigureDummyEnv([lambda: _FakeLiberoEnv(1)])

    with pytest.raises(ValueError, match="Expected 1 environment factories"):
        vector_env.reconfigure_env_fns([], id=[0])

    vector_env.close()
