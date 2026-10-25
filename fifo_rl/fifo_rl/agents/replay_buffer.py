"""Contract E: the replay memory the DQN learns from.

    buf = ReplayBuffer(capacity, state_dim, seed)
    buf.push(s, a, r, s2, done)
    batch = buf.sample(batch_size)      # Batch of numpy arrays, see below
    len(buf)                            # transitions stored (at most capacity)

A ring buffer of preallocated numpy arrays: push writes at the next slot and, once the buffer
is full, overwrites the oldest transition. Sampling is uniform with replacement from its own
seeded generator, so the same seed and the same pushes give the same batches.
"""

from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    """B transitions, one row per transition."""
    states: np.ndarray       # float32 (B, state_dim)
    actions: np.ndarray      # int64   (B,)
    rewards: np.ndarray      # float32 (B,)
    next_states: np.ndarray  # float32 (B, state_dim)
    dones: np.ndarray        # float32 (B,)  1.0 = episode ended after this transition


class ReplayBuffer:
    def __init__(self, capacity, state_dim, seed=0):
        """Input:  capacity, max transitions kept (10 000 in the default config);
                state_dim, length of a state vector (87);
                seed, seed of the sampling generator.
        Output: nothing.
        Logic:  allocate all arrays once; _next is the slot the next push writes to, _size
                how many slots hold a transition.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        if state_dim < 1:
            raise ValueError(f"state_dim must be at least 1, got {state_dim}")
        self.capacity = capacity
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._next = 0
        self._size = 0

    def __len__(self):
        return self._size

    def push(self, s, a, r, s2, done):
        """Input:  one transition: state, action id, reward, next state, done flag.
        Output: nothing.
        Logic:  write it at slot _next, then move _next one slot on (wrapping to 0), so the
                oldest transition is the one overwritten when the buffer is full.
        """
        i = self._next
        self._states[i] = s
        self._actions[i] = a
        self._rewards[i] = r
        self._next_states[i] = s2
        self._dones[i] = float(done)
        self._next = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        """Input:  batch_size, number of transitions B.
        Output: a Batch of copies (changing it does not change the buffer).
        Logic:  B slot indices drawn uniformly, with replacement, from the filled slots.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self.rng.integers(self._size, size=batch_size)
        return Batch(
            states=self._states[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_states=self._next_states[idx],
            dones=self._dones[idx],
        )
