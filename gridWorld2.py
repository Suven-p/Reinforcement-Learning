from enum import IntEnum
import numpy as np


class Actions(IntEnum):
    U = 0
    D = 1
    L = 2
    R = 3


class Env:
    """
    A m*n grid world where possible action are movement in up, down, left or right direction
    rewards_table stores mapping from state to reward
    It is assumed that movement from any direction to a given state provides same reward depending on destination state
    policy_table stores a deterministic mapping from state to optimal action
    In default state, the policy table assumes each action is optimal
    """

    def __init__(self, grid_size=(4, 4), terminal_states=[0, 15]):
        self.size = grid_size
        self.S = np.arange(grid_size[0] * grid_size[1])
        self.TerminalStates = terminal_states

        self.A = Actions
        self.R = np.array((-1, 0, -10))

        self.transitions = np.zeros(
            (len(self.S), len(self.A), len(self.S), len(self.R)))
        self.__policy = np.zeros((len(self.S), len(self.A)))

        for s in self.S:
            for a in self.A:
                for s_ in self.S:
                    for r in range(len(self.R)):
                        self.transitions[s, int(a), s_,
                                         r] = self._initial_prob(s, s_, a, self.R[r])

        for s in self.S:
            for a in self.A:
                self.__policy[s, int(a)] = 0.25

    def get_state_index(self, state):
        isValidShape = len(state) == 2
        isWithinLimit = (0 <= state[0] < self.size[0]) and (0 <= state[1] <
                                                            self.size[1])
        isValid = isValidShape and isWithinLimit
        if not isValid:
            raise ValueError("Invalid state {}".format(state))
        return state[0] * self.size[1] + state[1]

    def get_state_from_index(self, index):
        return (index // self.size[1], index % self.size[1])

    @property
    def policy(self):
        """
        Implements policy
        Returns pi(a | s): Probability of taking action a from state s
        """
        return self.__policy

    def __call__(self, s, a):
        """
        Simulate taking given action from given state
        Returns (new_state, reward)
        """
        a = int(a)
        p = self.transitions[s, a].sum(axis=1)
        s_ = np.random.choice(self.S, p=p)
        r = np.random.choice(self.R, p=self.transitions[s, a, s_])
        return (s_, r)

    def _initial_prob(self, s, s_, a, r):
        """
        Provides the initial probability distribution for model of problem
        Returns p(s', r | s, a) Probability that action a from state s results in state s' with reward r
        """
        if r not in self.R:
            return 0
        if s in self.TerminalStates:
            return 1 if s == s_ and r == 0 else 0
        new_reward = -1
        s = self.get_state_from_index(s)
        if a == self.A.U:
            new_row = s[0] - 1 if s[0] > 0 else 0
            new_state = (new_row, s[1])
        elif a == self.A.D:
            new_row = min(s[0] + 1, self.size[0] - 1)
            new_state = (new_row, s[1])
        elif a == self.A.L:
            new_col = s[1] - 1 if s[1] > 0 else 0
            new_state = (s[0], new_col)
        elif a == self.A.R:
            new_col = min(s[1] + 1, self.size[1] - 1)
            new_state = (s[0], new_col)
        else:
            raise NotImplementedError
        new_state = self.get_state_index(new_state)
        if s_ == new_state and r == new_reward:
            return 1
        return 0

    def prob(self, s_, r, s, a):
        return self.transitions[s, int(a), s_, np.where(self.R == r)[0][0]]


if __name__ == "__main__":
    env = Env(terminal_states=[0])
    for state in env.S:
        for action in env.A:
            assert env.policy[state, action] == 0.25

    assert np.array_equal(env(env.get_state_index((0, 0)), env.A.L),
                          (env.get_state_index((0, 0)), 0))
    assert np.array_equal(env(env.get_state_index((0, 0)), env.A.U),
                          (env.get_state_index((0, 0)), 0))
    assert np.array_equal(env(env.get_state_index((0, 0)), env.A.R),
                          (env.get_state_index((0, 0)), 0))
    assert np.array_equal(env(env.get_state_index((0, 0)), env.A.D),
                          (env.get_state_index((0, 0)), 0))

    assert np.array_equal(env(env.get_state_index((0, 3)), env.A.L),
                          (env.get_state_index((0, 2)), -1))
    assert np.array_equal(env(env.get_state_index((0, 3)), env.A.U),
                          (env.get_state_index((0, 3)), -1))
    assert np.array_equal(env(env.get_state_index((0, 3)), env.A.R),
                          (env.get_state_index((0, 3)), -1))
    assert np.array_equal(env(env.get_state_index((0, 3)), env.A.D),
                          (env.get_state_index((1, 3)), -1))

    assert np.array_equal(env(env.get_state_index((3, 0)), env.A.L),
                          (env.get_state_index((3, 0)), -1))
    assert np.array_equal(env(env.get_state_index((3, 0)), env.A.U),
                          (env.get_state_index((2, 0)), -1))
    assert np.array_equal(env(env.get_state_index((3, 0)), env.A.R),
                          (env.get_state_index((3, 1)), -1))
    assert np.array_equal(env(env.get_state_index((3, 0)), env.A.D),
                          (env.get_state_index((3, 0)), -1))

    assert np.array_equal(env(env.get_state_index((2, 2)), env.A.L),
                          (env.get_state_index((2, 1)), -1))
    assert np.array_equal(env(env.get_state_index((2, 2)), env.A.U),
                          (env.get_state_index((1, 2)), -1))

    assert np.array_equal(env(env.get_state_index((2, 2)), env.A.R),
                          (env.get_state_index((2, 3)), -1))
    assert np.array_equal(env(env.get_state_index((2, 2)), env.A.D),
                          (env.get_state_index((3, 2)), -1))

    assert np.array_equal(env(env.get_state_index((3, 3)), env.A.L),
                          (env.get_state_index((3, 2)), -1))
    assert np.array_equal(env(env.get_state_index((3, 3)), env.A.U),
                          (env.get_state_index((2, 3)), -1))
    assert np.array_equal(env(env.get_state_index((3, 3)), env.A.R),
                          (env.get_state_index((3, 3)), -1))
    assert np.array_equal(env(env.get_state_index((3, 3)), env.A.D),
                          (env.get_state_index((3, 3)), -1))

    env = Env(terminal_states=[0, 15])
    assert np.array_equal(env(env.get_state_index((3, 3)), env.A.L),
                          (env.get_state_index((3, 3)), 0))
    assert np.array_equal(env(env.get_state_index((3, 3)), env.A.U),
                          (env.get_state_index((3, 3)), 0))
    assert np.array_equal(env(env.get_state_index((3, 3)), env.A.R),
                          (env.get_state_index((3, 3)), 0))
    assert np.array_equal(env(env.get_state_index((3, 3)), env.A.D),
                          (env.get_state_index((3, 3)), 0))
