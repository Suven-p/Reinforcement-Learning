from enum import IntEnum
import numpy as np
import random


class Actions(IntEnum):
    U = 0
    D = 1
    L = 2
    R = 3


class Env:
    """
    A 4x4 grid world where possible action are movement in up, down, left or right direction
    rewards_table stores mapping from state to reward
    It is assumed that movement from any direction to a given state provides same reward depending on destination state
    policy_table stores a deterministic mapping from state to optimal action
    In default state, the policy table assumes each action is optimal
    """

    def __init__(self, grid_size=(4, 4), terminal_states=[(0, 0), (3, 3)]):
        self.S = np.array([(i, j) for i in range(grid_size[0])
                           for j in range(grid_size[1])])
        self.TerminalStates = terminal_states
        self.A = np.array([int(i) for i in Actions])
        self.R = np.array((-1, 0, -10))
        self.size = grid_size

    def get_state_index(self, state):
        if not np.any([np.all(self.S[k] == state) for k in range(len(self.S))]):
            raise ValueError("Invalid state {}".format(state))
        return state[0] * self.size[1] + state[1]

    def policy(self, state, action):
        """
        Implements policy
        Returns pi(a | s): Probability of taking action a from state s
        """
        if action in self.policy_table[state]:
            return 1 / len(self.policy_table[state])
        else:
            return 0

    def __call__(self, state, action):
        """
        Simulate taking given action from given state
        Returns (new_state, reward)
        """
        if state in self.TerminalStates:
            return state, 0
        if action == self.A.U:
            new_row = state[0]-1 if state[0] > 0 else 0
            new_state = (new_row, state[1])
        elif action == self.A.D:
            new_row = min(state[0]+1, self.size[0] - 1)
            new_state = (new_row, state[1])
        elif action == self.A.L:
            new_col = state[1]-1 if state[1] > 0 else 0
            new_state = (state[0], new_col)
        elif action == self.A.R:
            new_col = min(state[1]+1, self.size[1] - 1)
            new_state = (state[0], new_col)
        return ((new_state, self.rewards_table[state]),)

    def prob(self, from_state, to_state, action, reward):
        """
        Provides the probability distribution for model of problem
        Returns p(s', r | s, a) Probability that action a from state s results in state s' with reward r
        """
        if reward not in self.R:
            return 0
        new_state, new_reward = self.__call__(from_state, action)
        if to_state == new_state and reward == new_reward:
            return 1
        return 0

    def choose_action(self, state, discount):
        max_value = float('-inf')
        chosen_actions = []
        for action in self.A:
            new_state, current_reward = self.__call(state, action)
            expected_return = current_reward + \
                discount * self.state_values[new_state]
            if expected_return > max_value:
                max_value = expected_return
                chosen_actions = [action]
            elif expected_return == max_value:
                chosen_actions.append(action)
        return random.choice(chosen_actions)

    def get_reward(self, from_state, action, to_state):
        new_state, reward = self.__call__(from_state, action)
        if new_state == to_state:
            return reward
        else:
            return 0


if __name__ == "__main__":
    env = Env()
    for state in env.S:
        for action in env.A:
            assert env.policy(state, action) == 0.25
    assert env((0, 0), env.A.L) == ((0, 0), 0)
    assert env((0, 0), env.A.U) == ((0, 0), 0)
    assert env((0, 0), env.A.R) == ((0, 0), 0)
    assert env((0, 0), env.A.D) == ((0, 0), 0)

    assert env((2, 2), env.A.L) == ((2, 1), -1)
    assert env((2, 2), env.A.U) == ((1, 2), -1)
    assert env((2, 2), env.A.R) == ((2, 3), -1)
    assert env((2, 2), env.A.D) == ((3, 2), -1)

    assert env((3, 3), env.A.L) == ((3, 2), -1)
    assert env((3, 3), env.A.U) == ((2, 3), -1)
    assert env((3, 3), env.A.R) == ((3, 3), -1)
    assert env((3, 3), env.A.D) == ((3, 3), -1)
