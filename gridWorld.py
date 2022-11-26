from enum import Enum
import numpy as np
import random


class Env:
    """
    A 4x4 grid world where possible action are movement in up, down, left or right direction
    rewards_table stores mapping from state to reward
    It is assumed that movement from any direction to a given state provides same reward depending on destination state
    policy_table stores a deterministic mapping from state to optimal action
    In default state, the policy table assumes each action is optimal
    """
    States = [(i, j) for i in range(4) for j in range(4)]
    TerminalStates = ((0, 0), (3,3))
    Actions = Enum('Actions', ['U', 'D', 'L', 'R'])
    Rewards = (-1, )
    GridSize = (4,4)

    def __init__(self, state_values=None):
        if state_values is None:
            state_values = np.zeros(self.GridSize)
        self.state_values = state_values
        self.policy_table = {}
        self.rewards_table = {}
        for state in self.states:
            self.policy_table[state] = [a for a in self.actions]
            self.rewards[state] = -1

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
        if action == self.action.U:
            new_row = state[0]-1 if state[0] > 0 else 0
            new_action = (new_row, state[1])
        elif action == self.action.D:
            new_row = state[0]+1 if state[0] < self.gridSize[0] else 0
            new_action = (new_row, state[1])
        elif action == self.action.L:
            new_col = state[1]-1 if state[1] > 0 else 0
            new_action = (state[0], new_col)
        elif action == self.action.R:
            new_col = state[1]+1 if state[1] < self.gridSize[1] else 0
            new_action = (state[0], new_col)
        return (new_action, self.rewards[state])

    def prob(self, from_state, to_state, action, reward):
        """
        Provides the probability distribution for model of problem
        Returns p(s', r | s, a) Probability that action a from state s results in state s' with reward r
        """
        if reward not in self.rewards:
            return 0
        new_state, new_reward = self.__call(from_state, action)
        if to_state == new_state and reward == new_reward:
            return 1
        return 0

