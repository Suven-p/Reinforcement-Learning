import numpy as np
from gridWorld2 import Env


def compute_state_value(env: Env, state_values, s, gamma):
    new_value = 0
    for a in env.A:
        current = 0
        for s_ in env.S:
            for r in env.R:
                prob = env.prob(s_, r, s, a)
                current += prob * (r +
                                   gamma * state_values[s_])
        new_value += env.policy[s, a] * current
    return new_value


def update_state_values(env: Env, state_values, discount):
    new_state_values = np.zeros(state_values.shape)
    largest_diff = 0
    for s in env.S:
        if s in env.TerminalStates:
            continue
        new_value = compute_state_value(env, state_values, s, discount)
        new_state_values[s] = new_value
        largest_diff = max(largest_diff,
                           abs(state_values[s] - new_value))
    state_values = new_state_values
    return state_values, largest_diff


def policy_evaluation(env: Env, state_values=None, discount=1.0):
    num_iter = 0
    largest_diff = 0.0
    threshold = 1
    threshold = 0.000000000000001
    diff_history = []
    if state_values is None:
        state_values = np.zeros(env.S.size)
    while True:
        state_values, largest_diff = update_state_values(
            env, state_values, discount)
        diff_history.append(largest_diff)
        num_iter += 1
        if largest_diff < threshold:
            print(
                f"Converged with diff {largest_diff} after {num_iter} iterations."
            )
            break
    return state_values


def policy_improvement(env: Env, state_values, discount: float = 1.0):
    stable = True
    for s in env.S:
        if s in env.TerminalStates:
            continue
        optimal_actions = None
        optimal_action_value = float('-inf')
        for a in env.A:
            current_value = 0.0
            for s_ in env.S:
                for r in env.R:
                    prob = env.prob(s_, r, s, a)
                    current_value += prob * (
                        r + discount * state_values[s_])
            if abs(current_value - optimal_action_value) < 1e-6:
                optimal_actions.append(a)
            elif current_value > optimal_action_value:
                optimal_action_value = current_value
                optimal_actions = [a]
        new_policy = []
        for a in env.A:
            if a in optimal_actions:
                new_policy.append(1 / len(optimal_actions))
            else:
                new_policy.append(0)
        stable = stable and np.array_equal(new_policy, env.policy[s])
        env.policy[s] = new_policy
        return stable
