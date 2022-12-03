import numpy as np
from gridWorld import Env


def update_state_values(env: Env, state_values, gamma):
    new_state_values = np.zeros(state_values.shape)
    largest_diff = 0
    # Calculate sum of p * (r + gamma * V(s'))
    temp = env.transitions * (env.R + gamma * state_values.reshape((-1, 1)))
    sum_sr = np.sum(temp, axis=(2, 3))
    # Multiply return by policy
    sum_sa = env.policy * sum_sr
    new_state_values = np.sum(sum_sa, axis=1)
    largest_diff = np.max(np.abs(state_values - new_state_values))
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


def policy_improvement(env: Env, state_values, gamma: float = 1.0):
    stable = True
    values = env.transitions * (env.R + gamma * state_values.reshape((-1, 1)))
    values = np.sum(values, axis=(2, 3))
    optimal_actions_all = np.max(values, axis=1)
    for s in env.S:
        if s in env.TerminalStates:
            continue
        optimal_actions = np.where(np.isclose(
            values[s, :], optimal_actions_all[s]))[0]
        new_policy = np.zeros(len(env.A))
        new_policy[optimal_actions] = 1 / len(optimal_actions)
        stable = stable and np.array_equal(new_policy, env.policy[s])
        env.policy[s] = new_policy
    return stable
