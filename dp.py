import numpy as np
from gridWorld import Env


def compute_state_value(env: Env, s, gamma):
    new_value = 0
    for action in env.A:
        current = 0
        for s_ in env.S:
            for r in env.R:
                prob = env.prob(from_state=s,
                                to_state=s_,
                                action=action,
                                reward=r)

                current += prob * (r +
                                   gamma * env.state_values[s_])
        new_value += env.policy(s, action) * current
    return new_value


def update_state_values(env: Env, discount):
    new_state_values = np.zeros(env.state_values.shape)
    largest_diff = 0
    for s in env.S:
        if s in env.TerminalStates:
            continue
        new_value = compute_state_value(env, s, discount)
        new_state_values[s] = new_value
        largest_diff = max(largest_diff,
                           abs(env.state_values[s] - new_value))
    env.state_values = new_state_values
    return largest_diff


def policy_evaluation(env: Env, discount=1.0):
    num_iter = 0
    largest_diff = 0.0
    threshold = 1
    threshold = 0.000000000000001
    diff_history = []
    while True:
        largest_diff = update_state_values(env, discount)
        diff_history.append(largest_diff)
        num_iter += 1
        if largest_diff < threshold:
            print(
                f"Converged with diff {largest_diff} after {num_iter} iterations."
            )
            break


def policy_improvement(env: Env, discount: float = 1.0):
    while True:
        stable = True
        for s in env.S:
            if s in env.TerminalStates:
                continue
            optimal_actions = None
            optimal_action_value = float('-inf')
            for action in env.A:
                current_value = 0.0
                for new_state in env.S:
                    for reward in env.R:
                        prob = env.prob(s, new_state, action, reward)
                        current_value += prob * (
                            reward + discount * env.state_values[new_state])
                if abs(current_value - optimal_action_value) < 1e-6:
                    optimal_actions.append(action)
                elif current_value > optimal_action_value:
                    optimal_action_value = current_value
                    optimal_actions = [action]
            if len(optimal_actions) != len(env.policy_table[s]):
                stable = False
            elif len([
                    i for i in optimal_actions
                    if i not in env.policy_table[s]
            ]) > 0:
                stable = False
            env.policy_table[s] = optimal_actions
        if stable:
            break
