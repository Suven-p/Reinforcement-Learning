import numpy as np
import gridWorld

env = gridWorld.Env()


def compute_state_value(env, state, discount):
    new_value = 0
    for action in env.actions:
        current = 0
        for new_state in env.states:
            for reward in env.rewards:
                prob = env.prob(from_state=state,
                                to_state=new_state,
                                action=action,
                                reward=reward)
                obtained_reward = env.env_reward(state, action, new_state)
                current += prob * (obtained_reward +
                                   discount * env.state_values[new_state])
        new_value += env.policy(state, action) * current
    return new_value


def update_state_values(env, discount):
    new_state_values = np.zeros(env.state_values.shape)
    largest_diff = 0
    for state in env.states:
        if state in env.terminalStates:
            continue
        new_value = compute_state_value(env, state, discount)
        new_state_values[state] = new_value
        largest_diff = max(largest_diff,
                           abs(env.state_values[state] - new_value))
    env.state_values = new_state_values
    return largest_diff


def policy_evaluation(env, discount=1.0):
    num_iter = 0
    largest_diff = 0.0
    # threshold = 0.000000000000001
    threshold = 1
    diff_history = []
    while True:
        largest_diff = update_state_values(env, discount)
        diff_history.append(largest_diff)
        num_iter += 1
        print(largest_diff)
        if largest_diff < threshold:
            print(
                f"Converged with diff {largest_diff} after {num_iter} iterations."
            )
            break


def policy_improvement(env: gridWorld.Env, discount: float = 1.0):
    while True:
        stable = True
        for state in env.states:
            optimal_actions = None
            optimal_action_value = float('-inf')
            for action in env.actions:
                current_value = 0.0
                for new_state in env.states:
                    for reward in env.rewards:
                        prob = env.prob(state, new_state, action, reward)
                        env_reward = env.env_reward(state, action, new_state)
                        current_value += prob * (env_reward + discount *
                                                 env.state_values[new_state])
                if current_value > optimal_action_value:
                    optimal_action_value = current_value
                    optimal_actions = [action]
                    stable = False if action not in env.policy_table[
                        state] else True
                elif current_value == optimal_action_value:
                    optimal_actions.append(action)
                    stable = False if action not in env.policy_table[
                        state] else True
            env.policy_table[state] = optimal_actions
        if stable:
            break


policy_evaluation(env)
# policy_improvement(env)
# policy_evaluation(env)
# print(env.state_values, "\n\n\n")
# plt.plot(diff_history)
# plt.show()
