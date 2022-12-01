import numpy as np
import gridWorld2 as gw
from dp2 import policy_evaluation, policy_improvement

env = gw.Env()

stable = False
state_values = None
while not stable:
    state_values = policy_evaluation(env, state_values)
    stable = policy_improvement(env, state_values)

for state in env.S:
    print("For state {}, {}".format(state, state_values[state]))
    for action in env.A:
        print(action, env.policy[state, action])
    print("\n\n")
