import numpy as np
import gridWorld2 as gw
from dp2 import policy_evaluation, policy_improvement

env = gw.Env()

state_values = policy_evaluation(env)
policy_improvement(env, state_values)
policy_evaluation(env, state_values)
policy_improvement(env, state_values)

for state in env.S:
    print("For state {}".format(state))
    for action in env.A:
        print(action, env.policy(state, action))
    print("\n\n")
        
