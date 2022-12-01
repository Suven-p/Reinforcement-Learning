import numpy as np
import gridWorld
from dp import policy_evaluation, policy_improvement

env = gridWorld.Env()

policy_evaluation(env)
policy_improvement(env)
policy_evaluation(env)
policy_improvement(env)

for k, v in env.policy_table.items():
    print("State:", k, "Action:", v)
