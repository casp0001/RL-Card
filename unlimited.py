import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make('no-limit-holdem', config={'seed': 0})

env.set_agents([RandomAgent(num_actions=env.num_actions)])

print(env.num_actions) # 2