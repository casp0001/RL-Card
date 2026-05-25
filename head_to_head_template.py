from dqn_agent_relu import DQNAgent as DQNAgentReLU
from rlcard.agents import DQNAgent
import rlcard
import torch
import matplotlib.pyplot as plt
import os

from rlcard.utils import (
    reorganize,
    set_seed,
    tournament,
    Logger,
    plot_curve,
)


env = rlcard.make('limit-holdem', config={'seed': 0})
vanilla_DQN_agent = DQNAgent.from_checkpoint(torch.load(
        'experiments/limit_holdem_dqn_selfplay/activationrelu_lr0.001_bs32_buf40000_layers128_128_128_eps40000_opp1000_seed42/checkpoint_dqn.pt',
        map_location='cpu',weights_only=False
    ))
custom_DQN_agent = DQNAgentReLU.from_checkpoint(torch.load(
        'experiments/limit_holdem_dqn_selfplay/activationtanh_lr0.001_bs32_buf40000_layers128_128_128_eps40000_opp1000_seed42/checkpoint_dqn.pt',
        map_location='cpu',weights_only=False
    ))

env.set_agents([vanilla_DQN_agent, custom_DQN_agent])

payoffs = tournament(env, num=10000)

print(f'Vanilla DQN: {payoffs[0]:+.4f}')
print(f'Custom DQN: {payoffs[1]:+.4f}')

agents = ['Vanilla DQN', 'Custom DQN']
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(agents, payoffs, color=['steelblue', 'tomato'])
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.bar_label(bars, fmt='{:+.4f}', padding=4)
ax.set_ylabel('Average Payoff')
ax.set_title('Tournament Results (1000 games)')
plt.tight_layout()
os.makedirs('head_to_head_template', exist_ok=True)
plt.savefig('head_to_head_template/tournament.png', dpi=150)
plt.show()


