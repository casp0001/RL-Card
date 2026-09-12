ckpt = torch.load('experiments/.../model.ckpt')
agent._policy_network.load_state_dict(ckpt['policy_network'])
agent._optimizer_policy.load_state_dict(ckpt['optimizer_policy'])
for p in range(env.player_num):
    agent._advantage_networks[p].load_state_dict(ckpt[f'advantage_network_{p}'])
    agent._optimizer_advantages[p].load_state_dict(ckpt[f'optimizer_advantage_{p}'])