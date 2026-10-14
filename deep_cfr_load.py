''' Load a Deep CFR checkpoint written by limit_deepcfr_train_custom.py.

Usage:
    python deep_cfr_load.py --ckpt experiments/deepcfr/.../model.ckpt \
        --policy_layers 128 128 128 --advantage_layers 128 128 128
'''
import argparse

import torch
import rlcard

from deep_cfr_agent_new import DeepCFR, _DEVICE


def load_deep_cfr(ckpt_path, env, policy_layers, advantage_layers, load_optimizers=True):
    agent = DeepCFR(
        env=env,
        policy_network_layers=tuple(policy_layers),
        advantage_network_layers=tuple(advantage_layers),
    )
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)
    agent._policy_network.load_state_dict(ckpt['policy_network'])
    if load_optimizers:
        agent._optimizer_policy.load_state_dict(ckpt['optimizer_policy'])
    for p in range(env.num_players):
        agent._advantage_networks[p].load_state_dict(ckpt[f'advantage_network_{p}'])
        if load_optimizers:
            agent._optimizer_advantages[p].load_state_dict(ckpt[f'optimizer_advantage_{p}'])
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Load a Deep CFR checkpoint')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--policy_layers', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('--advantage_layers', type=int, nargs='+', default=[128, 128, 128])
    args = parser.parse_args()

    env = rlcard.make('limit-holdem', config={'allow_step_back': True})
    agent = load_deep_cfr(args.ckpt, env, args.policy_layers, args.advantage_layers)
    print(f'Loaded {args.ckpt}')
