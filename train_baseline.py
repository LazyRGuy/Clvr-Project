import gym
import argparse
import wandb
from ppo import PPO
from sprites_env import *

def train_baseline(args):
    if args.w:
        wandb.login()
        wandb.init(project="clvr-implementation-project", name=args.b)
    env_type = 'SpritesState' if args.b == 'oracle' else 'Sprites'
    env_id = f"{env_type}-v{args.n}"
    env = gym.make(env_id)
    ppo = PPO(baseline=args.b, env=env,total_timesteps=args.t, wandb=args.w)
    ppo.learn(args.t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training baselines')
    parser.add_argument('-b', type=str, default='oracle', help='baseline class')
    parser.add_argument('-n', type=int, default=0, help='number of distractors')
    parser.add_argument('-t', type=int, default=500000, help='total timesteps')
    parser.add_argument('-w', type=int, default=0, help='use wandb')
    args = parser.parse_args()
    train_baseline(args)
