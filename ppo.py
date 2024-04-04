import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from sprites_datagen.rewards import *
from sprites_env import *

from general_utils import *

import numpy as np
import torch
import torch.nn as nn
from model import Encoder
from stable_baselines3.common.distributions import DiagGaussianDistribution

class Others(nn.Module):
    def __init__(self, observation_space, action_space, baseline):
        super(Others, self).__init__()

        self.w, self.h = observation_space.shape[0], observation_space.shape[1]
        self.baseline = baseline

        self.encoder = Encoder()
        wts = 'weights/encoder-ns-4.pt'
        if 'reward_prediction' in baseline:
            self.encoder.load_state_dict(torch.load(wts))
        if 'finetune' not in baseline and baseline != 'image_scratch':
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.policy_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def init_weights(self):
        if 'finetune' not in self.baseline:
            for layer in self.encoder.modules():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        
        for layer in self.policy_net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.value_net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.view(-1, 1, self.w, self.h)
        x = self.encoder(obs)
        return self.policy_net(x), self.value_net(x)
    
class Oracle(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Oracle, self).__init__()
        input_size = observation_space.shape[0]

        self.policy_net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        self.value_net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def init_weights(self):
        for layer in self.policy_net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.value_net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        return self.policy_net(obs), self.value_net(obs)

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,baseline):
        super(ActorCritic, self).__init__()
        if baseline == 'oracle':
            self.features_extractor = Oracle(observation_space, action_space)
        else:
            self.features_extractor = Others(observation_space, action_space, baseline)
        
        self.action_dist = DiagGaussianDistribution(action_space.shape[0])
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=64, log_std_init=0.0)
        
    def init_weights(self):
        self.features_extractor.init_weights()
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        policy_net_output, value_net_output = self.features_extractor(obs)
        mean_actions = self.action_net(policy_net_output)
        dist = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = dist.sample()
        actions_log_probs = dist.log_prob(actions)
        return actions, value_net_output, actions_log_probs
    
    def evaluate_actions(self, obs, actions):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        policy_net_output, value_net_output = self.features_extractor(obs)
        mean_actions = self.action_net(policy_net_output)
        dist = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = actions.view(-1, 2)
        actions_log_probs = dist.log_prob(actions).view(-1, 1)
        return value_net_output, actions_log_probs, dist.entropy()

class PPO:
    def __init__(self, baseline, env, wandb, **hyperparameters):
        self._init_hyperparameters(hyperparameters)
        self.wandb = wandb
        
        self.baseline = baseline
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.num_distractors = env.spec.id[-1]
        self.actor_critic = ActorCritic(env.observation_space, env.action_space,baseline)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        
        self.max_grad_norm = 0.5
    
    def learn(self, total_timesteps):
        print(f"TRAINING WITH BASELINE CLASS: {self.baseline}")
        
        self.t = 0
        while self.t < total_timesteps:
            batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_rtgs = self.policy_rollout()
            with torch.no_grad():
                values, _, _ = self.actor_critic.evaluate_actions(batch_obs, batch_actions)
                values = values.view(-1)
            A_k = batch_rtgs - values
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            for _ in range(self.n_updates_per_iteration):
                self.update_policy(batch_obs, batch_actions, batch_log_probs, batch_rtgs, A_k)
            print("Steps taken:", self.t)
            if self.wandb:
                wandb_title = f"test"
                for batch_reward in batch_rewards:
                    for step_reward in batch_reward:
                        wandb.log({wandb_title: (step_reward+0.6)* self.max_timesteps_per_episode})

    def policy_rollout(self):
        batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_rtgs = [], [], [], [], []
        
        t = 0
        
        while t < self.timesteps_per_batch:
            eps_rewards = []
            obs = self.env.reset()
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                self.t += 1
                t += 1
                with torch.no_grad():
                    obs = torch.tensor(obs, dtype=torch.float)
                    batch_obs.append(obs)
                    action, _, log_prob = self.actor_critic(obs)
                action, log_prob = action.numpy(), log_prob.numpy()
                obs, reward, done, _ = self.env.step(action)
                eps_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob.reshape(-1))
                if done:
                    break
            batch_rewards.append(eps_rewards)
        with torch.no_grad():
            batch_rtgs = self.compute_rtgs(batch_rewards)
            
        batch_obs = torch.stack(batch_obs)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
            
        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_rtgs
    
    
    def update_policy(self, batch_obs, batch_actions, batch_log_probs, batch_rtgs, A_k):
        values, curr_log_probs, dist_entropy = self.actor_critic.evaluate_actions(batch_obs, batch_actions)
        values = values.view(-1)
        
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        ratios = ratios.view(-1)
        
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(batch_rtgs, values)
        entropy_loss = torch.mean(dist_entropy)
        
        self.optimizer.zero_grad()
        
        total_loss = value_loss * self.value_loss_coef + policy_loss - self.entropy_coef * entropy_loss
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.append(discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).flip(0)
        return batch_rtgs
    
    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 2080
        self.max_timesteps_per_episode = 40
        self.n_updates_per_iteration = 10
        self.gamma = 0.99
        self.clip = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.lr = 5e-4
        
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))