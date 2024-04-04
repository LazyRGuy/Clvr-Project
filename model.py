import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sprites_datagen.rewards import *


class Encoder(nn.Module):
    def __init__(self, output_dim=64):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1),       
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),       
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),      
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),     
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),       
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(in_features=128, out_features=output_dim)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)                
        x = x.view(x.size(0), -1)
        x = self.fc(x)                  
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(input_dim, 128)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),     
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.ReLU()        
        )
    
    def init_weights(self):
        # Initialize weights for the decoder
        for layer in self.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), -1, 1, 1)
        x = self.deconv(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class RewardPredictor(nn.Module):
    def __init__(self,
                 hidden_dim=64,
                 rewards=[HorPosReward]
                 ):
        super(RewardPredictor, self).__init__()
        
        self.encoder = Encoder(output_dim=hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim)
        self.lstm = LSTM(hidden_dim, hidden_dim)
        self.reward_heads = [MLP(hidden_dim, 1) for _ in range(len(rewards))]

    def init_weights(self):
        for layer in self.encoder.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.mlp.init_weights()
        for reward_head in self.reward_heads:
            reward_head.init_weights()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        x = self.lstm(x)
        
        rewards = []
        for reward_head in self.reward_heads:
            reward = reward_head(x)
            rewards.append(reward)
        
        rewards = torch.stack(rewards,dim=1)
        rewards = rewards.squeeze(-1) 
        
        return rewards

        
        