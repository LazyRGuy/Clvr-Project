import matplotlib.pyplot as plt
import argparse
import torch
from torch.utils.data import DataLoader

from sprites_datagen.moving_sprites import MovingSpriteDataset
from sprites_datagen.rewards import *
from general_utils import *
from model import RewardPredictor,Encoder,MLP,LSTM,Decoder
import torch.nn as nn

def create_fig(spec):
    encoder_h, encoder_v = Encoder(), Encoder() 
    mlp_h, mlp_v = MLP(), MLP()
    lstm_h, lstm_v = LSTM(), LSTM()
    decoder_h, decoder_v = Decoder(), Decoder()
    encoder_h.load_state_dict(torch.load(f"weights/encoder-{'h'}"))
    mlp_h.load_state_dict(torch.load(f"weights/mlp-{'h'}"))
    lstm_h.load_state_dict(torch.load(f"weights/lstm-{'h'}"))
    decoder_h.load_state_dict(torch.load(f"weights/decoder-{'h'}"))
    encoder_v.load_state_dict(torch.load(f"weights/encoder-{'v'}"))
    mlp_v.load_state_dict(torch.load(f"weights/mlp-{'v'}"))
    lstm_v.load_state_dict(torch.load(f"weights/lstm-{'v'}"))
    decoder_v.load_state_dict(torch.load(f"weights/decoder-{'v'}"))
    test_data = MovingSpriteDataset(spec)
    test_dataloader = DataLoader(test_data, shuffle=False)
    batch = next(iter(test_dataloader))
    test_images = batch.images.squeeze(0).detach()
    test_input = test_images
    imgs = [test_input]
    encoder_h.eval(); mlp_h.eval(); lstm_h.eval(); decoder_h.eval()
    encoder_v.eval(); mlp_v.eval(); lstm_v.eval(); decoder_v.eval()
     
    with torch.no_grad():
        output_v = lstm_v(mlp_v(encoder_v(test_input)))
        output_v = decoder_v(output_v).detach().squeeze(0)
        imgs.append(output_v)

        output_h = lstm_h(mlp_h(encoder_h(test_input)))
        output_h = decoder_h(output_h).detach().squeeze(0)
        imgs.append(output_h)
        
    for i in range(3):
        img = imgs[i]
        img = img.numpy()
        img_max = img.max()
        if img.min() == img.max():
            img_max += 1
        img = (img-img.min())/(img_max-img.min())
        img = (img*255).astype(np.uint8)
        imgs[i] = img
    fig, axes = plt.subplots(3, 8, figsize=(12, 6))
    axes[0,0].set_title(f'start')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    axes[0, 0].imshow(imgs[0][0].squeeze(), cmap='gray')
    for i in range(1,8):
        axes[0, i].set_title(f'{4*i-3}')
        for j in range(3):
            axes[j,i].axis("off")
            axes[j,i].imshow(imgs[j][4*i-3].squeeze(), cmap='gray')
    plt.savefig(f"figure_2.jpg")
    plt.close()

def train_encoder(spec,tag):
    model = RewardPredictor()
    model.init_weights()
    train_data = MovingSpriteDataset(spec)
    train_dataloader = DataLoader(train_data, shuffle=False)
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3,betas=(0.9,0.999))
    loss_fn = nn.MSELoss()
    min_loss = float("inf")
    model.train()
    for _ in range(20):
        temp_loss = 0
        print("encoder",tag,"epoch",_," start")
        for batch in train_dataloader:
            optimizer.zero_grad()
            imgs = batch.images.squeeze(0)
            rewards = torch.stack([batch.rewards[r].reshape(-1) for r in batch.rewards.keys()],dim=1)
            y_pred = model(imgs)
            loss = loss_fn(y_pred, rewards)
            temp_loss += loss.item()
            loss.backward()
            optimizer.step()
        if temp_loss < min_loss:
            min_loss = temp_loss
            torch.save(model.encoder.state_dict(), f"weights/encoder-{tag}")
            torch.save(model.mlp.state_dict(), f"weights/mlp-{tag}")
            torch.save(model.lstm.state_dict(), f"weights/lstm-{tag}")
        print("encoder",tag,"epoch",_,"end loss:",temp_loss)

def train_decoder(spec,tag):
    train_data = MovingSpriteDataset(spec)
    train_dataloader = DataLoader(train_data,shuffle=False)
    encoder = Encoder()
    mlp = MLP()
    lstm = LSTM()
    decoder = Decoder()
    encoder.load_state_dict(torch.load(f"weights/encoder-{tag}"))
    mlp.load_state_dict(torch.load(f"weights/mlp-{tag}"))
    lstm.load_state_dict(torch.load(f"weights/lstm-{tag}"))
    decoder.init_weights()
    encoder.eval(),mlp.eval(),lstm.eval()
    decoder.train()
    optimizer = torch.optim.RAdam(decoder.parameters(), lr=1e-3,betas=(0.9,0.999))
    loss_fn = nn.MSELoss()
    min_loss = float("inf")
    for _ in range(20):
        temp_loss = 0
        print("decoder",tag,"epoch",_, "start")
        for batch in train_dataloader:
            optimizer.zero_grad()
            imgs = batch.images.squeeze(0)
            y_pred = lstm(mlp(encoder(imgs)))
            y_pred = decoder(y_pred)
            loss = loss_fn(y_pred,imgs)
            temp_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        if temp_loss < min_loss:
            min_loss = temp_loss
            torch.save(decoder.state_dict(), f"weights/decoder-{tag}")
        print("decoder",tag,"epoch",_,"end loss:",temp_loss)  

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1)
    args = parser.parse_args()
    specs_v = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=1,
        rewards = [VertPosReward]
    )

    specs_h = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=1,
        rewards = [HorPosReward]
    )
    if(args.train):
        best_model_v = train_encoder(specs_v,'v')
        best_decoder_v = train_decoder(specs_v,'v')
        best_model_h = train_encoder(specs_h,'h')
        best_decoder_h = train_decoder(specs_h,'h')
    create_fig(specs_v)