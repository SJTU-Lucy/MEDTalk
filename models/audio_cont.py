from tqdm import tqdm
import os
import math
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from data.sequence_data import process_rig
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2Config
from data.audio_data import read_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def LipLoss(pred_rig, gt_rig):
    indices = list(range(3, 13)) + list(range(46, 92)) + list(range(115, 161))
    pred_lip = pred_rig[:, :, indices]
    gt_lip = gt_rig[:, :, indices]
    lip_loss = nn.MSELoss()(pred_lip, gt_lip)

    return lip_loss


class AudioContentNet(nn.Module):
    def __init__(self, args):
        super(AudioContentNet, self).__init__()
        if hasattr(args, "epoch"):
            self.epoch = args.epoch
        if hasattr(args, "save_path"):
            self.save_path = args.save_path
        if hasattr(args, "save_name"):
            self.save_name = args.save_name
        self.max_duration = args.max_duration
        self.emo_encoder = Encoder(args.input_dim, args.hidden_dim, args.latent_dim)
        self.cont_encoder = Encoder(args.input_dim, args.hidden_dim, args.latent_dim)
        self.decoder = Decoder(2 * args.latent_dim, args.hidden_dim, args.input_dim)

        # load frozen wav2vec2.0 audio encoder
        self.audio_encoder_config = Wav2Vec2Config.from_pretrained("C:/Users/86134/Desktop/pretrain_weights/wav2vec2-base-960h",
                                                                   local_files_only=True)
        self.audio_content_encoder = Wav2Vec2Model.from_pretrained("C:/Users/86134/Desktop/pretrain_weights/wav2vec2-base-960h",
                                                                   local_files_only=True)
        self.audio_content_encoder.feature_extractor._freeze_parameters()
        hidden_size = self.audio_encoder_config.hidden_size
        self.audio_content_map = Encoder(hidden_size, args.hidden_dim, args.latent_dim)

        if hasattr(args, "lr"):
            self.optimizer = torch.optim.Adam(list(self.audio_content_map.parameters()),
                                              lr=args.lr, weight_decay=0)
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)

    def freeze_params(self):
        for name, param in self.emo_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.cont_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.audio_content_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.decoder.named_parameters():
            param.requires_grad = False

    def decode(self, cont_embedding, emo_embedding):
        concat_embedding = torch.cat((emo_embedding, cont_embedding), 2)
        x_recon = self.decoder(concat_embedding)

        return x_recon

    def forward(self, audio, label):
        seq_len = label.shape[1]
        embeddings = self.audio_content_encoder(audio, seq_len=seq_len, output_hidden_states=True)
        hidden_states = embeddings.last_hidden_state
        audio_cont = self.audio_content_map(hidden_states)

        motion_cont = self.cont_encoder(label)
        emo_embedding = self.emo_encoder(label)
        x_recon = self.decode(audio_cont, emo_embedding)
        cont_sim = F.cosine_similarity(audio_cont.view(-1), motion_cont.view(-1), dim=-1)

        emb_loss = 1 - cont_sim
        recon_loss = F.mse_loss(x_recon, label)
        lip_loss = LipLoss(x_recon, label)
        loss = recon_loss + 0.1 * emb_loss + lip_loss * 2

        return x_recon, recon_loss, emb_loss, loss

    def load_prev(self, weight_path):
        weight = torch.load(weight_path, map_location=device)
        model_dict = self.state_dict()
        state_dict = {k: v for k, v in weight.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def load_weight(self, weight_path):
        weight = torch.load(weight_path, map_location=device)
        self.load_state_dict(weight)
        self.eval()

    def do_train(self, train_loader, test_loader):
        iteration = 0
        for e in range(self.epoch):
            # training phase
            loss_log = []
            emb_log = []
            recon_log = []
            self.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, data in pbar:
                iteration += 1
                audio = data[0].to(torch.float32).to(device)
                rig = data[1].to(torch.float32).to(device)
                x_recon, recon_loss, emb_loss, loss = self(audio, rig)
                loss_log.append(loss.item())
                emb_log.append(emb_loss.item())
                recon_log.append(recon_loss.item())

                # gradient backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                pbar.set_description("(Epoch {}, iteration {}) Emb:{:.7f} Recon:{:.7f} Total:{:.7f}"
                                     .format((e + 1), iteration, np.mean(emb_log), np.mean(recon_log), np.mean(loss_log)))

            # valiation phase
            loss_log = []
            emb_log = []
            recon_log = []
            self.eval()
            for data in test_loader:
                audio = data[0].to(torch.float32).to(device)
                rig = data[1].to(torch.float32).to(device)
                x_recon, recon_loss, emb_loss, loss = self(audio, rig)
                loss_log.append(loss.item())
                emb_log.append(emb_loss.item())
                recon_log.append(recon_loss.item())
            print("(Validation Epoch {}) Emb:{:.7f} Recon:{:.7f} Total:{:.7f}"
                  .format((e + 1), np.mean(emb_log), np.mean(recon_log), np.mean(loss_log)))

            self.scheduler.step()

        torch.save(self.state_dict(), os.path.join(self.save_path, self.save_name))

    def load_data(self, audio_path, rig_path):
        audio_data, rig_data = read_data(audio_path, rig_path)
        audio_data = torch.tensor(audio_data).to(torch.float32).to(device)
        rig_data = torch.tensor(rig_data).to(torch.float32).to(device)
        audio_data = torch.unsqueeze(audio_data, dim=0)
        rig_data = torch.unsqueeze(rig_data, dim=0)
        audio_len = audio_data.shape[1]
        rig_len = rig_data.shape[1]

        return audio_data, rig_data, audio_len, rig_len

    def validate(self, audio_path, rig_path, save_path):
        audio_data, rig_data, _, _ = self.load_data(audio_path, rig_path)
        recon_rig, _, _, _ = self(audio_data, rig_data)
        recon_rig = torch.squeeze(recon_rig)
        recon_rig = recon_rig.detach().cpu().numpy()
        np.savetxt(save_path, recon_rig, delimiter=",")

    def test_cross(self, audio_path1, audio_path2, rig_path1, rig_path2, save_path):
        audio_data1, rig_data1, audio_length1, rig_length1 = self.load_data(audio_path1, rig_path1)
        audio_data2, rig_data2, audio_length2, rig_length2 = self.load_data(audio_path2, rig_path2)
        rig_len = min(rig_length1, rig_length2)
        audio_len = min(audio_length1, audio_length2)
        audio_data1 = audio_data1[:, :audio_len]
        audio_data2 = audio_data2[:, :audio_len]
        rig_data1 = rig_data1[:, :rig_len]
        rig_data2 = rig_data2[:, :rig_len]

        recon11, _, _, _ = self(audio_data1, rig_data1)
        recon12, _, _, _ = self(audio_data2, rig_data1)
        recon21, _, _, _ = self(audio_data1, rig_data2)
        recon22, _, _, _ = self(audio_data2, rig_data2)
        recon11 = torch.squeeze(recon11)
        recon12 = torch.squeeze(recon12)
        recon21 = torch.squeeze(recon21)
        recon22 = torch.squeeze(recon22)
        recon11 = recon11.detach().cpu().numpy()
        recon12 = recon12.detach().cpu().numpy()
        recon21 = recon21.detach().cpu().numpy()
        recon22 = recon22.detach().cpu().numpy()

        np.savetxt(os.path.join(save_path, "recon11.txt"), recon11, delimiter=",")
        np.savetxt(os.path.join(save_path, "recon12.txt"), recon12, delimiter=",")
        np.savetxt(os.path.join(save_path, "recon21.txt"), recon21, delimiter=",")
        np.savetxt(os.path.join(save_path, "recon22.txt"), recon22, delimiter=",")