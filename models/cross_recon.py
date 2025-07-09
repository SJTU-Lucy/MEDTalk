from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.encoder import Encoder
from layers.decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossReconNet(nn.Module):
    def __init__(self, args):
        super(CrossReconNet, self).__init__()
        if hasattr(args, "epoch"):
            self.epoch = args.epoch
        if hasattr(args, "save_path"):
            self.save_path = args.save_path
        if hasattr(args, "save_name"):
            self.save_name = args.save_name
        if hasattr(args, "gradient_accumulation_steps"):
            self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_duration = args.max_duration
        self.emo_encoder = Encoder(args.input_dim, args.hidden_dim, args.latent_dim)
        self.cont_encoder = Encoder(args.input_dim, args.hidden_dim, args.latent_dim)
        self.decoder = Decoder(2 * args.latent_dim, args.hidden_dim, args.input_dim)

        if hasattr(args, "lr"):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=0)

    def decode(self, cont_embedding, emo_embedding):
        concat_embedding = torch.cat((emo_embedding, cont_embedding), 2)
        x_recon = self.decoder(concat_embedding)

        return x_recon

    def forward_single(self, x):
        cont_embedding = self.cont_encoder(x)
        emo_embedding = self.emo_encoder(x)
        x_recon = self.decode(cont_embedding, emo_embedding)
        recon_loss = F.mse_loss(x_recon, x)

        return x_recon, recon_loss

    def forward_cross(self, x11, x12, x21, x22):
        cont_emb1 = self.cont_encoder(x11)
        cont_emb2 = self.cont_encoder(x22)
        emo_emb1 = self.emo_encoder(x11)
        emo_emb2 = self.emo_encoder(x22)
        recon11 = self.decode(cont_emb1, emo_emb1)
        recon12 = self.decode(cont_emb2, emo_emb1)
        recon21 = self.decode(cont_emb1, emo_emb2)
        recon22 = self.decode(cont_emb2, emo_emb2)

        recon_loss11 = F.mse_loss(x11, recon11)
        recon_loss12 = F.mse_loss(x12, recon12)
        recon_loss21 = F.mse_loss(x21, recon21)
        recon_loss22 = F.mse_loss(x22, recon22)

        total_loss = recon_loss11 + recon_loss12 + recon_loss21 + recon_loss22

        return ([recon11, recon12, recon21, recon22],
                [recon_loss11, recon_loss12, recon_loss21, recon_loss22],
                total_loss)

    def forward_cycle(self, x11, x22):
        cont_emb1 = self.cont_encoder(x11)
        cont_emb2 = self.cont_encoder(x22)
        emo_emb1 = self.emo_encoder(x11)
        emo_emb2 = self.emo_encoder(x22)
        recon12 = self.decode(cont_emb2, emo_emb1)
        recon21 = self.decode(cont_emb1, emo_emb2)

        cont_emb1 = self.cont_encoder(recon21)
        cont_emb2 = self.cont_encoder(recon12)
        emo_emb1 = self.emo_encoder(recon12)
        emo_emb2 = self.emo_encoder(recon21)
        recon11 = self.decode(cont_emb1, emo_emb1)
        recon22 = self.decode(cont_emb2, emo_emb2)

        recon_loss11 = F.mse_loss(x11, recon11)
        recon_loss22 = F.mse_loss(x22, recon22)
        total_loss = recon_loss11 + recon_loss22

        return ([recon11, recon22],
                [recon_loss11, recon_loss22],
                total_loss)

    def load_weight(self, weight_path):
        weight = torch.load(weight_path, map_location=device)
        self.load_state_dict(weight)
        self.eval()

    def do_train(self, train_loader):
        self.train_loader = train_loader
        # self-reconstruction
        self.do_reconstruct()
        # overlap exchange
        self.do_content()
        self.do_emotion()
        # cycle exchange
        self.do_cycle()

    def do_reconstruct(self):
        iteration = 0
        for e in range(self.epoch):
            loss_log = []
            self.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            self.optimizer.zero_grad()

            for i, data in pbar:
                iteration += 1
                rig = data["self"].to(torch.float32).to(device)
                x_recon, loss = self.forward_single(rig)
                loss.backward()
                loss_log.append(loss.item())
                if i % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                pbar.set_description("(Epoch {}, iteration {}) Recon loss:{:.7f}"
                        .format((e + 1), iteration, np.mean(loss_log)))

    def do_emotion(self):
        iteration = 0
        for e in range(self.epoch):
            loss_log = []
            loss11_log = []
            loss12_log = []
            loss21_log = []
            loss22_log = []
            self.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            self.optimizer.zero_grad()

            for i, data in pbar:
                iteration += 1
                rig01, rig02 = data["emotion"]
                rig01, rig02 = rig01.to(torch.float32).to(device), rig02.to(torch.float32).to(device)
                recon_list, loss_list, loss = self.forward_cross(rig01, rig02, rig01, rig02)
                loss.backward()
                loss_log.append(loss.item())
                loss11_log.append(loss_list[0].item())
                loss12_log.append(loss_list[1].item())
                loss21_log.append(loss_list[2].item())
                loss22_log.append(loss_list[3].item())
                if i % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                pbar.set_description("(Epoch {}, iteration {}) Loss11:{:.7f} Loss12:{:.7f} Loss21:{:.7f} "
                                     "Loss22:{:.7f} Total loss:{:.7f}"
                        .format((e + 1), iteration, np.mean(loss11_log), np.mean(loss12_log),
                                np.mean(loss21_log), np.mean(loss22_log), np.mean(loss_log)))

    def do_content(self):
        iteration = 0
        for e in range(self.epoch):
            loss_log = []
            loss11_log = []
            loss12_log = []
            loss21_log = []
            loss22_log = []
            self.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            self.optimizer.zero_grad()

            for i, data in pbar:
                iteration += 1
                rig10, rig20 = data["content"]
                rig10, rig20 = rig10.to(torch.float32).to(device), rig20.to(torch.float32).to(device)
                recon_list, loss_list, loss = self.forward_cross(rig10, rig10, rig20, rig20)
                loss.backward()
                loss_log.append(loss.item())
                loss11_log.append(loss_list[0].item())
                loss12_log.append(loss_list[1].item())
                loss21_log.append(loss_list[2].item())
                loss22_log.append(loss_list[3].item())
                if i % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                pbar.set_description("(Epoch {}, iteration {}) Loss11:{:.7f} Loss12:{:.7f} Loss21:{:.7f} "
                                     "Loss22:{:.7f} Total loss:{:.7f}"
                        .format((e + 1), iteration, np.mean(loss11_log), np.mean(loss12_log),
                                np.mean(loss21_log), np.mean(loss22_log), np.mean(loss_log)))

    def do_cycle(self):
        iteration = 0
        for e in range(self.epoch):
            loss_log = []
            loss11_log = []
            loss22_log = []
            self.train()
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            self.optimizer.zero_grad()

            for i, data in pbar:
                iteration += 1
                rig11, rig22 = data["cross"]
                rig11, rig22 = rig11.to(torch.float32).to(device), rig22.to(torch.float32).to(device)
                recon_list, loss_list, loss = self.forward_cycle(rig11, rig22)
                loss.backward()
                loss_log.append(loss.item())
                loss11_log.append(loss_list[0].item())
                loss22_log.append(loss_list[1].item())
                if i % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                pbar.set_description("(Epoch {}, iteration {}) Loss11:{:.7f} Loss22:{:.7f} Total loss:{:.7f}"
                        .format((e + 1), iteration, np.mean(loss11_log), np.mean(loss22_log), np.mean(loss_log)))

        torch.save(self.state_dict(), os.path.join(self.save_path, self.save_name))

    def read_rig(self, rig_path):
        rig_data = np.loadtxt(rig_path, delimiter=",")
        rig_data = torch.tensor(rig_data).to(torch.float32).to(device)
        rig_data = torch.unsqueeze(rig_data, dim=0)
        rig_length = rig_data.shape[1]

        return rig_data, rig_length

    def validate(self, rig_path, save_path):
        rig_data, rig_length = self.read_rig(rig_path)
        recon_rig, _ = self.forward_single(rig_data)
        recon_rig = torch.squeeze(recon_rig)
        recon_rig = recon_rig.detach().cpu().numpy()
        np.savetxt(save_path, recon_rig, delimiter=",")

    def test_cross(self, rig_path1, rig_path2, save_path):
        rig_data1, rig_length1 = self.read_rig(rig_path1)
        rig_data2, rig_length2 = self.read_rig(rig_path2)
        seq_len = min(rig_length1, rig_length2)
        rig_data1 = rig_data1[:, :seq_len]
        rig_data2 = rig_data2[:, :seq_len]

        recon_list, _, _ = self.forward_cross(rig_data1, rig_data1, rig_data2, rig_data2)
        recon11, recon12, recon21, recon22 = recon_list
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

    def get_emotion_embedding(self, rig_path):
        rig_data, rig_length = self.read_rig(rig_path)
        emo_embedding = self.emo_encoder(rig_data)
        emo_embedding = torch.squeeze(emo_embedding)

        return emo_embedding

    def get_content_embedding(self, rig_path):
        rig_data, rig_length = self.read_rig(rig_path)
        content_embedding = self.cont_encoder(rig_data)
        content_embedding = torch.squeeze(content_embedding)

        return content_embedding