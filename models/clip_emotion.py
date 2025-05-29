from tqdm import tqdm
import os
import numpy as np
import librosa
from PIL import Image
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from transformers import Wav2Vec2Config, Wav2Vec2Processor
from transformers import CLIPProcessor, CLIPModel
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.wav2vec2 import Wav2Vec2Model
from funasr import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPNet(nn.Module):
    def __init__(self, args):
        super(CLIPNet, self).__init__()
        if hasattr(args, "epoch"):
            self.epoch = args.epoch
        if hasattr(args, "save_path"):
            self.save_path = args.save_path
        if hasattr(args, "save_name"):
            self.save_name = args.save_name
        self.max_duration = args.max_duration

        # basic AutoEncoder
        self.emo_encoder = Encoder(args.input_dim, args.hidden_dim, args.latent_dim)
        self.cont_encoder = Encoder(args.input_dim, args.hidden_dim, args.latent_dim)
        self.decoder = Decoder(2 * args.latent_dim, args.hidden_dim, args.input_dim)

        # audio content encoder
        self.audio_encoder_config = Wav2Vec2Config.from_pretrained("C:/Users/86134/Desktop/pretrain_weights/wav2vec2-base-960h",
                                                                   local_files_only=True)
        self.audio_content_encoder = Wav2Vec2Model.from_pretrained("C:/Users/86134/Desktop/pretrain_weights/wav2vec2-base-960h",
                                                                   local_files_only=True)
        self.audio_content_encoder.feature_extractor._freeze_parameters()
        hidden_size = self.audio_encoder_config.hidden_size
        self.audio_content_map = Encoder(hidden_size, args.hidden_dim, args.latent_dim)

        # audio and text embedding
        self.audio_emotion_encoder = AutoModel(model="iic/emotion2vec_base",
                                     disable_update=True,
                                     disable_log=True,
                                     disable_pbar=True,
                                     device="cuda")
        self.fusion_predictor = Encoder(input_dim=hidden_size, hidden_dim=256, latent_dim=1)

        # clip emotion embedding
        self.clip_encoder = CLIPModel.from_pretrained("C:/Users/86134/Desktop/pretrain_weights/clip-vit-base-patch32",
                                                      local_files_only=True)
        self.clip_processor = CLIPProcessor.from_pretrained("C:/Users/86134/Desktop/pretrain_weights/clip-vit-base-patch32",
                                                  local_files_only=True)
        self.clip_text_map = Encoder(512, 256, args.hidden_dim)
        self.clip_image_map = Encoder(512, 256, args.hidden_dim)

        # emotion encoder
        self.semantic_emotion_map = Encoder(args.hidden_dim, args.hidden_dim, args.latent_dim)

        # define optimizers
        if hasattr(args, "lr"):
            self.optimizer = torch.optim.Adam(list(self.semantic_emotion_map.parameters()) +
                                              list(self.clip_text_map.parameters()) +
                                              list(self.clip_image_map.parameters()),
                                              lr=args.lr, weight_decay=0)
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)

    def freeze_params(self):
        for name, param in self.emo_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.cont_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.decoder.named_parameters():
            param.requires_grad = False
        for name, param in self.audio_content_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.audio_content_map.named_parameters():
            param.requires_grad = False
        for name, param in self.fusion_predictor.named_parameters():
            param.requires_grad = False
        # for name, param in self.semantic_emotion_map.named_parameters():
        #     param.requires_grad = False

    def decode(self, cont_embedding, emo_embedding):
        concat_embedding = torch.cat((emo_embedding, cont_embedding), 2)
        x_recon = self.decoder(concat_embedding)

        return x_recon

    def forward(self, audio, label, image_path, text_path):
        seq_len = label.shape[1]
        # audio content embedding
        embeddings = self.audio_content_encoder(audio, seq_len=seq_len, output_hidden_states=True)
        hidden_states = embeddings.last_hidden_state
        audio_cont = self.audio_content_map(hidden_states)

        # audio emotion embedding
        res = self.audio_emotion_encoder.generate(audio, granularity="frame", extract_embedding=True)[0]
        audio_feature = res['feats']
        audio_feature = torch.from_numpy(audio_feature)
        audio_feature = torch.unsqueeze(audio_feature, dim=0).to(device)
        # interpolate from 50 fps -> 60 fps
        audio_feature = audio_feature.transpose(1, 2)
        audio_feature = F.interpolate(audio_feature, size=seq_len, align_corners=True, mode='linear')
        audio_feature = audio_feature.transpose(1, 2)
        # get emotion embedding by adjust norm
        pred_intensity = self.fusion_predictor(audio_feature)

        # text emotion embedding
        with open(text_path, "r") as f:
            text = f.read()
        text = [text]
        text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True).to(device)
        text_feature = self.clip_encoder.get_text_features(input_ids=text_inputs['input_ids'].to(device),
                                                           attention_mask=text_inputs['attention_mask'].to(device))
        text_emo = self.clip_text_map(text_feature)
        norm = torch.norm(text_emo, dim=-1, keepdim=True)
        text_emo_feature = text_emo * (pred_intensity / (norm + 1e-8))
        text_emo_feature = self.semantic_emotion_map(text_emo_feature)

        # image emotion embedding
        image = Image.open(image_path)
        image_inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
        image_feature = self.clip_encoder.get_image_features(**image_inputs)
        image_emo = self.clip_image_map(image_feature)
        norm = torch.norm(image_emo, dim=-1, keepdim=True)
        image_emo_feature = image_emo * (pred_intensity / (norm + 1e-8))
        image_emo_feature = self.semantic_emotion_map(image_emo_feature)

        motion_emo = self.emo_encoder(label)
        x_recon_text = self.decode(audio_cont, text_emo_feature)
        x_recon_image = self.decode(audio_cont, image_emo_feature)
        text_emo_sim = F.cosine_similarity(text_emo_feature.view(-1), motion_emo.view(-1), dim=-1)
        image_emo_sim = F.cosine_similarity(image_emo_feature.view(-1), motion_emo.view(-1), dim=-1)

        emb_loss = 2 - image_emo_sim - text_emo_sim
        recon_loss = F.mse_loss(x_recon_text, label) + F.mse_loss(x_recon_image, label)
        loss = recon_loss + 0.1 * emb_loss

        return [x_recon_text, x_recon_image], [recon_loss, emb_loss, loss]

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

    def do_train(self, train_loader):
        iteration = 0
        for e in range(self.epoch):
            # training phase
            recon_log = []
            emb_log = []
            loss_log = []
            self.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, data in pbar:
                iteration += 1
                self.optimizer.zero_grad()
                audio = data[0].to(torch.float32).to(device)
                rig = data[1].to(torch.float32).to(device)
                image_path = data[2][0]
                text_path = data[3][0]
                x_recon, loss_list = self(audio, rig, image_path, text_path)
                recon_log.append(loss_list[0].item())
                emb_log.append(loss_list[1].item())
                loss_log.append(loss_list[2].item())
                loss_list[2].backward()
                self.optimizer.step()

                pbar.set_description("(Epoch {}, iteration {}) Emb:{:.7f} Recon:{:.7f} Total:{:.7f}"
                                     .format((e + 1), iteration, np.mean(emb_log), np.mean(recon_log), np.mean(loss_log)))

            self.scheduler.step()

        torch.save(self.state_dict(), os.path.join(self.save_path, self.save_name))

    def forward_validate(self, audio, seq_len, image_path=None, text_path=None):
        # audio content embedding
        embeddings = self.audio_content_encoder(audio, seq_len=seq_len, output_hidden_states=True)
        hidden_states = embeddings.last_hidden_state
        audio_cont = self.audio_content_map(hidden_states)

        # audio emotion embedding
        res = self.audio_emotion_encoder.generate(audio, granularity="frame", extract_embedding=True)[0]
        audio_feature = res['feats']
        audio_feature = torch.from_numpy(audio_feature)
        audio_feature = torch.unsqueeze(audio_feature, dim=0).to(device)
        # interpolate from 50 fps -> 60 fps
        audio_feature = audio_feature.transpose(1, 2)
        audio_feature = F.interpolate(audio_feature, size=seq_len, align_corners=True, mode='linear')
        audio_feature = audio_feature.transpose(1, 2)
        pred_intensity = self.fusion_predictor(audio_feature)

        # text emotion embedding
        if text_path is not None:
            with open(text_path, "r") as f:
                text = f.read()
            text = [text]
            text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True).to(device)
            text_feature = self.clip_encoder.get_text_features(input_ids=text_inputs['input_ids'].to(device),
                                                               attention_mask=text_inputs['attention_mask'].to(device))
            text_emo = self.clip_text_map(text_feature)
            norm = torch.norm(text_emo, dim=-1, keepdim=True)
            text_emo_feature = text_emo * (pred_intensity / (norm + 1e-8))
            text_emo_feature = self.semantic_emotion_map(text_emo_feature)
            x_recon_text = self.decode(audio_cont, text_emo_feature)

            return x_recon_text

        if image_path is not None:
            # image emotion embedding
            image = Image.open(image_path)
            image_inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
            image_feature = self.clip_encoder.get_image_features(**image_inputs)
            image_emo = self.clip_image_map(image_feature)
            norm = torch.norm(image_emo, dim=-1, keepdim=True)
            image_emo_feature = image_emo * (pred_intensity / (norm + 1e-8))
            image_emo_feature = self.semantic_emotion_map(image_emo_feature)
            x_recon_image = self.decode(audio_cont, image_emo_feature)
            return x_recon_image

        return None

    def load_data(self, audio_path):
        # read audio data
        processor = Wav2Vec2Processor.from_pretrained("C:/Users/86134/Desktop/pretrain_weights/wav2vec2-base-960h",
                                                      local_files_only=True)
        sampling_rate = 16000

        speech_array, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
        audio_data = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)
        audio_data = torch.tensor(audio_data).to(torch.float32).to(device)
        audio_data = torch.unsqueeze(audio_data, dim=0)
        audio_len = audio_data.shape[1]
        seq_len = int(audio_len * 60 / 16000)

        return audio_data, seq_len

    def validate(self, audio_path, save_path, image_path=None, text_path=None):
        audio_data, seq_len = self.load_data(audio_path)
        recon_rig = self.forward_validate(audio_data, seq_len, image_path, text_path)
        recon_rig = torch.squeeze(recon_rig)
        recon_rig = recon_rig.detach().cpu().numpy()
        recon_rig = recon_rig.T
        recon_rig = signal.savgol_filter(recon_rig, window_length=5, polyorder=2, mode="nearest").T
        np.savetxt(save_path, recon_rig, delimiter=",")

    def forward_demo(self, audio_data, rig_data, image_path):
        seq_len = rig_data.shape[1]
        cont_emb = self.cont_encoder(rig_data)

        # audio emotion embedding
        res = self.audio_emotion_encoder.generate(audio_data, granularity="frame", extract_embedding=True)[0]
        audio_feature = res['feats']
        audio_feature = torch.from_numpy(audio_feature)
        audio_feature = torch.unsqueeze(audio_feature, dim=0).to(device)
        # interpolate from 50 fps -> 60 fps
        audio_feature = audio_feature.transpose(1, 2)
        audio_feature = F.interpolate(audio_feature, size=seq_len, align_corners=True, mode='linear')
        audio_feature = audio_feature.transpose(1, 2)
        pred_intensity = self.fusion_predictor(audio_feature)

        # image emotion embedding
        image = Image.open(image_path)
        image_inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
        image_feature = self.clip_encoder.get_image_features(**image_inputs)
        image_emo = self.clip_image_map(image_feature)
        norm = torch.norm(image_emo, dim=-1, keepdim=True)
        image_emo_feature = image_emo * (pred_intensity / (norm + 1e-8))
        image_emo_feature = self.semantic_emotion_map(image_emo_feature)
        x_recon_image = self.decode(cont_emb, image_emo_feature)

        return x_recon_image

    def demo(self, audio_path, rig_path, image_path):
        rig_data = np.loadtxt(rig_path, delimiter=",")
        rig_data = torch.tensor(rig_data).to(torch.float32).to(device)
        rig_data = torch.unsqueeze(rig_data, dim=0)
        audio_data, _ = self.load_data(audio_path)

        recon_rig = self.forward_demo(audio_data, rig_data, image_path)
        recon_rig = torch.squeeze(recon_rig)
        recon_rig = recon_rig.detach().cpu().numpy()
        recon_rig = recon_rig.T
        recon_rig = signal.savgol_filter(recon_rig, window_length=5, polyorder=2, mode="nearest").T

        return recon_rig

