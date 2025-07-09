from tqdm import tqdm
import os
import librosa
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from transformers import Wav2Vec2Config, Wav2Vec2Processor, BertModel, BertTokenizer
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.cross_modal_encoder import AudioTextFusion
from layers.wav2vec2 import Wav2Vec2Model
from funasr import AutoModel
import whisper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_id = {"ang": 0, "dis": 1, "fea": 2, "hap": 3, "neu": 4, "sad": 5, "sur": 6}


def get_intensity(rig, scale=1/10):
    intensity_index = [29, 30, 31, 32, 98, 99, 100, 101] + [36, 43, 105, 112] + [53, 122]
    inten_data = rig[:, :, intensity_index]
    intensity = torch.sum(torch.abs(inten_data), dim=2, keepdim=True)
    intensity = intensity * scale

    return intensity


class AudioSemanticNet(nn.Module):
    def __init__(self, args):
        super(AudioSemanticNet, self).__init__()
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
        self.audio_encoder_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_content_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_content_encoder.feature_extractor._freeze_parameters()
        hidden_size = self.audio_encoder_config.hidden_size
        self.audio_content_map = Encoder(hidden_size, args.hidden_dim, args.latent_dim)

        # audio and text embedding
        self.audio_emotion_encoder = AutoModel(model="iic/emotion2vec_base",
                                     disable_update=True,
                                     disable_log=True,
                                     disable_pbar=True,
                                     device="cuda")
        self.text_tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.text_encoder = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.fusion_predictor = AudioTextFusion(embed_dim=hidden_size, num_heads=8)

        # emotion embedding
        self.emotion_embedding = nn.Sequential(
            nn.Embedding(7, 256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, args.hidden_dim)
        )
        # emotion encoder
        self.semantic_emotion_map = Encoder(args.hidden_dim, args.hidden_dim, args.latent_dim)

        # define optimizers
        if hasattr(args, "lr"):
            self.optimizer = torch.optim.Adam(list(self.fusion_predictor.parameters()) +
                                              list(self.emotion_embedding.parameters()) +
                                              list(self.semantic_emotion_map.parameters()),
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
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False

    def decode(self, cont_embedding, emo_embedding):
        concat_embedding = torch.cat((emo_embedding, cont_embedding), 2)
        x_recon = self.decoder(concat_embedding)

        return x_recon

    def forward(self, audio, label, emo, text):
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

        # text embedding
        inputs = self.text_tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            return_tensors='pt',
        ).to(device)
        outputs = self.text_encoder(**inputs)
        text_feature = outputs.last_hidden_state

        # get emotion embedding by adjust norm
        pred_intensity = self.fusion_predictor(audio_feature, text_feature, text_feature)
        emo_embed = self.emotion_embedding(emo)
        norm = torch.norm(emo_embed, dim=-1, keepdim=True)
        emo_feature = emo_embed * (pred_intensity / (norm + 1e-8))
        emo_feature = self.semantic_emotion_map(emo_feature)

        motion_emo = self.emo_encoder(label)
        x_recon = self.decode(audio_cont, emo_feature)

        # loss calculation
        gt_intensity = get_intensity(label)
        out_intensity = get_intensity(x_recon)
        inten_loss = F.mse_loss(out_intensity, gt_intensity)
        cont_sim = F.cosine_similarity(emo_feature.view(-1), motion_emo.view(-1), dim=-1)
        emb_loss = 1 - cont_sim
        recon_loss = F.mse_loss(x_recon, label)
        loss = recon_loss + 0.1 * emb_loss + 0.1 * inten_loss

        return x_recon, [recon_loss, emb_loss, inten_loss, loss]

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
            recon_log = []
            emb_log = []
            inten_log = []
            loss_log = []
            self.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, data in pbar:
                iteration += 1
                self.optimizer.zero_grad()
                audio = data[0].to(torch.float32).to(device)
                rig = data[1].to(torch.float32).to(device)
                emotion = data[2].to(torch.long).to(device)
                text = data[3]
                x_recon, loss_list = self(audio, rig, emotion, text)
                recon_log.append(loss_list[0].item())
                emb_log.append(loss_list[1].item())
                inten_log.append(loss_list[2].item())
                loss_log.append(loss_list[3].item())
                loss_list[3].backward()
                self.optimizer.step()

                pbar.set_description("(Epoch {}, iteration {}) Emb:{:.7f} Recon:{:.7f} Intensity:{:.7f} Total:{:.7f}"
                                     .format((e + 1), iteration, np.mean(emb_log), np.mean(recon_log),
                                             np.mean(inten_log), np.mean(loss_log)))

            # valiation phase
            recon_log = []
            emb_log = []
            inten_log = []
            loss_log = []
            self.eval()
            for data in test_loader:
                audio = data[0].to(torch.float32).to(device)
                rig = data[1].to(torch.float32).to(device)
                emotion = data[2].to(torch.long).to(device)
                text = data[3]
                x_recon, loss_list = self(audio, rig, emotion, text)
                recon_log.append(loss_list[0].item())
                emb_log.append(loss_list[1].item())
                inten_log.append(loss_list[2].item())
                loss_log.append(loss_list[3].item())
            print("Epoch {} Emb:{:.7f} Recon:{:.7f} Intensity:{:.7f} Total:{:.7f}"
                  .format((e + 1), np.mean(emb_log), np.mean(recon_log),
                          np.mean(inten_log), np.mean(loss_log)))

            self.scheduler.step()

        torch.save(self.state_dict(), os.path.join(self.save_path, self.save_name))

    def forward_validate(self, audio, seq_len, emo, text):
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

        # text embedding
        inputs = self.text_tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            return_tensors='pt',
        ).to(device)
        outputs = self.text_encoder(**inputs)
        text_feature = outputs.last_hidden_state

        # get emotion embedding by adjust norm
        pred_intensity = self.fusion_predictor(audio_feature, text_feature, text_feature)
        emo_embed = self.emotion_embedding(emo)
        norm = torch.norm(emo_embed, dim=-1, keepdim=True)
        emo_feature = emo_embed * (pred_intensity / (norm + 1e-8))
        emo_feature = self.semantic_emotion_map(emo_feature)

        x_recon = self.decode(audio_cont, emo_feature)

        return x_recon

    def load_data(self, audio_path):
        # read audio data
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        sampling_rate = 16000

        speech_array, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
        audio_data = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)

        # get text data
        result = self.asr.transcribe(audio_path, language='Chinese')
        text = result["text"]

        audio_data = torch.tensor(audio_data).to(torch.float32).to(device)
        audio_data = torch.unsqueeze(audio_data, dim=0)
        text = [text]
        audio_len = audio_data.shape[1]
        seq_len = int(audio_len * 60 / 16000)
        print(text, seq_len)

        return audio_data, seq_len, text

    def validate(self, audio_path, emotion, save_path):
        self.asr = whisper.load_model("base")
        audio_data, seq_len, text = self.load_data(audio_path)
        emo = emotion_id[emotion[0:3]]
        emo = torch.tensor([emo]).to(torch.long).to(device)
        recon_rig = self.forward_validate(audio_data, seq_len, emo=emo, text=text)
        recon_rig = torch.squeeze(recon_rig)
        recon_rig = recon_rig.detach().cpu().numpy()
        recon_rig = recon_rig.T
        recon_rig = signal.savgol_filter(recon_rig, window_length=5, polyorder=2, mode="nearest").T
        np.savetxt(save_path, recon_rig, delimiter=",")

    def get_emotion_embedding(self, audio_path):
        self.asr = whisper.load_model("base")
        audio_data, seq_len, emotion, text = self.load_data(audio_path)
        # audio emotion embedding
        res = self.audio_emotion_encoder.generate(audio_data, granularity="frame", extract_embedding=True)[0]
        audio_feature = res['feats']
        audio_feature = torch.from_numpy(audio_feature)
        audio_feature = torch.unsqueeze(audio_feature, dim=0).to(device)
        # interpolate from 50 fps -> 60 fps
        audio_feature = audio_feature.transpose(1, 2)
        audio_feature = F.interpolate(audio_feature, size=seq_len, align_corners=True, mode='linear')
        audio_feature = audio_feature.transpose(1, 2)

        # text embedding
        inputs = self.text_tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            return_tensors='pt',
        ).to(device)
        outputs = self.text_encoder(**inputs)
        text_feature = outputs.last_hidden_state

        # get emotion embedding by adjust norm
        pred_intensity = self.fusion_predictor(audio_feature, text_feature, text_feature)
        emo_embed = self.emotion_embedding(emotion)
        norm = torch.norm(emo_embed, dim=-1, keepdim=True)
        emo_feature = emo_embed * (pred_intensity / (norm + 1e-8))
        emo_feature = self.semantic_emotion_map(emo_feature)
        emo_feature = torch.squeeze(emo_feature)

        return emo_feature

    def custom_validate(self, audio_path, save_path, label):
        self.asr = whisper.load_model("base")
        audio_data, seq_len, _, text = self.load_data(audio_path)
        emotion = torch.tensor([label]).to(torch.long).to(device)
        recon_rig = self.forward_validate(audio_data, seq_len, emo=emotion, text=text)
        recon_rig = torch.squeeze(recon_rig)
        recon_rig = recon_rig.detach().cpu().numpy()
        recon_rig = recon_rig.T
        recon_rig = signal.savgol_filter(recon_rig, window_length=5, polyorder=2, mode="nearest").T
        np.savetxt(save_path, recon_rig, delimiter=",")