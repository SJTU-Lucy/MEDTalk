import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.sequence_data import getSeqDataLoader
from data.audio_data import getAudioDataLoader
from data.text_data import getTextDataLoader
from data.clip_data import getCLIPDataLoader
from models.cross_recon import CrossReconNet
from models.audio_cont import AudioContentNet
from models.audio_semantic import AudioSemanticNet
from models.clip_emotion import CLIPNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_cross_reconstruction():
    parser = argparse.ArgumentParser(description='NETTalk Stage 1: self-supervised cross reconstruction')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="weights")
    parser.add_argument("--save_name", type=str, default="cross_recon.pth")
    parser.add_argument("--input_dim", type=int, default=174)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--emotion_dim", type=int, default=7)
    parser.add_argument("--max_duration", type=int, default=5000)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=15)
    args = parser.parse_args()

    model = CrossReconNet(args).to(device)
    train_loader = getSeqDataLoader("C:/Users/86134/Desktop/data/pair_rig")
    model.do_train(train_loader)


def train_audio_content():
    parser = argparse.ArgumentParser(description='NETTalk Stage 2: Audio-driven')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="weights")
    parser.add_argument("--save_name", type=str, default="audio_cont.pth")
    parser.add_argument("--input_dim", type=int, default=174)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--emotion_dim", type=int, default=7)
    parser.add_argument("--max_duration", type=int, default=5000)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()

    model = AudioContentNet(args).to(device)
    model.load_prev('weights/cross_recon.pth')
    model.freeze_params()
    train_loader, test_loader = getAudioDataLoader("C:/Users/86134/Desktop/data/all_emotions/train",
                                                   "C:/Users/86134/Desktop/data/all_emotions/validation")
    model.do_train(train_loader, test_loader)


def train_semantic_emotion():
    parser = argparse.ArgumentParser(description='NETTalk Stage 3: Audio-driven Emotion')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="weights")
    parser.add_argument("--save_name", type=str, default="audio_semantic.pth")
    parser.add_argument("--input_dim", type=int, default=174)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--emotion_dim", type=int, default=7)
    parser.add_argument("--max_duration", type=int, default=5000)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=20)
    args = parser.parse_args()

    model = AudioSemanticNet(args).to(device)
    model.load_prev('weights/audio_cont.pth')
    model.freeze_params()
    train_loader, test_loader = getTextDataLoader("C:/Users/86134/Desktop/data/all_emotions/train",
                                                  "C:/Users/86134/Desktop/data/all_emotions/validation",
                                                  "data/corpus_train.csv", "data/corpus_test.csv")
    model.do_train(train_loader, test_loader)


def train_CLIP():
    parser = argparse.ArgumentParser(description='NETTalk Stage 4: Image/Text Guide')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="weights")
    parser.add_argument("--save_name", type=str, default="clip_emotion.pth")
    parser.add_argument("--input_dim", type=int, default=174)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--emotion_dim", type=int, default=7)
    parser.add_argument("--max_duration", type=int, default=5000)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=15)
    args = parser.parse_args()

    model = CLIPNet(args).to(device)
    model.load_prev('weights/audio_cont.pth')
    model.freeze_params()
    train_loader = getCLIPDataLoader("C:/Users/86134/Desktop/data/RAVDESS")
    model.do_train(train_loader)


if __name__ == "__main__":
    train_cross_reconstruction()
    train_audio_content()
    train_semantic_emotion()
    train_CLIP()
