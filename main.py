import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.sequence_data import SeqDataset, getSeqDataLoader
from data.audio_data import AudioDataset, getAudioDataLoader
from data.text_data import TextDataset, getTextDataLoader
from data.clip_data import CLIPDataset, getCLIPDataLoader
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


def test_cross_reconstruction():
    parser = argparse.ArgumentParser(description='NETTalk Stage 1 self-supervised cross reconstruction')
    parser.add_argument("--input_dim", type=int, default=174)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--emotion_dim", type=int, default=7)
    parser.add_argument("--max_duration", type=int, default=5000)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--weight_path", type=str, default="weights/cross_recon.pth")
    args = parser.parse_args()

    rig_path = "C:/Users/86134/Desktop/data/all_emotions/validation/VALID_CTR"
    save_path = "result/cross_reconstruction"

    # build model
    model = CrossReconNet(args).to(device)
    model.load_weight(args.weight_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for root, dirs, files in os.walk(rig_path):
        save_root = root.replace(rig_path, save_path)
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        for file in files:
            if file.endswith(".txt"):
                rig_file = os.path.join(root, file)
                save_file = rig_file.replace(rig_path, save_path)
                print(rig_file, save_file)
                model.validate(rig_file, save_file)


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


def test_audio_content():
    parser = argparse.ArgumentParser(description='NETTalk Stage 2: Audio-driven')
    parser.add_argument("--input_dim", type=int, default=174)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--emotion_dim", type=int, default=7)
    parser.add_argument("--max_duration", type=int, default=5000)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--weight_path", type=str, default="weights/audio_cont.pth")
    args = parser.parse_args()

    data_path = "C:/Users/86134/Desktop/data/all_emotions/validation"
    rig_path = os.path.join(data_path, "VALID_CTR")
    audio_path = os.path.join(data_path, "WAV")
    save_path = "result/audio_content"

    # build model
    model = AudioContentNet(args).to(device)
    model.load_weight(args.weight_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for root, dirs, files in os.walk(rig_path):
        save_root = root.replace(rig_path, save_path)
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        for file in files:
            if file.endswith(".txt"):
                rig_file = os.path.join(root, file)
                audio_file = rig_file.replace(rig_path, audio_path).replace(".txt", ".wav")
                save_file = rig_file.replace(rig_path, save_path)
                print(audio_file, rig_file, save_file)
                model.validate(audio_file, rig_file, save_file)


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


def test_semantic_emotion():
    parser = argparse.ArgumentParser(description='NETTalk Stage 3: Audio-driven Emotion')
    parser.add_argument("--input_dim", type=int, default=174)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--emotion_dim", type=int, default=7)
    parser.add_argument("--max_duration", type=int, default=5000)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--weight_path", type=str, default="weights/audio_semantic.pth")
    args = parser.parse_args()

    audio_path = "C:/Users/86134/Desktop/data/all_emotions/validation/WAV"
    save_path = "result/audio_semantic"

    # build model
    model = AudioSemanticNet(args).to(device)
    model.load_weight(args.weight_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for root, dirs, files in os.walk(audio_path):
        save_root = root.replace(audio_path, save_path)
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        for file in files:
            if file.endswith(".wav"):
                audio_file = os.path.join(root, file)
                save_file = audio_file.replace(audio_path, save_path).replace(".wav", ".txt")
                print(audio_file, save_file)
                model.validate(audio_file, save_file)


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


def test_CLIP():
    parser = argparse.ArgumentParser(description='NETTalk Stage 4: Image/Text Guide')
    parser.add_argument("--input_dim", type=int, default=174)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--emotion_dim", type=int, default=7)
    parser.add_argument("--max_duration", type=int, default=5000)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--weight_path", type=str, default="weights/clip_emotion.pth")
    args = parser.parse_args()

    audio_path = "C:/Users/86134/Desktop/data/all_emotions/validation/WAV"
    image_save_path = "result/clip_image"
    text_save_path = "result/clip_text"

    image_path = "C:/Users/86134/Desktop/data/RAVDESS/image/01-01-03-02-01-01-24.png"
    text_path = "C:/Users/86134/Desktop/data/RAVDESS/text/01-01-03-02-01-01-24.txt"

    # build model
    model = CLIPNet(args).to(device)
    model.load_weight(args.weight_path)

    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    if not os.path.exists(text_save_path):
        os.mkdir(text_save_path)
    for root, dirs, files in os.walk(audio_path):
        image_save_root = root.replace(audio_path, image_save_path)
        text_save_root = root.replace(audio_path, text_save_path)
        if not os.path.exists(image_save_root):
            os.mkdir(image_save_root)
        if not os.path.exists(text_save_root):
            os.mkdir(text_save_root)
        for file in files:
            if file.endswith(".wav"):
                audio_file = os.path.join(root, file)
                image_save_file = audio_file.replace(audio_path, image_save_path).replace(".wav", ".txt")
                text_save_file = audio_file.replace(audio_path, text_save_path).replace(".wav", ".txt")
                print(audio_file, image_save_file, text_save_file)
                model.validate(audio_file, image_save_file, image_path=image_path, text_path=None)
                model.validate(audio_file, text_save_file, image_path=None, text_path=text_path)


if __name__ == "__main__":
    # train_cross_reconstruction()
    # test_cross_reconstruction()
    # train_audio_content()
    # test_audio_content()
    # train_semantic_emotion()
    # test_semantic_emotion()
    # train_CLIP()
    test_CLIP()
