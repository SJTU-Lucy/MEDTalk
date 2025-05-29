import argparse
import os
import torch
from models.clip_emotion import CLIPNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicting with audio and given text guidance')
    parser.add_argument("--input_dim", type=int, default=174)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--emotion_dim", type=int, default=7)
    parser.add_argument("--max_duration", type=int, default=5000)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--text", type=str, default="assets/text/01-01-03-02-01-01-24.txt")
    parser.add_argument("--audio", type=str, default="assets/audio/angry_gd01_001_006.wav")
    parser.add_argument("--weight_path", type=str, default="weights/clip_emotion.pth")
    args = parser.parse_args()

    save_path = "result/demo_text"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # build model
    model = CLIPNet(args).to(device)
    model.load_weight(args.weight_path)

    audio_file = args.audio
    guide_file = args.text
    if not os.path.exists(audio_file):
        print("No such audio file from ", audio_file)
    elif not os.path.exists(guide_file):
        print("No such text file from ", guide_file)
    else:
        save_file = os.path.join(save_path, os.path.basename(audio_file).replace(".wav", ".txt"))
        print(audio_file, guide_file, save_file)
        model.validate(audio_file, save_file, image_path=None, text_path=guide_file)

