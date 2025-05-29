import numpy as np
import os
import cv2

bs_path = "C:/Users/86134/Desktop/data/RAVDESS/blendshape"
video_path = "C:/Users/86134/Desktop/data/RAVDESS/video"
save_path = "C:/Users/86134/Desktop/data/RAVDESS//image"
if not os.path.exists(save_path):
    os.mkdir(save_path)

emo_indices = [
    0, 1, 2, 3, 4,
    5, 6, 7,
    18, 19, 20, 21,
    30, 31, 42, 43,
    48, 49
]

for file in os.listdir(bs_path):
    print(file[:-4])
    bs_file = os.path.join(bs_path, file[:-4]+".npy")
    video_file = os.path.join(video_path, file[:-4]+".mp4")
    save_file = os.path.join(save_path, file[:-4]+".png")

    bs_data = np.load(bs_file)
    bs_sums = np.sum(bs_data[:, emo_indices], axis=1)  # 按行求和

    max_frame_idx = np.argmax(bs_sums)
    print(f"BlendShape 之和最大值出现在帧: {max_frame_idx}")

    cap = cv2.VideoCapture(video_file)
    fps = 30  # 视频帧率
    target_frame = max_frame_idx  # 直接使用索引
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_file, frame)
        print(f"成功保存最大 BlendShape 之和的帧: {save_file}")
    else:
        print("读取目标帧失败")

    cap.release()

