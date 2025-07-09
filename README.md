# MEDTalk: Multimodal-Controlled 3D Facial Animation with Dynamic Emotions by Disentangled Embedding
This is the official repository for the paper "MEDTalk: Multimodal-Controlled 3D Facial Animation with Dynamic Emotions by Disentangled Embedding"

### 1.Environment

- Python 3.10
- Pytorch 2.3
- CUDA 12.1

Clone the repo:

```
git clone https://github.com/SJTU-Lucy/MEDTalk.git
cd MEDTalk
```

Create conda environment:

```
conda create -n medtalk python=3.10
conda activate medtalk
# conda
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# install other requirements
pip install -r requirements.txt
```

### 2. Demo

Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1uRIZlO_0F2w-nzAjetAqKAbAWAJP5P8-). Put the pretrained models under **weights** folder.

MEDTalk provides a multi-modal audio-driven facial expression generation method. Users can use either emotion labels, images of the reference expression or text descriptions to control the style of generated expressions.

```
# emotion label (angry/disgust/fear/happy/neutral/sad/surprised)
python demo_label.py --emotion EMOTION_LABEL --audio YOUR_AUDIO_PATH
# reference image
python demo_image.py --image YOUR_TEXT_PATH --audio YOUR_AUDIO_PATH
# text description
python demo_text.py --text YOUR_TEXT_PATH --audio YOUR_AUDIO_PATH
```

The generated animation will be saved in `result` folder.

### 3.Visualization

The predicted expressions are formatted as 174-dimensional metahuman controller parameter sequences and are saved in a .txt file. The results can be visualized using either Maya or UE5. The scripts for visualization have been uploaded to **visualization** folder.

**(1) MAYA**

- Import your MetaHuman character from Quixel Bridge or somewhere else.
- Set the camera in panel-orthographic to **front**.
- Open *render.py* in Maya.
- Modify the input and output settings from line 135 to 140.
- If the output videos looks weird, adjusting camera settings from line 117 to 120.

**(2) Unreal Engine 5**

Unlike Maya, the output controller rig sequences cannot be directly be imported into UE5. So we use Maya as the import intermediary, exporting **.fbx** file by Maya, and then import it into UE5. The workflow is introduced by this video: [How to Rig and Animate a Metahuman: Maya to Unreal Engine 5 Workflow - YouTube](https://www.youtube.com/watch?v=OYjq4aRgKWg). As for setting key frames, *set_frames.py* can be used to set frame-wise controller values in Maya. 

### 4.Citation

If you find our work helpful for your research, please cite our paper:

```
@misc{liu2025medtalkmultimodalcontrolled3d,
      title={MEDTalk: Multimodal Controlled 3D Facial Animation with Dynamic Emotions by Disentangled Embedding}, 
      author={Chang Liu and Ye Pan and Chenyang Ding and Susanto Rahardja and Xiaokang Yang},
      year={2025},
      eprint={2507.06071},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.06071}, 
}
```

