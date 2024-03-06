



# Audiocraft ----Learning

原始仓库： https://github.com/facebookresearch/audiocraft

系统 linux x86 + anaconda

__Notes：__  windows 必要库triton不支持 windows安装平台需要改写setup.py文件， mac 平台m1芯片不支持cuda，这个库也没有支持m1芯片的

## Install

Notes: pytorch 2.1.0 cuda 12.1

```python
conda crate -n Cs330Project # 创建环境
conda activate Cs330Project # 激活
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia  # 安装 pytorch

conda install pip #  下载pip 安装工具
python -m pip install -U audiocraft  # stable release
```



## Test

__Notes:__ MusicGen.get_pretrained(‘facebook/musicgen-melody')  当第一次load模型的时候，需要下载，但是，如果没有__科学上网__，导致下载不了。 

### Test Code

只有一点 就是descriptions 和官网(https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)上的不一样。

```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['lowlowlowlow low low ', 'energetic EDM', 'sad jazz', 'Blue Blue Blue', 'a bird', 'a b c d e f g ']
wav = model.generate(descriptions)  # generates 3 samples.

#melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
#wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}_1', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

### Result



<audio src="0_1.wav"></audio>

lowlowlowlow low low 

<audio src="1_1.wav"></audio>

energetic EDM

<audio src="2_1.wav"></audio>

sad jazz

<audio src="3_1.wav"></audio>

Blue Blue Blue

<audio src="4_1.wav"></audio>

a bird

<audio src="5_1.wav"></audio>

a b c d e f g 



## Training

Contining