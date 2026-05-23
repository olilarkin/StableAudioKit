# stableaudio3-ios

[English](README.md) | [简体中文](README.zh-CN.md)

在 iPhone 上用 Stable Audio 3 生成音乐和音效。

这是一个 iOS app 和 runtime，用 MLX Swift 在 iPhone 本地跑 Stable Audio 3。你输入 prompt，点播放，app 就会在手机上生成一段短的双声道 WAV 并播放。不需要云端服务器。

当前 app 跑通的是 `small-music` 和 `small-sfx`。下一个适合移动端的目标是 `medium`。最大的模型不是这个项目的目标。

## 演示

https://github.com/user-attachments/assets/5435773b-fb4b-492e-ac8e-33a8d211979b

## 这个项目解决什么

Stable Audio 3 官方已经有 Mac 上的 MLX 路径。这个项目把这条路径搬到 iPhone 上：

- 把官方 MLX 权重转成 iOS app 方便加载的文件
- 在 iOS app 里加载模型
- 在手机本地生成音频
- 模型第一次加载后保持常驻，后续生成更快
- 输出真实设备上的生成耗时日志

## 可以拿来试什么

- 鼓点 groove
- 短音乐 loop
- 音效
- 氛围 texture
- one-shot 音频创意

建议先用 `1s` 和 `4 steps` 测延迟。更在意质量时，再加时长或采样步数。

## 快速开始

克隆仓库：

```bash
git clone https://github.com/kellyvv/StableAudio3-IOS.git
cd StableAudio3-IOS
```

安装工具：

```bash
brew install xcodegen
python3 -m venv .venv
source .venv/bin/activate
pip install -U mlx numpy huggingface_hub
```

登录 Hugging Face：

```bash
hf auth login
```

你需要先在 Hugging Face 上获得 Stable Audio 3 权重访问权限，并接受上游 Stability AI 和 Gemma 条款。

下载官方 MLX 权重：

```bash
hf download stabilityai/stable-audio-3-optimized \
  --include "MLX/t5gemma_f16.npz" \
  --include "MLX/dit_sm-music_f16.npz" \
  --include "MLX/dit_sm-sfx_f16.npz" \
  --include "MLX/same_s_decoder_f32.npz" \
  --local-dir Models/stable-audio-3-optimized
```

转换成 iOS app 使用的格式：

```bash
python3 Scripts/prepare_weights.py
```

打开 app 工程：

```bash
xcodegen generate
open StableAudio3iOS.xcodeproj
```

在 Xcode 里选择你的开发团队，选择一台真实 iPhone，然后运行。app 打开后选择 Music 或 SFX，再点播放。

## 本地会生成哪些文件

转换脚本会在 `Resources/Weights/` 下生成这些文件：

```text
t5gemma_f16.safetensors
dit_sm-music_f16.safetensors
dit_sm-sfx_f16.safetensors
same_s_decoder_f32.safetensors
t5gemma_tokenizer.model
sa3_conditioner_sm-music.safetensors
sa3_conditioner_sm-sfx.safetensors
manifest.json
```

这些文件会被 git 忽略。这个仓库不直接提供模型权重。

## 耗时

第一次生成会加载模型，所以更慢。再生成一次，才是 warm 状态下的速度。Xcode 里会看到类似日志：

```text
[SA3] cache hit DiT
[SA3] step 1/4 320ms total=...
[SA3] total 1800ms model=Small SFX prompt="..." seconds=1.0 steps=4 latentLength=...
```

## 构建检查

不签名也可以先检查工程是否能构建：

```bash
xcodebuild -quiet \
  -project StableAudio3iOS.xcodeproj \
  -scheme StableAudio3iOS \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

## 注意

- 请用真实 iPhone，模拟器不适合测这个。
- 当前跑通的是 `small-music` 和 `small-sfx`。
- `medium` 是下一步目标。
- 加入本地权重后 app 会很大，两个 small 模型资源约 2.5 GB。

## License

这个仓库里的代码使用 MIT License。

模型权重不包含在本仓库里。Stable Audio 3 权重受 Stability AI Community License 约束；T5Gemma 受 Gemma Terms of Use 约束。下载、转换或分发权重前，请先阅读 `NOTICE` 和 `THIRD_PARTY_LICENSES.md`。
