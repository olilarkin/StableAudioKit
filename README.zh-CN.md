# stableaudio3-ios

[English](README.md) | [简体中文](README.zh-CN.md)

在 iPhone 上用 MLX Swift 本地运行 Stable Audio 3 Small Music。

这个项目是一个 iOS app 和运行时，用来在 Apple 设备上做文本生成音频。它会把官方 Stable Audio 3 optimized MLX checkpoint 转成 app 可以直接加载的 safetensors，用 MLX Swift 在设备端推理，生成音频后写成 WAV 并在 app 里播放。

## 能做什么

- 直接在 iPhone 上根据文字 prompt 生成双声道音频
- 跑 Stable Audio 3 `small-music` 路径，使用 `same-s` decoder
- 测试短创意素材，比如鼓点 groove、氛围 texture、riff、音乐 cue
- 选择短时长预设做延迟测试：1s、2s、5s、10s、15s
- 选择 4 或 8 个采样步
- 第一次生成后保持 pipeline 常驻，后续生成复用已加载权重
- 在 Xcode console 输出每个阶段耗时，包括 T5、DiT sampling、decoder、WAV 写入

当前范围：

- 只支持 text-to-audio
- 只支持 Stable Audio 3 Small Music
- 目标是 iOS 17+ 真机
- 本地端侧推理，不需要服务器

## 快速开始

```bash
git clone https://github.com/kellyvv/StableAudio3-IOS.git
cd StableAudio3-IOS
```

安装本地工具：

```bash
brew install xcodegen
python3 -m venv .venv
source .venv/bin/activate
pip install -U mlx numpy huggingface_hub
```

使用已经接受 Stable Audio 3 和 Gemma 上游条款的 Hugging Face 账号登录：

```bash
hf auth login
```

下载官方 optimized MLX checkpoint：

```bash
hf download stabilityai/stable-audio-3-optimized \
  --include "MLX/t5gemma_f16.npz" \
  --include "MLX/dit_sm-music_f16.npz" \
  --include "MLX/same_s_decoder_f32.npz" \
  --local-dir Models/stable-audio-3-optimized
```

为 iOS app 转换权重：

```bash
python3 Scripts/prepare_weights.py
```

生成并打开 Xcode 工程：

```bash
xcodegen generate
open StableAudio3iOS.xcodeproj
```

在 Xcode 里选择你的开发团队，选择一台真实 iPhone，然后运行 `StableAudio3iOS` scheme。app 会显示权重是否就绪，并提供 `Generate & Play` 按钮。

## 命令行构建检查

不签名也可以先验证工程能否构建：

```bash
xcodebuild -quiet \
  -project StableAudio3iOS.xcodeproj \
  -scheme StableAudio3iOS \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

## 权重文件

这个仓库不包含模型权重。转换完成后，app 会使用这些文件：

```text
Resources/Weights/t5gemma_f16.safetensors
Resources/Weights/dit_sm-music_f16.safetensors
Resources/Weights/same_s_decoder_f32.safetensors
Resources/Weights/t5gemma_tokenizer.model
Resources/Weights/sa3_conditioner.safetensors
Resources/Weights/manifest.json
```

这些生成文件已被 git 忽略。

## 耗时日志

从 Xcode 运行 app，点击 `Generate & Play`。第一次生成会包含模型加载：

```text
[SA3] cache miss DiT, loading weights
[SA3] step 1/4 320ms total=...
[SA3] total 1800ms prompt="..." seconds=1.0 steps=4 latentLength=...
```

再生成一次可以测 warm latency。warm run 应该能看到 cache hit：

```text
[SA3] cache hit tokenizer
[SA3] cache hit T5Gemma
[SA3] cache hit DiT
[SA3] cache hit decoder
```

如果想做后端式实验，先用 1s / 4 steps 测延迟，再根据质量和耗时决定是否增加时长或采样步。

## 项目结构

```text
StableAudio3iOS/          SwiftUI app 和 MLX Swift runtime
Scripts/prepare_weights.py  NPZ -> safetensors 转换脚本
Resources/Weights/        本地生成权重，git 忽略
project.yml               XcodeGen 工程配置
```

## 注意事项

- 请用真实 iPhone 测试。Simulator 不代表目标 MLX/Metal 路径。
- 第一次生成是 cold run。延迟数据看第二次或第三次更准确。
- 权重放进 app 后包体会很大，模型资源约 1.6 GB，不含 app 额外开销。
- 这是一个实用原型 runtime，不是完整的 Stable Audio 3 产品界面。

## License

仓库代码使用 MIT License。

模型权重不包含在本仓库中，也不受本仓库 license 覆盖。Stable Audio 3 权重受 Stability AI Community License 约束；T5Gemma 组件受 Gemma Terms of Use 约束。

下载、转换或分发任何模型权重前，请先阅读 `NOTICE` 和 `THIRD_PARTY_LICENSES.md`。
