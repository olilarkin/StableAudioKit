# stableaudio3-ios

[English](README.md) | [简体中文](README.zh-CN.md)

在 iOS 上使用 MLX Swift 运行 Stable Audio 3 Small Music。

这个仓库包含 iOS demo/runtime、权重转换脚本，以及 small-music 文本生成音频路径的 Swift MLX 移植：

```text
prompt -> T5Gemma -> conditioning -> DiT sampler -> SAME-S decoder -> WAV playback
```

默认延迟测试配置是 1 秒音频、4 个采样步。第一次生成会把权重加载进一个常驻的 pipeline actor；后续生成会复用 tokenizer、T5Gemma、conditioner、DiT 和 decoder 权重，并在 Xcode console 输出每个阶段的耗时。

## 包含内容

- SwiftUI iOS app target
- T5Gemma、SA3 conditioning、DiT small-music、SAME-S decoder、sampler、WAV writer 和播放逻辑的 MLX Swift 实现
- 将官方 MLX NPZ checkpoint 转成 app 内 safetensors 资源的 Python 脚本
- XcodeGen 工程配置

## 不包含内容

这个仓库不包含模型权重。

以下生成文件已被 git 忽略：

```text
Resources/Weights/t5gemma_f16.safetensors
Resources/Weights/dit_sm-music_f16.safetensors
Resources/Weights/same_s_decoder_f32.safetensors
Resources/Weights/t5gemma_tokenizer.model
Resources/Weights/sa3_conditioner.safetensors
Resources/Weights/manifest.json
```

请在接受上游 Stability AI 和 Gemma 条款后，再下载和使用模型权重。

## 环境要求

- macOS 和 Xcode 16+
- iOS 17+ 真机，且带 Apple GPU
- XcodeGen
- 安装了 `mlx` 和 `numpy` 的 Python 环境
- 有权限访问 Hugging Face 上的 `stabilityai/stable-audio-3-optimized`

## 准备权重

先安装 Hugging Face CLI，并使用已接受上游模型条款的账号登录。

把官方 optimized MLX checkpoint 下载到本地被忽略的 `Models/` 目录：

```bash
hf download stabilityai/stable-audio-3-optimized \
  --include "MLX/t5gemma_f16.npz" \
  --include "MLX/dit_sm-music_f16.npz" \
  --include "MLX/same_s_decoder_f32.npz" \
  --local-dir Models/stable-audio-3-optimized
```

为 iOS app 转换 checkpoint：

```bash
python3 Scripts/prepare_weights.py
```

脚本会把生成资源写到 `Resources/Weights/`。

## 构建

生成 Xcode 工程：

```bash
xcodegen generate
```

命令行构建：

```bash
xcodebuild -quiet \
  -project StableAudio3iOS.xcodeproj \
  -scheme StableAudio3iOS \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

如果要安装到真机，打开 `StableAudio3iOS.xcodeproj`，选择你的开发团队，然后把 `StableAudio3iOS` scheme 运行到 iPhone。

## 耗时日志

从 Xcode 启动 app，点击 `Generate & Play`。console 会输出 cold run 和 warm run 的耗时：

```text
[SA3] cache miss DiT, loading weights
[SA3] step 1/4 320ms total=...
[SA3] total 1800ms prompt="..." seconds=1.0 steps=4 latentLength=...
```

连续生成第二次、第三次可以测 warm pipeline 延迟。warm run 应该能看到 tokenizer、T5Gemma、conditioner、DiT 和 decoder 的 cache hit。

## License

仓库代码使用 MIT License。模型权重不包含在本仓库中，也不受本仓库 MIT License 覆盖。

下载、转换或分发任何模型权重前，请先阅读 `NOTICE` 和 `THIRD_PARTY_LICENSES.md`。
