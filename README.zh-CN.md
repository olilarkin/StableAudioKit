# stableaudio3-ios

在 iPhone 本地运行 Stable Audio 3，用 MLX Swift 生成音乐和音效。

[English](README.md) | [简体中文](README.zh-CN.md)

<table align="center">
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/5435773b-fb4b-492e-ac8e-33a8d211979b" controls width="360"></video>
    </td>
  </tr>
</table>

## 这是做什么的

这是一个 iOS app 和 runtime。你输入 prompt，选择 `Music` 或 `SFX`，点播放，
iPhone 会在本地生成一段双声道 WAV 并直接播放。

- 不需要服务器
- 不需要流式返回
- 可以生成音乐 loop、架子鼓短音、音效
- 当前支持 `small-music` 和 `small-sfx`
- 下一个适合移动端的目标是 `medium`

app 共享 T5Gemma 文本编码器和 SAME-S decoder。切换 `Music` / `SFX` 时，
主要是在切换不同的 DiT 模型。

## 快速开始

克隆仓库：

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

登录 Hugging Face：

```bash
hf auth login
```

你需要先在 Hugging Face 上获得 Stable Audio 3 权重访问权限，并接受上游
Stability AI 和 Gemma 条款。

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

生成 Xcode 工程，并在真实 iPhone 上运行：

```bash
xcodegen generate
open StableAudio3iOS.xcodeproj
```

在 Xcode 里选择你的开发团队，然后运行到手机。

## App 模式

| 模式 | 模型 | 适合 |
| --- | --- | --- |
| Music | `dit_sm-music_f16` | 音乐 loop、groove、带音高的创意 |
| SFX | `dit_sm-sfx_f16` | 音效、架子鼓短音、拟音 |

质量选项使用同一个 sampler：

| 选项 | 步数 | 用途 |
| --- | ---: | --- |
| Fast | 4 | 快速测试 |
| Better | 8 | 默认质量 |
| Best | 16 | 更慢，但更干净 |

架子鼓短音预设使用 2 steps，方便测试很低延迟的 one-shot。

## 本地权重文件

`Scripts/prepare_weights.py` 会在 `Resources/Weights/` 生成这些文件：

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

## 耗时日志

第一次生成会加载权重，所以更慢。再生成一次，才是 warm 状态下的速度。
Xcode 里会看到类似日志：

```text
[SA3] cache hit DiT Small SFX
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

- 请用真实 iPhone，模拟器不适合跑这个 runtime。
- 同时放入两个 small 模型后 app 会很大，本地模型文件大约 2.5 GB。
- 最大的 Stable Audio 3 模型不是这个项目的目标。

## License

这个仓库里的代码使用 MIT License。

模型权重不包含在本仓库里。Stable Audio 3 权重受 Stability AI Community
License 约束；T5Gemma 受 Gemma Terms of Use 约束。下载、转换、分发或使用
权重前，请先阅读 `NOTICE` 和 `THIRD_PARTY_LICENSES.md`。
