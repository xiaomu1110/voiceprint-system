# 配电柜智能语音识别系统

基于 SpeechBrain ECAPA-TDNN 的声纹识别系统，用于配电柜操作人员身份认证。系统只支持 NVIDIA CUDA GPU 加速运行；如果没有可用 GPU、驱动异常或安装了 CPU 版 PyTorch，系统会直接失败，不会降级到 CPU。

## 主要功能

- 用户管理：添加、删除用户
- 声纹录入：支持上传音频、浏览器录音、引导式多段录入
- 引导注册：分段预处理、质量评分、处理后音频播放、持续追加融合
- 身份识别：1:N 识别当前说话人是谁
- 身份验证：1:1 验证是否为指定用户
- 验证日志：查看和清空认证记录
- 系统状态页：展示 GPU、CUDA、PyTorch、模型、显存、数据统计等运行信息
- 系统重置：管理员认证后清空用户、声纹和日志

## 环境要求

- Windows 或 Linux
- Python 3.11
- uv
- ffmpeg
- NVIDIA GPU
- NVIDIA 显卡驱动
- CUDA 版 PyTorch

安装脚本会自动安装 uv、ffmpeg、Python 3.11，并根据显卡驱动安装 CUDA PyTorch。没有可用 NVIDIA GPU 时，安装和运行都会失败。

## 快速开始

### Windows

```powershell
git clone https://github.com/xiaomu1110/voiceprint-system.git
cd voiceprint-system
.\scripts\setup.ps1
.\.venv\Scripts\python.exe web_server.py
```

### Linux

```bash
git clone https://github.com/xiaomu1110/voiceprint-system.git
cd voiceprint-system
bash scripts/setup.sh
.venv/bin/python web_server.py
```

启动后在浏览器打开：

```text
http://localhost:5000
```

## GPU 运行策略

本系统是 GPU-only 版本。

| 场景 | 行为 |
|---|---|
| 有 NVIDIA GPU 和驱动 | 安装 CUDA PyTorch，使用 GPU 运行 |
| 未检测到 NVIDIA GPU | 安装失败 |
| 安装了 CPU 版 PyTorch | 环境检查失败，运行失败 |
| CUDA 初始化失败 | 运行失败，不回退 CPU |

多 GPU 机器可以通过环境变量选择设备：

```powershell
$env:VOICEPRINT_CUDA_DEVICE_INDEX="0"
```

Linux:

```bash
export VOICEPRINT_CUDA_DEVICE_INDEX=0
```

## 常用命令

| 操作 | 命令 |
|---|---|
| 安装依赖 | `uv sync` |
| 安装 CUDA PyTorch | `uv run python scripts/bootstrap.py --mode gpu` |
| 检查环境 | `uv run --no-sync python scripts/doctor.py` |
| 启动服务 | `uv run --no-sync python web_server.py` |
| 添加依赖 | `uv add <package>` |

## 声纹录入流程

推荐使用“引导注册”。

当前录入逻辑不是简单地上传后直接保存，而是采用更接近成熟声纹系统的分段流程：

1. 选择用户
2. 按提示朗读文本，或上传已有录音文件
3. 后端将音频转为 16 kHz 单声道 WAV
4. 对音频做带通滤波和质量评估
5. 前端展示质量分、有效语音时长、噪声、削波等信息
6. 前端支持播放处理后的 WAV
7. 点击完成注册后，后端提取每段 embedding
8. 对新片段做质量门控和一致性检查
9. 将通过的片段加入该用户的历史注册片段
10. 把历史片段和本次新增片段重新融合为一个声纹档案

后续如果同一个用户继续录入或上传新音频，系统会继续累积融合，不会丢掉已有片段。

## 页面说明

系统采用侧边栏导航：

- 系统状态：查看模型、GPU、CUDA、PyTorch、显存和数据统计
- 用户管理：添加、删除用户
- 声纹注册：普通注册和引导式注册
- 身份验证：进行 1:N 识别或 1:1 验证
- 验证日志：查看认证历史

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VOICEPRINT_CUDA_DEVICE_INDEX` | `0` | CUDA GPU 设备编号 |
| `VOICEPRINT_DATA_DIR` | `data` | 数据目录 |
| `VOICEPRINT_DB_PATH` | `data/voiceprints.db` | SQLite 数据库路径 |
| `VOICEPRINT_THRESHOLD` | `0.75` | 声纹相似度阈值 |
| `VOICEPRINT_SAMPLE_RATE` | `16000` | 音频采样率 |
| `VOICEPRINT_HOST` | `0.0.0.0` | 服务监听地址 |
| `VOICEPRINT_PORT` | `5000` | 服务端口 |
| `VOICEPRINT_DEBUG` | `false` | Flask 调试模式 |

## 项目结构

```text
voiceprint-system/
├── web_server.py       # Flask Web 服务和 API
├── config.py           # 配置项和环境变量
├── database.py         # SQLite 数据库管理
├── voiceprint.py       # 声纹处理、GPU 检测、质量评估和融合逻辑
├── pyproject.toml      # 项目依赖配置
├── uv.lock             # uv 锁文件
├── .python-version     # Python 版本
├── scripts/
│   ├── setup.ps1       # Windows 安装脚本
│   ├── setup.sh        # Linux 安装脚本
│   ├── bootstrap.py    # GPU 检测和 CUDA PyTorch 安装
│   └── doctor.py       # 环境检查
├── templates/
│   └── index.html      # 前端页面
├── static/
│   ├── css/style.css   # 样式
│   └── js/app.js       # 前端逻辑
└── data/               # 运行时数据，不提交到仓库
    ├── audio_files/
    ├── models/
    └── voiceprints.db
```

## 技术栈

- 声纹模型：SpeechBrain ECAPA-TDNN
- 后端：Flask
- 前端：HTML、CSS、JavaScript
- 数据库：SQLite
- 音频处理：soundfile、scipy、ffmpeg
- 深度学习：PyTorch CUDA
- 环境管理：uv

## 许可证

MIT License
