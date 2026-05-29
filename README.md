# Voiceprint Recognition System

基于 ECAPA-TDNN 声纹识别模型的配电柜操作人员身份认证系统。

## 功能特性

- **用户管理**：添加、删除用户，支持部门信息管理
- **声纹注册**：上传音频或浏览器录音，引导式多段录音自动融合为单一声纹向量
- **身份识别**：上传音频，1:N 识别说话人身份
- **身份验证**：验证音频是否是指定用户（1:1）
- **验证日志**：查看历史验证记录
- **系统重置**：管理员验证后一键清空所有数据

## 技术架构

- **声纹模型**：SpeechBrain ECAPA-TDNN（192维嵌入向量）
- **后端**：Flask REST API
- **前端**：HTML + CSS + JavaScript（单页应用）
- **数据库**：SQLite
- **音频处理**：soundfile + scipy（带通滤波去噪）+ ffmpeg（格式转换）
- **浏览器录音**：Web Audio API (MediaRecorder)

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/xiaomu1110/voiceprint-system.git
cd voiceprint-system

# 安装系统依赖（Ubuntu/Debian）
sudo apt-get install -y libsndfile1 ffmpeg

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt

# 启动服务
python3 web_server.py

# 访问 http://localhost:5000
```

## 系统要求

- Python 3.10+
- FFmpeg（用于音频格式转换）
- 系统依赖：libsndfile1, libportaudio2

### 安装系统依赖

```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1 ffmpeg libportaudio2

# macOS
brew install libsndfile portaudio ffmpeg
```

## 项目结构

```
voiceprint-system/
├── web_server.py           # Flask Web 服务器（API + 静态页面）
├── config.py               # 配置文件
├── database.py             # SQLite 数据库管理
├── voiceprint.py           # 声纹处理核心（模型加载、特征提取、比对）
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 配置
├── docker-compose.yml      # Docker Compose 配置
├── templates/
│   └── index.html          # 前端主页面
├── static/
│   ├── css/style.css       # 样式
│   └── js/app.js           # 前端交互逻辑
└── data/
    ├── audio_files/        # 音频文件存储
    └── models/             # 模型缓存
```

## 配置说明

在 `config.py` 中可修改：

```python
SIMILARITY_THRESHOLD = 0.75  # 相似度阈值（0-1），越低越宽松
SAMPLE_RATE = 16000          # 音频采样率
EMBEDDING_DIM = 192          # 声纹向量维度
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 系统状态（模型加载进度） |
| GET | `/api/users` | 获取所有用户 |
| POST | `/api/users` | 添加用户 |
| DELETE | `/api/users/<id>` | 删除用户及其声纹 |
| POST | `/api/voiceprint/register` | 单条声纹注册 |
| POST | `/api/voiceprint/register-batch` | 批量注册（多录音融合） |
| POST | `/api/voiceprint/identify` | 身份识别（1:N） |
| POST | `/api/voiceprint/verify` | 身份验证（1:1） |
| GET | `/api/logs` | 获取验证日志 |
| DELETE | `/api/logs` | 清空日志 |
| POST | `/api/reset` | 重置系统（需管理员认证） |

## 音频要求

- **推荐格式**：WAV、FLAC、OGG
- **其他格式**：MP3、M4A、WebM（自动通过 FFmpeg 转换）
- **建议时长**：3-10 秒
- **采样率**：16kHz（自动转换）

## 管理员重置

点击右上角"重置系统"按钮，输入管理员账号密码：
- 账号：`admin`
- 密码：`123456`

## 许可证

MIT License
