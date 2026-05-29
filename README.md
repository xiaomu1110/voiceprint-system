# Voiceprint Recognition System

基于 ECAPA-TDNN 声纹识别模型的配电柜操作人员身份认证系统。

## 功能

- 用户管理（添加/删除）
- 声纹注册（上传音频 / 浏览器录音 / 引导式多段录音融合）
- 身份识别（1:N）和身份验证（1:1）
- 验证日志
- 系统重置（管理员验证）

## 快速开始

```bash
# 1. 安装系统依赖
sudo apt-get install -y libsndfile1 ffmpeg python3-venv

# 2. 克隆项目
git clone https://github.com/xiaomu1110/voiceprint-system.git
cd voiceprint-system

# 3. 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. 启动
python3 web_server.py
```

访问 http://localhost:5000

## 配置

编辑 `config.py`：

```python
SIMILARITY_THRESHOLD = 0.75  # 相似度阈值（0-1）
SAMPLE_RATE = 16000          # 采样率
```

## 管理员重置

点击右上角"重置系统"，输入账号 `admin`，密码 `123456`。

## 项目结构

```
├── web_server.py       # Flask Web 服务器
├── config.py           # 配置
├── database.py         # 数据库管理
├── voiceprint.py       # 声纹处理核心
├── requirements.txt    # Python 依赖
├── templates/
│   └── index.html      # 前端页面
├── static/
│   ├── css/style.css   # 样式
│   └── js/app.js       # 前端逻辑
└── data/               # 数据存储
```

## 技术栈

- 声纹模型：SpeechBrain ECAPA-TDNN
- 后端：Flask
- 前端：HTML + CSS + JavaScript
- 数据库：SQLite
- 音频处理：soundfile + scipy + ffmpeg

## 许可证

MIT License
