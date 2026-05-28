# Voiceprint Recognition System

基于ECAPA-TDNN声纹识别模型的配电柜操作人员身份认证系统。

## 功能特性

- **用户管理**：添加、删除用户，支持部门信息管理
- **声纹注册**：上传音频文件，自动提取声纹特征存储
- **身份识别**：上传音频文件，识别说话人身份
- **身份验证**：验证音频是否是指定用户
- **验证日志**：查看历史验证记录

## 技术架构

- **声纹识别模型**：SpeechBrain ECAPA-TDNN（EER 0.170%）
- **前端界面**：tkinter
- **后端**：Python 3.11
- **数据库**：SQLite
- **音频处理**：soundfile + scipy
- **容器化**：Docker

## 快速开始

### 方式一：Docker部署（推荐）

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/voiceprint-system.git
cd voiceprint-system

# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 方式二：本地运行

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行系统
python app.py
```

## 项目结构

```
voiceprint-system/
├── app.py                  # 主程序入口
├── config.py               # 配置文件
├── database.py             # 数据库管理
├── voiceprint.py           # 声纹处理核心
├── test_system.py          # 系统测试脚本
├── requirements.txt        # 依赖列表
├── Dockerfile              # Docker配置
├── docker-compose.yml      # Docker Compose配置
├── ui/                     # 前端界面
│   ├── __init__.py
│   ├── main_window.py      # 主窗口
│   ├── register_frame.py   # 用户管理界面
│   ├── verify_frame.py     # 身份验证界面
│   └── log_frame.py        # 日志查看界面
└── data/                   # 数据存储
    ├── audio_files/        # 音频文件
    ├── models/             # 模型文件
    └── voiceprints.db      # 数据库
```

## 配置说明

在 `config.py` 中可以修改以下配置：

```python
SIMILARITY_THRESHOLD = 0.75  # 相似度阈值（0-1）
SAMPLE_RATE = 16000          # 采样率
EMBEDDING_DIM = 192          # 嵌入向量维度
```

## 音频要求

- **推荐格式**：WAV、FLAC、OGG
- **其他格式**：MP3、M4A（需要FFmpeg）
- **建议时长**：3-10秒
- **采样率**：16kHz（自动转换）

## 开发计划

- [ ] REST API接口
- [ ] Web管理界面
- [ ] 实时音频录制
- [ ] 批量注册功能
- [ ] 用户权限管理

## 许可证

Private - 仅限内部使用

## 技术支持

如有问题，请联系系统管理员。
