"""
配置文件 - 声纹识别系统
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio_files")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# 数据库配置
DB_PATH = os.path.join(DATA_DIR, "voiceprints.db")

# 声纹识别配置
SIMILARITY_THRESHOLD = 0.75  # 相似度阈值
SAMPLE_RATE = 16000  # 采样率
EMBEDDING_DIM = 192  # 嵌入向量维度

# 模型配置
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"

# 确保目录存在
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
