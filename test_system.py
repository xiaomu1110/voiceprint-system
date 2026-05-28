"""
系统测试脚本 - 声纹识别系统
用于测试声纹注册和验证功能
"""
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager
from voiceprint import VoiceprintExtractor, VoiceprintDatabase
from config import SIMILARITY_THRESHOLD, AUDIO_DIR


def create_test_audio(filename, duration=3, sample_rate=16000):
    """
    创建测试音频文件（正弦波）
    
    Args:
        filename: 文件名
        duration: 时长（秒）
        sample_rate: 采样率
        
    Returns:
        filepath: 文件路径
    """
    import soundfile as sf
    
    # 生成正弦波
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 使用随机频率模拟不同人的声音
    freq = np.random.uniform(100, 400)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    
    # 添加一些随机噪声
    noise = np.random.normal(0, 0.01, len(audio))
    audio = audio + noise
    
    # 保存音频
    filepath = os.path.join(AUDIO_DIR, filename)
    sf.write(filepath, audio, sample_rate)
    
    return filepath


def test_system():
    """测试系统功能"""
    print("=" * 50)
    print("配电柜智能语音识别系统 - 功能测试")
    print("=" * 50)
    
    # 1. 初始化系统
    print("\n[1/6] 初始化系统...")
    db = DatabaseManager()
    extractor = VoiceprintExtractor()
    vp_db = VoiceprintDatabase(db, extractor)
    print("[OK] 系统初始化成功")
    
    # 2. 测试添加用户
    print("\n[2/6] 测试添加用户...")
    user1_id = db.add_user("张三", "配电运维部")
    user2_id = db.add_user("李四", "电力调度部")
    print(f"[OK] 添加用户成功: 张三(ID:{user1_id}), 李四(ID:{user2_id})")
    
    # 3. 创建测试音频
    print("\n[3/6] 创建测试音频...")
    audio1_path = create_test_audio("test_zhangsan.wav")
    audio2_path = create_test_audio("test_lisi.wav")
    print(f"[OK] 创建测试音频成功")
    
    # 4. 测试声纹注册
    print("\n[4/6] 测试声纹注册...")
    vp1_id = vp_db.register_voiceprint(user1_id, audio1_path)
    vp2_id = vp_db.register_voiceprint(user2_id, audio2_path)
    print(f"[OK] 声纹注册成功: 张三(VP:{vp1_id}), 李四(VP:{vp2_id})")
    
    # 5. 测试身份识别
    print("\n[5/6] 测试身份识别...")
    result = vp_db.verify_speaker(audio1_path, threshold=0.5)
    print(f"  识别结果: {'成功' if result['matched'] else '失败'}")
    if result['matched']:
        print(f"  匹配用户: {result['user_name']}")
        print(f"  相似度: {result['similarity']:.4f}")
    else:
        print(f"  最高相似度: {result['similarity']:.4f}")
    
    # 6. 测试身份验证
    print("\n[6/6] 测试身份验证...")
    result = vp_db.verify_speaker_by_id(audio1_path, user1_id, threshold=0.5)
    print(f"  验证结果: {'通过' if result['matched'] else '失败'}")
    print(f"  相似度: {result['similarity']:.4f}")
    
    # 清理测试数据
    print("\n" + "=" * 50)
    print("测试完成，清理测试数据...")
    
    # 删除测试用户
    db.delete_user(user1_id)
    db.delete_user(user2_id)
    
    # 删除测试音频
    for f in [audio1_path, audio2_path]:
        if os.path.exists(f):
            os.remove(f)
    
    print("[OK] 测试数据清理完成")
    print("=" * 50)
    print("\n系统测试通过！")


if __name__ == "__main__":
    test_system()
