"""
声纹处理核心模块 - 声纹识别系统
使用SpeechBrain的ECAPA-TDNN预训练模型提取声纹特征
"""
import torch
import numpy as np
import os
import tempfile
import subprocess
import soundfile as sf
from config import MODEL_NAME, SAMPLE_RATE, SIMILARITY_THRESHOLD


class VoiceprintExtractor:
    """声纹特征提取器"""
    
    # 支持的音频格式
    SUPPORTED_FORMATS = {'.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aac', '.wma', '.opus'}
    
    # 需要转换的格式
    NEED_CONVERT = {'.mp3', '.m4a', '.aac', '.wma', '.opus'}
    
    def __init__(self, model_name=MODEL_NAME):
        """
        初始化声纹特征提取器
        
        Args:
            model_name: SpeechBrain模型名称
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ffmpeg_path = self._find_ffmpeg()
        self._load_model()
    
    def _find_ffmpeg(self):
        """查找ffmpeg路径"""
        # 常见的ffmpeg路径
        possible_paths = [
            'ffmpeg',
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            os.path.expanduser(r'~\ffmpeg\bin\ffmpeg.exe'),
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run(
                    [path, '-version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except:
                continue
        
        return None
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=os.path.join("data", "models", "ecapa_tdnn"),
                run_opts={"device": self.device}
            )
            print(f"模型加载成功，使用设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def _convert_audio_ffmpeg(self, audio_path):
        """
        使用ffmpeg转换音频格式
        
        Args:
            audio_path: 原始音频文件路径
            
        Returns:
            wav_path: 转换后的wav文件路径
        """
        if not self.ffmpeg_path:
            raise ValueError("未找到ffmpeg，请安装ffmpeg或使用WAV/FLAC/OGG格式")
        
        # 生成临时文件路径
        temp_dir = tempfile.gettempdir()
        temp_name = f"vp_temp_{os.getpid()}.wav"
        temp_path = os.path.join(temp_dir, temp_name)
        
        try:
            # 使用ffmpeg转换
            cmd = [
                self.ffmpeg_path,
                '-i', audio_path,
                '-ar', str(SAMPLE_RATE),  # 采样率
                '-ac', '1',                # 单声道
                '-y',                      # 覆盖输出文件
                temp_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise ValueError(f"ffmpeg转换失败: {result.stderr[:200]}")
            
            return temp_path
            
        except subprocess.TimeoutExpired:
            raise ValueError("音频转换超时")
    
    def _convert_audio_soundfile(self, audio_path):
        """
        使用soundfile转换音频格式（适用于MP3等）
        
        Args:
            audio_path: 原始音频文件路径
            
        Returns:
            wav_path: 转换后的wav文件路径
        """
        # 生成临时文件路径
        temp_dir = tempfile.gettempdir()
        temp_name = f"vp_temp_{os.getpid()}.wav"
        temp_path = os.path.join(temp_dir, temp_name)
        
        try:
            # 尝试使用soundfile直接读取
            audio, sr = sf.read(audio_path, dtype='float32')
            
            # 转换为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # 保存为wav
            sf.write(temp_path, audio, sr)
            
            return temp_path
            
        except Exception as e:
            raise ValueError(f"音频转换失败: {e}")
    
    def _convert_audio(self, audio_path):
        """
        转换音频格式
        
        Args:
            audio_path: 原始音频文件路径
            
        Returns:
            wav_path: 转换后的wav文件路径
            is_temp: 是否是临时文件
        """
        ext = os.path.splitext(audio_path)[1].lower()
        
        # 如果已经是支持的格式，直接返回
        if ext in {'.wav', '.flac', '.ogg'}:
            return audio_path, False
        
        # 尝试使用ffmpeg转换
        if self.ffmpeg_path:
            try:
                temp_path = self._convert_audio_ffmpeg(audio_path)
                return temp_path, True
            except Exception as e:
                print(f"ffmpeg转换失败，尝试其他方式: {e}")
        
        # 尝试使用soundfile转换
        try:
            temp_path = self._convert_audio_soundfile(audio_path)
            return temp_path, True
        except Exception as e:
            raise ValueError(f"音频格式转换失败，请安装ffmpeg或使用WAV/FLAC/OGG格式: {e}")
    
    def load_audio(self, audio_path, target_sr=SAMPLE_RATE):
        """
        加载音频文件（支持多种格式）
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率
            
        Returns:
            waveform: 音频波形张量
            sample_rate: 采样率
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 检查文件格式
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的音频格式: {ext}，支持的格式: {', '.join(self.SUPPORTED_FORMATS)}")
        
        # 转换格式（如果需要）
        temp_path = None
        try:
            wav_path, is_temp = self._convert_audio(audio_path)
            if is_temp:
                temp_path = wav_path
            
            # 使用soundfile加载音频
            audio, sample_rate = sf.read(wav_path, dtype='float32')
            
            # 转换为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # 重采样（如果需要）
            if sample_rate != target_sr:
                from scipy import signal
                num_samples = int(len(audio) * target_sr / sample_rate)
                audio = signal.resample(audio, num_samples)
                sample_rate = target_sr
            
            # 转换为torch张量
            waveform = torch.FloatTensor(audio).unsqueeze(0)
            
            return waveform, sample_rate
            
        finally:
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def extract_embedding(self, audio_path):
        """
        提取声纹嵌入向量
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            embedding: 192维声纹嵌入向量
        """
        try:
            # 加载音频
            waveform, sample_rate = self.load_audio(audio_path)
            
            # 提取嵌入向量
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform)
            
            # 转换为numpy数组
            embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"提取声纹特征失败: {e}")
            raise
    
    def extract_embedding_from_tensor(self, waveform):
        """
        从音频张量提取声纹嵌入向量
        
        Args:
            waveform: 音频波形张量
            
        Returns:
            embedding: 192维声纹嵌入向量
        """
        try:
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform)
            
            embedding = embedding.squeeze().cpu().numpy()
            return embedding
            
        except Exception as e:
            print(f"提取声纹特征失败: {e}")
            raise
    
    def compute_similarity(self, embedding1, embedding2):
        """
        计算两个声纹嵌入向量的余弦相似度
        
        Args:
            embedding1: 声纹嵌入向量1
            embedding2: 声纹嵌入向量2
            
        Returns:
            similarity: 余弦相似度 (0-1)
        """
        # 归一化
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def verify_speaker(self, audio_path, reference_embeddings, threshold=SIMILARITY_THRESHOLD):
        """
        验证说话人身份
        """
        test_embedding = self.extract_embedding(audio_path)
        
        scores = []
        for user_id, ref_embedding in reference_embeddings:
            similarity = self.compute_similarity(test_embedding, ref_embedding)
            scores.append((user_id, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if scores:
            best_user_id, best_similarity = scores[0]
            
            if best_similarity >= threshold:
                return {
                    "matched": True,
                    "user_id": best_user_id,
                    "similarity": best_similarity,
                    "all_scores": scores
                }
        
        return {
            "matched": False,
            "user_id": None,
            "similarity": scores[0][1] if scores else 0.0,
            "all_scores": scores
        }
    
    def verify_speaker_by_id(self, audio_path, user_id, user_embeddings, threshold=SIMILARITY_THRESHOLD):
        """
        验证是否是指定用户
        """
        test_embedding = self.extract_embedding(audio_path)
        
        similarities = []
        for ref_embedding in user_embeddings:
            similarity = self.compute_similarity(test_embedding, ref_embedding)
            similarities.append(similarity)
        
        max_similarity = max(similarities) if similarities else 0.0
        
        return {
            "matched": max_similarity >= threshold,
            "similarity": max_similarity
        }


class VoiceprintDatabase:
    """声纹数据库管理"""
    
    def __init__(self, db_manager, extractor):
        self.db = db_manager
        self.extractor = extractor
    
    def register_voiceprint(self, user_id, audio_path):
        """注册声纹"""
        embedding = self.extractor.extract_embedding(audio_path)
        vp_id = self.db.add_voiceprint(user_id, embedding, audio_path)
        return vp_id
    
    def verify_speaker(self, audio_path, threshold=SIMILARITY_THRESHOLD):
        """验证说话人身份（识别是谁）

        注意：一个用户可能有多个声纹注册录音（如引导注册的8段），
        此方法会将同一用户的多个相似度取最高值，每个用户只返回一次。
        """
        all_voiceprints = self.db.get_all_voiceprints()

        if not all_voiceprints:
            return {
                "matched": False,
                "user_id": None,
                "user_name": None,
                "similarity": 0.0,
                "message": "声纹数据库为空，请先注册声纹"
            }

        # 1. 获取所有录音的 embedding
        reference_embeddings = [
            (vp['user_id'], vp['embedding']) for vp in all_voiceprints
        ]

        # 2. 获取所有比对分数（每个录音一条）
        result = self.extractor.verify_speaker(audio_path, reference_embeddings, threshold)

        # 3. 如果没匹配到，直接返回
        if not result['matched']:
            result['user_name'] = None
            result['department'] = None
            return result

        # 4. 将 all_scores 按用户聚合：每个用户取最高相似度
        user_scores = {}  # user_id -> (max_similarity, any_similarity)
        for uid, score in result.get('all_scores', []):
            if uid not in user_scores:
                user_scores[uid] = score  # 首次，直接存
            else:
                user_scores[uid] = max(user_scores[uid], score)  # 取最大

        # 5. 按用户得分排序
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)

        # 6. 找到最高分用户
        best_uid, best_score = sorted_users[0]
        is_matched = best_score >= threshold

        # 7. 获取用户信息
        user = self.db.get_user(best_uid)
        best_user_name = user['name'] if user else "未知"
        best_department = user.get('department', "") if user else ""

        # 8. 构建聚合后的 all_scores（每个用户只出现一次）
        aggregated_scores = [(uid, score) for uid, score in sorted_users]

        # 9. 返回结果
        result['user_id'] = best_uid
        result['user_name'] = best_user_name
        result['department'] = best_department
        result['similarity'] = best_score
        result['matched'] = is_matched
        result['all_scores'] = aggregated_scores

        self.db.add_verify_log(
            best_uid,
            best_score,
            "匹配" if is_matched else "不匹配",
            audio_path
        )

        return result
    
    def verify_speaker_by_id(self, audio_path, user_id, threshold=SIMILARITY_THRESHOLD):
        """验证是否是指定用户"""
        user_voiceprints = self.db.get_voiceprints(user_id)
        
        if not user_voiceprints:
            return {
                "matched": False,
                "similarity": 0.0,
                "message": "该用户没有注册声纹"
            }
        
        user_embeddings = [vp['embedding'] for vp in user_voiceprints]
        
        result = self.extractor.verify_speaker_by_id(
            audio_path, user_id, user_embeddings, threshold
        )
        
        user = self.db.get_user(user_id)
        self.db.add_verify_log(
            user_id,
            result['similarity'],
            "匹配" if result['matched'] else "不匹配",
            audio_path
        )
        
        return result


# 测试代码
if __name__ == "__main__":
    from database import DatabaseManager
    
    print("初始化声纹特征提取器...")
    extractor = VoiceprintExtractor()
    
    print("初始化数据库...")
    db = DatabaseManager()
    
    print("初始化声纹数据库...")
    vp_db = VoiceprintDatabase(db, extractor)
    
    print("\n声纹处理模块初始化成功！")
    print(f"使用设备: {extractor.device}")
    print(f"相似度阈值: {SIMILARITY_THRESHOLD}")
    print(f"支持的音频格式: {', '.join(extractor.SUPPORTED_FORMATS)}")
    print(f"FFmpeg路径: {extractor.ffmpeg_path or '未找到'}")
