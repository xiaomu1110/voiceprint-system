"""
数据库管理模块 - 声纹识别系统
"""
import sqlite3
import numpy as np
import os
from datetime import datetime
from config import DB_PATH


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """初始化数据库表结构"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建声纹特征表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voiceprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                audio_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        # 创建验证日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verify_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                similarity REAL,
                result TEXT,
                audio_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_user(self, name, department=""):
        """添加用户"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, department) VALUES (?, ?)",
            (name, department)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    
    def get_user(self, user_id):
        """获取用户信息"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None
    
    def get_all_users(self):
        """获取所有用户"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return users
    
    def delete_user(self, user_id):
        """删除用户及其声纹"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 删除用户的音频文件
        cursor.execute("SELECT audio_path FROM voiceprints WHERE user_id = ?", (user_id,))
        audio_files = cursor.fetchall()
        for row in audio_files:
            if row['audio_path'] and os.path.exists(row['audio_path']):
                try:
                    os.remove(row['audio_path'])
                except:
                    pass
        
        # 删除声纹记录
        cursor.execute("DELETE FROM voiceprints WHERE user_id = ?", (user_id,))
        # 删除用户
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        conn.commit()
        conn.close()
    
    def add_voiceprint(self, user_id, embedding, audio_path=None):
        """添加声纹特征"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 将numpy数组转换为bytes
        embedding_bytes = embedding.tobytes()
        
        cursor.execute(
            "INSERT INTO voiceprints (user_id, embedding, audio_path) VALUES (?, ?, ?)",
            (user_id, embedding_bytes, audio_path)
        )
        vp_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return vp_id
    
    def get_voiceprints(self, user_id):
        """获取用户的所有声纹特征"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM voiceprints WHERE user_id = ?", (user_id,))
        voiceprints = []
        for row in cursor.fetchall():
            vp = dict(row)
            # 将bytes转换回numpy数组
            vp['embedding'] = np.frombuffer(vp['embedding'], dtype=np.float32)
            voiceprints.append(vp)
        conn.close()
        return voiceprints
    
    def get_all_voiceprints(self):
        """获取所有声纹特征"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT v.*, u.name, u.department 
            FROM voiceprints v 
            JOIN users u ON v.user_id = u.id
        ''')
        voiceprints = []
        for row in cursor.fetchall():
            vp = dict(row)
            vp['embedding'] = np.frombuffer(vp['embedding'], dtype=np.float32)
            voiceprints.append(vp)
        conn.close()
        return voiceprints
    
    def delete_voiceprint(self, vp_id):
        """删除单个声纹"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 获取音频文件路径
        cursor.execute("SELECT audio_path FROM voiceprints WHERE id = ?", (vp_id,))
        row = cursor.fetchone()
        if row and row['audio_path'] and os.path.exists(row['audio_path']):
            try:
                os.remove(row['audio_path'])
            except:
                pass
        
        cursor.execute("DELETE FROM voiceprints WHERE id = ?", (vp_id,))
        conn.commit()
        conn.close()
    
    def add_verify_log(self, user_id, similarity, result, audio_path=None):
        """添加验证日志"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO verify_logs (user_id, similarity, result, audio_path) VALUES (?, ?, ?, ?)",
            (user_id, similarity, result, audio_path)
        )
        conn.commit()
        conn.close()
    
    def get_verify_logs(self, limit=100):
        """获取验证日志"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT l.*, u.name 
            FROM verify_logs l 
            LEFT JOIN users u ON l.user_id = u.id
            ORDER BY l.created_at DESC 
            LIMIT ?
        ''', (limit,))
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return logs
    
    def clear_verify_logs(self):
        """清空验证日志"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM verify_logs")
        conn.commit()
        conn.close()


# 测试代码
if __name__ == "__main__":
    db = DatabaseManager()
    
    # 测试添加用户
    user_id = db.add_user("张三", "配电运维部")
    print(f"添加用户成功，ID: {user_id}")
    
    # 测试获取用户
    user = db.get_user(user_id)
    print(f"获取用户: {user}")
    
    # 测试添加声纹
    embedding = np.random.randn(192).astype(np.float32)
    vp_id = db.add_voiceprint(user_id, embedding, "test.wav")
    print(f"添加声纹成功，ID: {vp_id}")
    
    # 测试获取声纹
    voiceprints = db.get_voiceprints(user_id)
    print(f"获取声纹: {len(voiceprints)} 条记录")
    
    # 测试获取所有用户
    users = db.get_all_users()
    print(f"所有用户: {users}")
    
    # 测试删除用户
    db.delete_user(user_id)
    print("删除用户成功")
    
    print("数据库模块测试完成！")
