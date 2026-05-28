"""
Web 服务器 - 声纹识别系统

基于 Flask 提供 REST API 和静态页面服务。
模型在后台线程异步加载，前端通过 /api/status 轮询加载状态。

API 端点：
  GET  /api/status              - 系统状态
  GET  /api/users               - 用户列表
  POST /api/users               - 添加用户
  DELETE /api/users/<id>        - 删除用户
  POST /api/voiceprint/register - 单条声纹注册
  POST /api/voiceprint/register-batch - 批量注册（多条录音融合为一个向量）
  POST /api/voiceprint/identify - 身份识别（1:N）
  POST /api/voiceprint/verify   - 身份验证（1:1）
  GET  /api/logs                - 验证日志
  DELETE /api/logs              - 清空日志
  POST /api/reset               - 重置系统（需管理员认证）
"""
import os
import sys
import threading
import tempfile
import uuid
import json
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SIMILARITY_THRESHOLD, SAMPLE_RATE, AUDIO_DIR, MODEL_DIR
from database import DatabaseManager

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

db_manager = DatabaseManager()
extractor = None
vp_database = None
model_loaded = False
model_error = None


def load_model_async():
    global extractor, vp_database, model_loaded, model_error
    try:
        from voiceprint import VoiceprintExtractor, VoiceprintDatabase
        extractor = VoiceprintExtractor()
        vp_database = VoiceprintDatabase(db_manager, extractor)
        model_loaded = True
    except Exception as e:
        model_error = str(e)


threading.Thread(target=load_model_async, daemon=True).start()


# ========== 页面路由 ==========

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


# ========== 系统状态 API ==========

@app.route('/api/status')
def get_status():
    return jsonify({
        'model_loaded': model_loaded,
        'model_error': model_error,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'sample_rate': SAMPLE_RATE
    })


# ========== 数据重置 API ==========

ADMIN_USER = 'admin'
ADMIN_PASS = '123456'


@app.route('/api/reset', methods=['POST'])
def reset_system():
    """重置系统数据（需要管理员验证）"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '请提供认证信息'}), 400

    username = data.get('username', '')
    password = data.get('password', '')

    if username != ADMIN_USER or password != ADMIN_PASS:
        return jsonify({'error': '账号或密码错误'}), 401

    try:
        # 删除所有用户及其声纹和音频文件
        users = db_manager.get_all_users()
        for user in users:
            db_manager.delete_user(user['id'])

        # 清空验证日志
        db_manager.clear_verify_logs()

        return jsonify({'message': f'系统已重置，删除了 {len(users)} 个用户'})
    except Exception as e:
        return jsonify({'error': f'重置失败: {str(e)}'}), 500


# ========== 用户管理 API ==========

@app.route('/api/users', methods=['GET'])
def get_users():
    users = db_manager.get_all_users()
    result = []
    for user in users:
        voiceprints = db_manager.get_voiceprints(user['id'])
        result.append({
            'id': user['id'],
            'name': user['name'],
            'department': user['department'] or '',
            'voiceprint_count': len(voiceprints),
            'created_at': user['created_at']
        })
    return jsonify(result)


@app.route('/api/users', methods=['POST'])
def add_user():
    data = request.get_json()
    if not data or not data.get('name', '').strip():
        return jsonify({'error': '用户名不能为空'}), 400
    name = data['name'].strip()
    department = data.get('department', '').strip()
    try:
        user_id = db_manager.add_user(name, department)
        return jsonify({'id': user_id, 'name': name, 'department': department}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = db_manager.get_user(user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404
    try:
        db_manager.delete_user(user_id)
        return jsonify({'message': f'用户 {user["name"]} 已删除'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:user_id>/voiceprints', methods=['GET'])
def get_user_voiceprints(user_id):
    user = db_manager.get_user(user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404
    voiceprints = db_manager.get_voiceprints(user_id)
    result = [{
        'id': vp['id'],
        'user_id': vp['user_id'],
        'audio_path': vp['audio_path'],
        'created_at': vp['created_at']
    } for vp in voiceprints]
    return jsonify(result)


@app.route('/api/voiceprints/<int:vp_id>', methods=['DELETE'])
def delete_voiceprint(vp_id):
    try:
        db_manager.delete_voiceprint(vp_id)
        return jsonify({'message': '声纹已删除'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== 声纹注册 API ==========

@app.route('/api/voiceprint/register', methods=['POST'])
def register_voiceprint():
    if not model_loaded:
        return jsonify({'error': '声纹识别模型尚未加载完成，请稍候'}), 503

    user_id = request.form.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': '请选择用户'}), 400

    user = db_manager.get_user(user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404

    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400

    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({'error': '请上传音频文件'}), 400

    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.webm'}:
        return jsonify({'error': f'不支持的音频格式: {ext}'}), 400

    temp_path = None
    try:
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_reg_{uuid.uuid4().hex}{ext}")
        audio_file.save(temp_path)

        filename = f"{user_id}_{uuid.uuid4().hex[:8]}.wav"
        dest_path = os.path.join(AUDIO_DIR, filename)

        vp_id = vp_database.register_voiceprint(user_id, temp_path)

        return jsonify({
            'id': vp_id,
            'user_id': user_id,
            'user_name': user['name'],
            'message': '声纹注册成功'
        }), 201

    except Exception as e:
        return jsonify({'error': f'声纹注册失败: {str(e)}'}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ========== 身份识别 API ==========

@app.route('/api/voiceprint/identify', methods=['POST'])
def identify_speaker():
    if not model_loaded:
        return jsonify({'error': '声纹识别模型尚未加载完成，请稍候'}), 503

    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400

    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({'error': '请上传音频文件'}), 400

    threshold = request.form.get('threshold', SIMILARITY_THRESHOLD, type=float)

    temp_path = None
    try:
        ext = os.path.splitext(audio_file.filename)[1].lower() or '.wav'
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_id_{uuid.uuid4().hex}{ext}")
        audio_file.save(temp_path)

        result = vp_database.verify_speaker(temp_path, threshold)

        # 聚合 all_scores：同一用户只保留最高分
        raw_scores = result.get('all_scores', [])
        user_best = {}
        for uid, score in raw_scores:
            if uid not in user_best or score > user_best[uid]:
                user_best[uid] = score
        aggregated = sorted(user_best.items(), key=lambda x: x[1], reverse=True)[:10]

        return jsonify({
            'matched': result['matched'],
            'user_id': result.get('user_id'),
            'user_name': result.get('user_name'),
            'department': result.get('department', ''),
            'similarity': result.get('similarity', 0),
            'message': result.get('message', ''),
            'all_scores': [
                {'user_id': uid, 'user_name': db_manager.get_user(uid)['name'] if db_manager.get_user(uid) else '未知', 'similarity': score}
                for uid, score in aggregated
            ]
        })

    except Exception as e:
        return jsonify({'error': f'身份识别失败: {str(e)}'}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ========== 身份验证 API ==========

@app.route('/api/voiceprint/verify', methods=['POST'])
def verify_speaker():
    if not model_loaded:
        return jsonify({'error': '声纹识别模型尚未加载完成，请稍候'}), 503

    user_id = request.form.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': '请选择要验证的用户'}), 400

    user = db_manager.get_user(user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404

    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400

    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({'error': '请上传音频文件'}), 400

    threshold = request.form.get('threshold', SIMILARITY_THRESHOLD, type=float)

    temp_path = None
    try:
        ext = os.path.splitext(audio_file.filename)[1].lower() or '.wav'
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_vf_{uuid.uuid4().hex}{ext}")
        audio_file.save(temp_path)

        result = vp_database.verify_speaker_by_id(temp_path, user_id, threshold)

        return jsonify({
            'matched': result['matched'],
            'similarity': result['similarity'],
            'user_id': user_id,
            'user_name': user['name'],
            'department': user.get('department', ''),
            'message': result.get('message', '')
        })

    except Exception as e:
        return jsonify({'error': f'身份验证失败: {str(e)}'}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# ========== 日志 API ==========

@app.route('/api/logs', methods=['GET'])
def get_logs():
    limit = request.args.get('limit', 1000, type=int)
    logs = db_manager.get_verify_logs(limit)
    result = [{
        'id': log['id'],
        'user_id': log['user_id'],
        'user_name': log.get('name', '未知'),
        'similarity': log['similarity'],
        'result': log['result'],
        'audio_path': log['audio_path'],
        'created_at': log['created_at']
    } for log in logs]
    return jsonify(result)


@app.route('/api/logs', methods=['DELETE'])
def clear_logs():
    try:
        db_manager.clear_verify_logs()
        return jsonify({'message': '日志已清空'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== 引导注册 API ==========

REGISTRATION_TEXTS = [
    {"category": "元音覆盖", "text": "山丹丹花开红艳艳，嘀嗒嘀嗒嘀嗒", "tone": "平静、清晰"},
    {"category": "鼻音/边音", "text": "蓝蓝天上白云飘，白云下面马儿跑", "tone": "平静、清晰"},
    {"category": "声调覆盖", "text": "人工智能改变世界，科技创新引领未来", "tone": "缓慢、字正腔圆"},
    {"category": "唇齿音", "text": "粉红凤凰飞，红旗飘扬在风中", "tone": "正常语速"},
    {"category": "开口音", "text": "配电柜操作需谨慎，安全防护第一重要", "tone": "严肃、清晰"},
    {"category": "情感表达", "text": "您好，我是配电运维班班长张伟，现在为您演示声纹识别系统", "tone": "自然、亲切"},
    {"category": "数字与专业术语", "text": "电流电压正常，系统运行稳定，各项参数均在安全范围", "tone": "汇报语气"},
    {"category": "日常口语", "text": "喂，小张，今天的巡检工作准备好了吗", "tone": "自然对话语气"},
]


@app.route('/api/registration/texts', methods=['GET'])
def get_registration_texts():
    return jsonify(REGISTRATION_TEXTS)


@app.route('/api/voiceprint/register-batch', methods=['POST'])
def register_voiceprint_batch():
    """批量注册声纹：多条录音提取embedding后取平均，融合为一个声纹向量"""
    if not model_loaded:
        return jsonify({'error': '声纹识别模型尚未加载完成，请稍候'}), 503

    user_id = request.form.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': '请选择用户'}), 400

    user = db_manager.get_user(user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404

    files = request.files.getlist('audios')
    if not files:
        return jsonify({'error': '请上传音频文件'}), 400

    temp_paths = []
    results = []
    try:
        for audio_file in files:
            ext = os.path.splitext(audio_file.filename)[1].lower() or '.wav'
            temp_path = os.path.join(tempfile.gettempdir(), f"vp_batch_{uuid.uuid4().hex}{ext}")
            audio_file.save(temp_path)
            temp_paths.append(temp_path)
            results.append({'filename': audio_file.filename, 'success': True})

        # 多条录音融合为一个声纹向量
        vp_id = vp_database.batch_register_voiceprint(user_id, temp_paths)

        return jsonify({
            'total': len(files),
            'success': len(files),
            'failed': 0,
            'vp_id': vp_id,
            'user_name': user['name'],
            'message': f'声纹注册成功（{len(files)}条录音已融合）'
        })

    except Exception as e:
        return jsonify({'error': f'声纹注册失败: {str(e)}', 'total': len(files), 'success': 0, 'failed': len(files)}), 500

    finally:
        for p in temp_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass


if __name__ == '__main__':
    print("=" * 50)
    print("配电柜智能语音识别系统 - Web版")
    print("=" * 50)
    print(f"正在加载声纹识别模型...")
    print(f"请访问 http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
