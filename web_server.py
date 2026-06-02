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
  POST /api/voiceprint/identify - 身份识别（1:N，详细结果）
  POST /api/voiceprint/identify-simple - 身份识别（1:N，只返回身份信息）
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

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SIMILARITY_THRESHOLD, SAMPLE_RATE, AUDIO_DIR, MODEL_DIR, HOST, PORT, DEBUG,
    MODEL_NAME, CUDA_DEVICE_INDEX, ADMIN_USER, ADMIN_PASS,
)
from database import DatabaseManager

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

db_manager = DatabaseManager()
extractor = None
vp_database = None
model_loaded = False
model_error = None
device_info = {}
SUPPORTED_AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.webm'}
ENROLLMENT_AUDIO_DIR = os.path.join(AUDIO_DIR, "enrollment")


def load_model_async():
    global extractor, vp_database, model_loaded, model_error, device_info
    try:
        from voiceprint import VoiceprintExtractor, VoiceprintDatabase, get_gpu_diagnostics
        device_info = get_gpu_diagnostics()
        extractor = VoiceprintExtractor()
        vp_database = VoiceprintDatabase(db_manager, extractor)
        device_info = extractor.get_device_info()
        model_loaded = True
    except Exception as e:
        model_error = str(e)
        try:
            from voiceprint import get_gpu_diagnostics
            device_info = get_gpu_diagnostics()
        except Exception as diagnostic_error:
            device_info = {
                "requires_gpu": True,
                "device_policy": "gpu_required",
                "diagnostic_error": str(diagnostic_error),
            }


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
    stats = db_manager.get_stats()
    return jsonify({
        'status': 'ready' if model_loaded else ('error' if model_error else 'loading'),
        'model_loaded': model_loaded,
        'model_error': model_error,
        'model_name': MODEL_NAME,
        'runtime_requirement': 'NVIDIA CUDA GPU required',
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'sample_rate': SAMPLE_RATE,
        'cuda_device_index': CUDA_DEVICE_INDEX,
        'data': {
            'users': stats['users'],
            'voiceprints': stats['voiceprints'],
            'verify_logs': stats['verify_logs'],
            'audio_dir': AUDIO_DIR,
            'model_dir': MODEL_DIR,
        },
        **device_info,
    })


def model_not_ready_response():
    if model_error:
        return jsonify({'error': f'GPU 模型加载失败: {model_error}'}), 503
    return jsonify({'error': '声纹识别模型尚未加载完成，请稍候'}), 503


def get_enrollment_dir(user_id):
    path = os.path.join(ENROLLMENT_AUDIO_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def validate_audio_ext(filename):
    ext = os.path.splitext(filename or '')[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        raise ValueError(f'不支持的音频格式: {ext or "未知"}')
    return ext


def quality_from_processed_file(path):
    audio, sr = sf.read(path, dtype='float32')
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return extractor.evaluate_audio_quality(audio, sr)


# ========== 数据重置 API ==========


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
        segments = db_manager.get_enrollment_segments(user['id'])
        result.append({
            'id': user['id'],
            'name': user['name'],
            'department': user['department'] or '',
            'voiceprint_count': len(segments) if segments else len(voiceprints),
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
        return model_not_ready_response()

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
    if ext not in SUPPORTED_AUDIO_EXTS:
        return jsonify({'error': f'不支持的音频格式: {ext}'}), 400

    temp_path = None
    try:
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_reg_{uuid.uuid4().hex}{ext}")
        audio_file.save(temp_path)

        token = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join(get_enrollment_dir(user_id), token)
        quality = extractor.prepare_enrollment_audio(temp_path, output_path)
        result = vp_database.batch_register_voiceprint(user_id, [{
            'path': output_path,
            'metadata': {
                'source': 'single',
                'prompt_text': audio_file.filename,
                'quality_score': quality['score'],
                'effective_duration': quality['effective_duration'],
                'duration': quality['duration'],
                'rms': quality['rms'],
                'clipping_ratio': quality['clipping_ratio'],
                'snr_db': quality['snr_db'],
            }
        }], min_quality_score=None, consistency_threshold=None)

        return jsonify({
            'id': result['vp_id'],
            'user_id': user_id,
            'user_name': user['name'],
            'quality': quality,
            'total_segments': result['total_segments'],
            'message': f'声纹注册成功（累计融合 {result["total_segments"]} 段）'
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
        return model_not_ready_response()

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
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'身份识别失败: {str(e)}'}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.route('/api/voiceprint/identify-simple', methods=['POST'])
def identify_speaker_simple():
    if not model_loaded:
        return model_not_ready_response()

    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400

    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({'error': '请上传音频文件'}), 400

    threshold = request.form.get('threshold', SIMILARITY_THRESHOLD, type=float)

    temp_path = None
    try:
        ext = os.path.splitext(audio_file.filename)[1].lower() or '.wav'
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_id_simple_{uuid.uuid4().hex}{ext}")
        audio_file.save(temp_path)

        result = vp_database.verify_speaker(temp_path, threshold)
        return jsonify({
            'matched': result['matched'],
            'user_id': result.get('user_id'),
            'user_name': result.get('user_name'),
            'department': result.get('department', ''),
            'message': result.get('message', ''),
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
        return model_not_ready_response()

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
            'user_id': result.get('user_id', user_id),
            'user_name': result.get('user_name', user['name']),
            'department': result.get('department', user.get('department', '')),
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
        'user_name': log.get('name') or '未知',
        'mode': log.get('mode_code', log.get('mode', 'identify')),
        'mode_label': log.get('mode_label', '身份识别'),
        'similarity': log['similarity'],
        'normalized_confidence': log.get('normalized_confidence'),
        'threshold': log.get('threshold'),
        'threshold_type': log.get('threshold_type'),
        'threshold_type_label': log.get('threshold_type_label'),
        'candidate_count': log.get('candidate_count', 0),
        'result': log['result'],
        'result_code': log.get('result_code', 'no_match'),
        'result_label': log.get('result_label', log['result']),
        'audio_path': log['audio_path'],
        'candidates': log.get('candidates', []),
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


@app.route('/api/voiceprint/enrollment/segment', methods=['POST'])
def prepare_enrollment_segment():
    """上传一段注册音频，生成处理后的可播放 WAV，并返回质量评估。"""
    if not model_loaded:
        return model_not_ready_response()

    user_id = request.form.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': '请选择用户'}), 400
    if not db_manager.get_user(user_id):
        return jsonify({'error': '用户不存在'}), 404
    if 'audio' not in request.files:
        return jsonify({'error': '请上传音频文件'}), 400

    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({'error': '请上传音频文件'}), 400

    temp_path = None
    try:
        ext = validate_audio_ext(audio_file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"vp_stage_{uuid.uuid4().hex}{ext}")
        audio_file.save(temp_path)

        token = f"{uuid.uuid4().hex}.wav"
        output_dir = get_enrollment_dir(user_id)
        output_path = os.path.join(output_dir, token)
        quality = extractor.prepare_enrollment_audio(temp_path, output_path)

        return jsonify({
            'token': token,
            'url': f'/api/voiceprint/enrollment/audio/{user_id}/{token}',
            'quality': quality,
            'sample_rate': SAMPLE_RATE,
            'prompt_index': request.form.get('prompt_index', type=int),
            'prompt_text': request.form.get('prompt_text', ''),
            'source': request.form.get('source', 'upload'),
        })
    except Exception as e:
        return jsonify({'error': f'音频预处理失败: {str(e)}'}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route('/api/voiceprint/enrollment/audio/<int:user_id>/<path:filename>')
def get_enrollment_audio(user_id, filename):
    """播放处理后的注册音频。"""
    safe_name = os.path.basename(filename)
    return send_from_directory(get_enrollment_dir(user_id), safe_name, mimetype='audio/wav')


@app.route('/api/voiceprint/register-batch', methods=['POST'])
def register_voiceprint_batch():
    """批量/增量注册声纹：新片段与该用户历史片段持续融合为一个声纹档案。"""
    if not model_loaded:
        return model_not_ready_response()

    user_id = request.form.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': '请选择用户'}), 400

    user = db_manager.get_user(user_id)
    if not user:
        return jsonify({'error': '用户不存在'}), 404

    files = request.files.getlist('audios')
    staged_segments = []
    if request.form.get('segments'):
        try:
            staged_segments = json.loads(request.form.get('segments'))
            if isinstance(staged_segments, dict):
                staged_segments = [staged_segments]
            if not isinstance(staged_segments, list):
                return jsonify({'error': '片段数据格式错误'}), 400
        except json.JSONDecodeError:
            return jsonify({'error': '片段数据格式错误'}), 400

    if not files and not staged_segments:
        return jsonify({'error': '请上传音频文件或提交已处理片段'}), 400

    temp_paths = []
    audio_items = []
    try:
        for segment in staged_segments:
            token = os.path.basename(segment.get('token', ''))
            if not token:
                continue
            path = os.path.join(get_enrollment_dir(user_id), token)
            if not os.path.exists(path):
                return jsonify({'error': f'已处理片段不存在: {token}'}), 400
            quality = quality_from_processed_file(path)
            audio_items.append({
                'path': path,
                'metadata': {
                    'source': segment.get('source', 'upload'),
                    'prompt_index': segment.get('prompt_index'),
                    'prompt_text': segment.get('prompt_text', ''),
                    'quality_score': quality['score'],
                    'effective_duration': quality['effective_duration'],
                    'duration': quality['duration'],
                    'rms': quality['rms'],
                    'clipping_ratio': quality['clipping_ratio'],
                    'snr_db': quality['snr_db'],
                }
            })

        for audio_file in files:
            ext = validate_audio_ext(audio_file.filename)
            temp_path = os.path.join(tempfile.gettempdir(), f"vp_batch_{uuid.uuid4().hex}{ext}")
            audio_file.save(temp_path)
            temp_paths.append(temp_path)

            token = f"{uuid.uuid4().hex}.wav"
            output_path = os.path.join(get_enrollment_dir(user_id), token)
            quality = extractor.prepare_enrollment_audio(temp_path, output_path)
            audio_items.append({
                'path': output_path,
                'metadata': {
                    'source': 'upload',
                    'prompt_text': audio_file.filename,
                    'quality_score': quality['score'],
                    'effective_duration': quality['effective_duration'],
                    'duration': quality['duration'],
                    'rms': quality['rms'],
                    'clipping_ratio': quality['clipping_ratio'],
                    'snr_db': quality['snr_db'],
                }
            })

        result = vp_database.batch_register_voiceprint(user_id, audio_items)

        return jsonify({
            'total': len(audio_items),
            'success': result['added_segments'],
            'failed': len(result['rejected']),
            'vp_id': result['vp_id'],
            'segment_ids': result['segment_ids'],
            'total_segments': result['total_segments'],
            'rejected': result['rejected'],
            'user_name': user['name'],
            'message': f'声纹注册成功（新增 {result["added_segments"]} 段，累计融合 {result["total_segments"]} 段）'
        })

    except Exception as e:
        return jsonify({'error': f'声纹注册失败: {str(e)}', 'total': len(files) + len(staged_segments), 'success': 0, 'failed': len(files) + len(staged_segments)}), 500

    finally:
        for p in temp_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass


def main():
    print("=" * 50)
    print("Voiceprint Recognition System")
    print("=" * 50)
    print("GPU mode only: NVIDIA CUDA is required")
    print("Loading voiceprint model...")
    print(f"Listening on http://{HOST}:{PORT}")
    print("=" * 50)
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)


if __name__ == '__main__':
    main()
