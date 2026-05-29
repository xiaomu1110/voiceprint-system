// ========== 全局状态 ==========
let modelLoaded = false;
let usersList = [];
let systemStatus = null;

// 录音相关
let regMediaRecorder = null;
let regAudioChunks = [];
let regRecordTimer = null;
let regRecordSeconds = 0;

let verifyMediaRecorder = null;
let verifyAudioChunks = [];
let verifyRecordTimer = null;
let verifyRecordSeconds = 0;

let guideMediaRecorder = null;
let guideAudioChunks = [];
let guideRecordTimer = null;
let guideRecordSeconds = 0;

// 引导注册状态
let guideTexts = [];
let guideCurrentIndex = 0;
let guideResults = [];
let guideSegments = [];

// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initFileInputs();
    checkModelStatus();
    loadUsers();
});

// ========== 标签切换 ==========
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('tab-' + btn.dataset.tab).classList.add('active');

            if (btn.dataset.tab === 'users') loadUsers();
            if (btn.dataset.tab === 'register') refreshUserSelects();
            if (btn.dataset.tab === 'verify') refreshUserSelects();
            if (btn.dataset.tab === 'logs') loadLogs();
        });
    });
}

// ========== 文件输入初始化 ==========
function initFileInputs() {
    document.getElementById('reg-file').addEventListener('change', function() {
        document.getElementById('reg-file-name').textContent = this.files[0] ? this.files[0].name : '未选择文件';
    });
    document.getElementById('verify-file').addEventListener('change', function() {
        document.getElementById('verify-file-name').textContent = this.files[0] ? this.files[0].name : '未选择文件';
    });
    document.getElementById('guide-file').addEventListener('change', function() {
        const file = this.files[0];
        this.value = '';
        if (file) processGuideAudio(file, 'upload');
    });
    document.getElementById('add-user-form').addEventListener('submit', function(e) {
        e.preventDefault();
        addUser();
    });
}

// ========== 模型状态检查 ==========
function checkModelStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            systemStatus = data;
            renderSystemStatus(data);
            modelLoaded = data.model_loaded;
            if (data.model_loaded) {
                document.getElementById('loading-overlay').classList.add('hidden');
                document.getElementById('status-indicator').className = 'status-dot status-ready';
                document.getElementById('status-text').textContent = `GPU 就绪 · ${data.device_name || data.gpu_name || 'CUDA'}`;
            } else if (data.model_error) {
                document.getElementById('loading-overlay').classList.add('hidden');
                document.getElementById('loading-text').textContent = '模型加载失败: ' + data.model_error;
                document.getElementById('status-indicator').className = 'status-dot status-error';
                document.getElementById('status-text').textContent = 'GPU 模型加载失败';
            } else {
                document.getElementById('status-text').textContent = 'GPU 模型加载中...';
                setTimeout(checkModelStatus, 1000);
            }
        })
        .catch(() => setTimeout(checkModelStatus, 2000));
}

function renderSystemStatus(data) {
    setText('runtime-mode', data.model_runtime === 'gpu' ? 'GPU 加速' : '仅支持 GPU');
    setText('gpu-device', data.device_name || data.gpu_name || firstSmiGpuName(data) || '未检测到可用 CUDA GPU');
    setText('cuda-state', data.cuda_available ? `可用 (${data.gpu_count || 0} 个设备)` : '不可用');
    setText('torch-state', `${data.torch_version || '未知'} / CUDA ${data.torch_cuda_version || '无'}`);
    setText('model-state', data.model_loaded ? '已加载' : (data.model_error ? '加载失败' : '加载中'));

    const stats = data.data || {};
    setText('data-state', `用户 ${stats.users ?? '-'} · 声纹 ${stats.voiceprints ?? '-'} · 日志 ${stats.verify_logs ?? '-'}`);

    const lines = [
        `设备策略: ${data.runtime_requirement || 'NVIDIA CUDA GPU required'}`,
        `当前设备: ${data.device || '-'}`,
        `GPU 名称: ${data.device_name || data.gpu_name || '-'}`,
        `显存: ${data.gpu_memory_allocated || '-'} 已用 / ${data.gpu_memory_reserved || '-'} 预留 / ${data.gpu_memory_total || '-' } 总量`,
        `计算能力: ${data.gpu_compute_capability || '-'}`,
        `cuDNN: ${data.cudnn_enabled ? '启用' : '未启用'}${data.cudnn_version ? ' (' + data.cudnn_version + ')' : ''}`,
        `nvidia-smi: ${data.nvidia_smi_available ? '可用' : '不可用'}`,
    ];

    if (data.nvidia_smi_gpus && data.nvidia_smi_gpus.length > 0) {
        lines.push('驱动检测: ' + data.nvidia_smi_gpus.map(g => `${g.name} / Driver ${g.driver_version} / ${g.memory_total}`).join('; '));
    }
    if (data.model_error) {
        lines.push(`失败原因: ${data.model_error}`);
    }
    if (data.diagnostic_error) {
        lines.push(`诊断错误: ${data.diagnostic_error}`);
    }

    const detail = document.getElementById('gpu-detail');
    if (detail) detail.textContent = lines.join('\n');
}

function firstSmiGpuName(data) {
    return data.nvidia_smi_gpus && data.nvidia_smi_gpus[0] ? data.nvidia_smi_gpus[0].name : '';
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// ========== 用户管理 ==========
function loadUsers() {
    fetch('/api/users')
        .then(r => r.json())
        .then(users => {
            usersList = users;
            const tbody = document.getElementById('users-tbody');
            tbody.innerHTML = users.length === 0
                ? '<tr><td colspan="6" style="text-align:center;color:#999;">暂无用户</td></tr>'
                : users.map(u => `
                    <tr>
                        <td>${u.id}</td>
                        <td>${escHtml(u.name)}</td>
                        <td>${escHtml(u.department)}</td>
                        <td>${u.voiceprint_count}</td>
                        <td>${u.created_at || ''}</td>
                        <td>
                            <button class="btn btn-sm btn-sm-danger" onclick="deleteUser(${u.id}, '${escHtml(u.name)}')">删除</button>
                        </td>
                    </tr>
                `).join('');
            refreshUserSelects();
        })
        .catch(err => console.error('加载用户失败:', err));
}

function addUser() {
    const name = document.getElementById('user-name').value.trim();
    const dept = document.getElementById('user-dept').value.trim();
    if (!name) return;

    fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, department: dept })
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            alert('添加失败: ' + data.error);
        } else {
            document.getElementById('user-name').value = '';
            document.getElementById('user-dept').value = '';
            loadUsers();
        }
    })
    .catch(err => alert('添加失败: ' + err));
}

function deleteUser(id, name) {
    if (!confirm(`确定要删除用户 "${name}" 及其所有声纹吗？`)) return;
    fetch(`/api/users/${id}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(data => {
            if (data.error) alert('删除失败: ' + data.error);
            else loadUsers();
        })
        .catch(err => alert('删除失败: ' + err));
}

function refreshUserSelects() {
    const options = usersList.length === 0
        ? '<option value="">暂无用户</option>'
        : usersList.map(u => `<option value="${u.id}">${escHtml(u.name)}</option>`).join('');
    ['reg-user', 'verify-user', 'guide-user'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = options;
    });
}

// ========== 声纹注册 ==========
function registerVoiceprint() {
    const userId = document.getElementById('reg-user').value;
    if (!userId) { showStatus('reg-status', '请选择用户', 'error'); return; }

    const fileInput = document.getElementById('reg-file');
    const audioPlayback = document.getElementById('reg-audio-playback');
    let audioFile = fileInput.files[0];

    if (!audioFile && audioPlayback.src && audioPlayback.src !== window.location.href) {
        // 使用录音的blob
        const blob = regRecordedBlob;
        if (!blob) { showStatus('reg-status', '请选择音频文件或录制音频', 'error'); return; }
        audioFile = new File([blob], 'recording.webm', { type: 'audio/webm' });
    }

    if (!audioFile) { showStatus('reg-status', '请选择音频文件或录制音频', 'error'); return; }

    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('audio', audioFile);

    showStatus('reg-status', '正在注册声纹...', 'loading');
    document.getElementById('reg-submit-btn').disabled = true;

    fetch('/api/voiceprint/register', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                showStatus('reg-status', data.error, 'error');
            } else {
                showStatus('reg-status', `声纹注册成功！声纹ID: ${data.id}`, 'success');
                fileInput.value = '';
                document.getElementById('reg-file-name').textContent = '未选择文件';
                audioPlayback.style.display = 'none';
                regRecordedBlob = null;
                loadUsers();
            }
        })
        .catch(err => showStatus('reg-status', '注册失败: ' + err, 'error'))
        .finally(() => document.getElementById('reg-submit-btn').disabled = false);
}

let regRecordedBlob = null;

// ========== 录音功能 ==========
async function startRecording(mediaRecorderRef, audioChunksRef, onBlobReady) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream, { mimeType: getSupportedMimeType() });
        const chunks = [];

        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
        mediaRecorder.onstop = () => {
            const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
            stream.getTracks().forEach(t => t.stop());
            onBlobReady(blob);
        };

        mediaRecorder.start(100);
        return mediaRecorder;
    } catch (err) {
        alert('无法访问麦克风: ' + err.message);
        return null;
    }
}

function getSupportedMimeType() {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
    for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return '';
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
}

// 注册页录音
function toggleRegRecording() {
    const btn = document.getElementById('reg-record-btn');
    if (regMediaRecorder && regMediaRecorder.state === 'recording') {
        regMediaRecorder.stop();
        btn.classList.remove('recording');
        btn.innerHTML = '<span class="record-icon"></span> 开始录音';
        clearInterval(regRecordTimer);
        return;
    }

    regRecordSeconds = 0;
    document.getElementById('reg-record-time').textContent = '00:00';
    regRecordTimer = setInterval(() => {
        regRecordSeconds++;
        document.getElementById('reg-record-time').textContent = formatTime(regRecordSeconds);
    }, 1000);

    startRecording(null, null, blob => {
        regRecordedBlob = blob;
        const audio = document.getElementById('reg-audio-playback');
        audio.src = URL.createObjectURL(blob);
        audio.style.display = 'block';
    }).then(recorder => {
        if (recorder) {
            regMediaRecorder = recorder;
            btn.classList.add('recording');
            btn.innerHTML = '<span class="record-icon"></span> 停止录音';
        } else {
            clearInterval(regRecordTimer);
        }
    });
}

// 验证页录音
let verifyRecordedBlob = null;

function toggleVerifyRecording() {
    const btn = document.getElementById('verify-record-btn');
    if (verifyMediaRecorder && verifyMediaRecorder.state === 'recording') {
        verifyMediaRecorder.stop();
        btn.classList.remove('recording');
        btn.innerHTML = '<span class="record-icon"></span> 开始录音';
        clearInterval(verifyRecordTimer);
        return;
    }

    verifyRecordSeconds = 0;
    document.getElementById('verify-record-time').textContent = '00:00';
    verifyRecordTimer = setInterval(() => {
        verifyRecordSeconds++;
        document.getElementById('verify-record-time').textContent = formatTime(verifyRecordSeconds);
    }, 1000);

    startRecording(null, null, blob => {
        verifyRecordedBlob = blob;
        const audio = document.getElementById('verify-audio-playback');
        audio.src = URL.createObjectURL(blob);
        audio.style.display = 'block';
    }).then(recorder => {
        if (recorder) {
            verifyMediaRecorder = recorder;
            btn.classList.add('recording');
            btn.innerHTML = '<span class="record-icon"></span> 停止录音';
        } else {
            clearInterval(verifyRecordTimer);
        }
    });
}

// ========== 身份验证 ==========
function onVerifyModeChange() {
    const mode = document.querySelector('input[name="verify-mode"]:checked').value;
    document.getElementById('verify-user-group').style.display = mode === 'verify' ? 'block' : 'none';
}

function startVerification() {
    const mode = document.querySelector('input[name="verify-mode"]:checked').value;
    const threshold = document.getElementById('threshold-slider').value;
    const fileInput = document.getElementById('verify-file');
    let audioFile = fileInput.files[0];

    if (!audioFile && verifyRecordedBlob) {
        audioFile = new File([verifyRecordedBlob], 'recording.webm', { type: 'audio/webm' });
    }

    if (!audioFile) { showStatus('verify-status', '请选择音频文件或录制音频', 'error'); return; }

    const formData = new FormData();
    formData.append('audio', audioFile);
    formData.append('threshold', threshold);

    let url = '/api/voiceprint/identify';
    if (mode === 'verify') {
        const userId = document.getElementById('verify-user').value;
        if (!userId) { showStatus('verify-status', '请选择要验证的用户', 'error'); return; }
        formData.append('user_id', userId);
        url = '/api/voiceprint/verify';
    }

    showStatus('verify-status', '正在验证...', 'loading');

    fetch(url, { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                showStatus('verify-status', data.error, 'error');
                return;
            }
            showStatus('verify-status', '验证完成', 'success');
            if (mode === 'verify') {
                showVerifyResult(data);
            } else {
                showIdentifyResult(data);
            }
        })
        .catch(err => showStatus('verify-status', '验证失败: ' + err, 'error'));
}

function showVerifyResult(data) {
    const icon = document.getElementById('result-icon');
    const text = document.getElementById('result-text');
    const detail = document.getElementById('result-detail');

    if (data.matched) {
        icon.textContent = '✓';
        icon.className = 'result-icon success';
        text.textContent = `验证通过: ${data.user_name}`;
    } else {
        icon.textContent = '✗';
        icon.className = 'result-icon fail';
        text.textContent = `验证失败: 不是 ${data.user_name}`;
    }

    detail.textContent = [
        `验证模式: 身份验证`,
        `目标用户: ${data.user_name}`,
        `用户部门: ${data.department || '-'}`,
        `相似度阈值: ${document.getElementById('threshold-slider').value}`,
        `实际相似度: ${data.similarity.toFixed(4)}`,
        `验证结果: ${data.matched ? '通过' : '失败'}`
    ].join('\n');
}

function showIdentifyResult(data) {
    const icon = document.getElementById('result-icon');
    const text = document.getElementById('result-text');
    const detail = document.getElementById('result-detail');

    if (data.matched) {
        icon.textContent = '✓';
        icon.className = 'result-icon success';
        text.textContent = `识别成功: ${data.user_name}`;
    } else {
        icon.textContent = '✗';
        icon.className = 'result-icon fail';
        text.textContent = '识别失败: 未找到匹配身份';
    }

    let lines = [
        `验证模式: 身份识别`,
        `相似度阈值: ${document.getElementById('threshold-slider').value}`,
        `最高相似度: ${data.similarity.toFixed(4)}`,
    ];
    if (data.matched) {
        lines.push(`匹配用户: ${data.user_name}`);
        lines.push(`用户部门: ${data.department || '-'}`);
    }
    lines.push(`\n验证结果: ${data.matched ? '成功' : '失败'}`);

    if (data.all_scores && data.all_scores.length > 0) {
        lines.push('\n--- 所有用户相似度 ---');
        data.all_scores.forEach(s => {
            lines.push(`${s.user_name}: ${s.similarity.toFixed(4)}`);
        });
    }

    detail.textContent = lines.join('\n');
}

function clearResult() {
    document.getElementById('result-icon').textContent = '';
    document.getElementById('result-icon').className = 'result-icon';
    document.getElementById('result-text').textContent = '等待验证...';
    document.getElementById('result-detail').textContent = '';
    document.getElementById('verify-status').className = 'status-msg';
    document.getElementById('verify-status').textContent = '';
}

// ========== 验证日志 ==========
function loadLogs() {
    fetch('/api/logs')
        .then(r => r.json())
        .then(logs => {
            const tbody = document.getElementById('logs-tbody');
            tbody.innerHTML = logs.length === 0
                ? '<tr><td colspan="5" style="text-align:center;color:#999;">暂无日志</td></tr>'
                : logs.map(l => `
                    <tr>
                        <td>${l.id}</td>
                        <td>${escHtml(l.user_name)}</td>
                        <td>${l.similarity ? l.similarity.toFixed(4) : 'N/A'}</td>
                        <td style="color:${l.result === '匹配' ? '#4caf50' : '#f44336'}">${escHtml(l.result)}</td>
                        <td>${l.created_at || ''}</td>
                    </tr>
                `).join('');

            const total = logs.length;
            const success = logs.filter(l => l.result === '匹配').length;
            document.getElementById('logs-stats').textContent = `共 ${total} 条记录 | 成功: ${success} | 失败: ${total - success}`;
        })
        .catch(err => console.error('加载日志失败:', err));
}

function clearLogs() {
    if (!confirm('确定要清空所有验证日志吗？')) return;
    fetch('/api/logs', { method: 'DELETE' })
        .then(r => r.json())
        .then(data => {
            if (data.error) alert('清空失败: ' + data.error);
            else loadLogs();
        })
        .catch(err => alert('清空失败: ' + err));
}

// ========== 引导注册 ==========
function startGuideRegistration() {
    const userId = document.getElementById('guide-user').value;
    if (!userId) { alert('请选择用户'); return; }

    guideCurrentIndex = 0;
    guideResults = [];
    guideSegments = [];

    // 重置按钮状态
    document.getElementById('guide-skip-btn').style.display = 'inline-block';
    document.getElementById('guide-finish-btn').style.display = 'inline-block';
    document.getElementById('guide-finish-btn').disabled = true;
    document.getElementById('guide-record-btn').disabled = false;
    document.getElementById('guide-upload-btn').disabled = false;
    document.getElementById('guide-processed-audio').style.display = 'none';

    fetch('/api/registration/texts')
        .then(r => r.json())
        .then(texts => {
            guideTexts = texts;
            document.getElementById('guide-area').style.display = 'block';
            loadGuideSegment(0);
        })
        .catch(err => alert('加载注册文本失败: ' + err));
}

function loadGuideSegment(index) {
    if (index >= guideTexts.length) {
        // 所有段落录完，显示完成按钮
        document.getElementById('guide-quality').textContent = '所有段落录制完成，请点击"完成注册"';
        document.getElementById('guide-quality').style.color = '#1a73e8';
        document.getElementById('guide-record-btn').disabled = true;
        document.getElementById('guide-upload-btn').disabled = true;
        document.getElementById('guide-skip-btn').style.display = 'none';
        document.getElementById('guide-finish-btn').style.display = 'inline-block';
        document.getElementById('guide-progress-text').textContent = `${guideResults.length}/${guideTexts.length} 完成`;
        document.getElementById('guide-progress-fill').style.width = '100%';
        return;
    }

    guideCurrentIndex = index;
    const text = guideTexts[index];

    document.getElementById('guide-category').textContent = `[${text.category}]`;
    document.getElementById('guide-text').textContent = text.text;
    document.getElementById('guide-tone').textContent = `语调提示：${text.tone}`;
    document.getElementById('guide-record-time').textContent = '00:00';
    document.getElementById('guide-quality').textContent = '';

    const total = guideTexts.length;
    document.getElementById('guide-progress-text').textContent = `${index}/${total} 完成`;
    document.getElementById('guide-progress-fill').style.width = `${(index / total) * 100}%`;

    // 更新按钮状态
    const btn = document.getElementById('guide-record-btn');
    btn.classList.remove('recording');
    btn.innerHTML = '<span class="record-icon"></span> 开始录音';
    btn.disabled = false;
    document.getElementById('guide-upload-btn').disabled = false;
}

function toggleGuideRecording() {
    const btn = document.getElementById('guide-record-btn');
    if (guideMediaRecorder && guideMediaRecorder.state === 'recording') {
        guideMediaRecorder.stop();
        btn.classList.remove('recording');
        btn.innerHTML = '<span class="record-icon"></span> 开始录音';
        clearInterval(guideRecordTimer);
        return;
    }

    guideRecordSeconds = 0;
    document.getElementById('guide-record-time').textContent = '00:00';
    guideRecordTimer = setInterval(() => {
        guideRecordSeconds++;
        document.getElementById('guide-record-time').textContent = formatTime(guideRecordSeconds);
    }, 1000);

    startRecording(null, null, blob => {
        const audioFile = new File([blob], `guide_${guideCurrentIndex + 1}.webm`, { type: 'audio/webm' });
        processGuideAudio(audioFile, 'recording');
    }).then(recorder => {
        if (recorder) {
            guideMediaRecorder = recorder;
            btn.classList.add('recording');
            btn.innerHTML = '<span class="record-icon"></span> 停止录音';
        } else {
            clearInterval(guideRecordTimer);
        }
    });
}

function processGuideAudio(audioFile, source) {
    const userId = document.getElementById('guide-user').value;
    if (!userId) { alert('请选择用户'); return; }
    if (!guideTexts.length) { alert('请先开始引导注册'); return; }

    const currentText = guideTexts[guideCurrentIndex] || { text: '', category: '', tone: '' };
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('audio', audioFile);
    formData.append('source', source);
    formData.append('prompt_index', guideCurrentIndex);
    formData.append('prompt_text', currentText.text);

    const qualityEl = document.getElementById('guide-quality');
    qualityEl.textContent = '正在预处理音频...';
    qualityEl.style.color = '#1a73e8';
    document.getElementById('guide-record-btn').disabled = true;
    document.getElementById('guide-upload-btn').disabled = true;

    fetch('/api/voiceprint/enrollment/segment', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            const quality = data.quality;
            const segment = {
                token: data.token,
                playback_url: data.url,
                source,
                prompt_index: guideCurrentIndex,
                text: currentText.text || audioFile.name,
                quality,
            };
            guideSegments.push(segment);
            guideResults.push(segment);

            const audio = document.getElementById('guide-processed-audio');
            audio.src = data.url;
            audio.style.display = 'block';

            qualityEl.textContent = `质量: ${quality.quality_text} (${quality.score}分，有效语音 ${quality.effective_duration}s) · ${quality.message}`;
            qualityEl.style.color = quality.score >= 85 ? '#4caf50' : quality.score >= 70 ? '#ff9800' : quality.score >= 60 ? '#1a73e8' : '#f44336';
            document.getElementById('guide-finish-btn').disabled = guideSegments.length === 0;
            updateGuideResults();

            if (guideCurrentIndex < guideTexts.length - 1) {
                setTimeout(() => loadGuideSegment(guideCurrentIndex + 1), 600);
            } else {
                setTimeout(() => loadGuideSegment(guideTexts.length), 600);
            }
        })
        .catch(err => {
            qualityEl.textContent = '预处理失败: ' + err.message;
            qualityEl.style.color = '#f44336';
        })
        .finally(() => {
            if (guideCurrentIndex < guideTexts.length) {
                document.getElementById('guide-record-btn').disabled = false;
                document.getElementById('guide-upload-btn').disabled = false;
            }
        });
}

function evaluateRecordingQuality(blob, duration) {
    // 简单的时长评估
    let durationScore, durationText;
    if (duration >= 2 && duration <= 8) {
        durationScore = 95;
        durationText = '良好';
    } else if (duration >= 1 && duration <= 12) {
        durationScore = 70;
        durationText = '一般';
    } else {
        durationScore = 50;
        durationText = '待改进';
    }

    return { score: durationScore, duration, quality_text: durationText };
}

function updateGuideResults() {
    const container = document.getElementById('guide-results');
    container.innerHTML = guideResults.map(r => {
        const qualityClass = r.quality.score >= 85 ? 'quality-good' : r.quality.score >= 70 ? 'quality-ok' : 'quality-bad';
        return `<div class="guide-result-item">
            <span>#${r.prompt_index + 1} ${escHtml(r.text).substring(0, 15)}...</span>
            <span>${r.quality.effective_duration}s/${r.quality.duration}s</span>
            <span class="${qualityClass}">${r.quality.quality_text} ${r.quality.score}</span>
            <button class="btn btn-sm btn-secondary" onclick="playGuideProcessed('${r.playback_url}')">播放</button>
        </div>`;
    }).join('');
}

function playGuideProcessed(url) {
    const audio = document.getElementById('guide-processed-audio');
    audio.src = url;
    audio.style.display = 'block';
    audio.play();
}

function skipGuideSegment() {
    if (confirm('确定跳过这段录音吗？')) {
        loadGuideSegment(guideCurrentIndex + 1);
    }
}

function finishGuideRegistration() {
    const count = guideSegments.length;
    if (count === 0) {
        alert('没有可提交的音频片段，请至少录制或上传一段');
        return;
    }

    const avgScore = guideResults.reduce((sum, r) => sum + r.quality.score, 0) / count;
    const qualityMsg = avgScore >= 85 ? '优秀' : avgScore >= 70 ? '良好' : avgScore >= 60 ? '可用' : '需补录';

    const userId = document.getElementById('guide-user').value;
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('segments', JSON.stringify(guideSegments.map(s => ({
        token: s.token,
        source: s.source,
        prompt_index: s.prompt_index,
        prompt_text: s.text,
    }))));

    // 显示提交状态
    const qualityEl = document.getElementById('guide-quality');
    qualityEl.textContent = '正在进行质量门控、一致性校验和声纹融合...';
    qualityEl.style.color = '#1a73e8';
    document.getElementById('guide-record-btn').disabled = true;
    document.getElementById('guide-upload-btn').disabled = true;
    document.getElementById('guide-finish-btn').disabled = true;

    fetch('/api/voiceprint/register-batch', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                alert('注册失败: ' + data.error);
            } else {
                document.getElementById('guide-area').style.display = 'none';
                alert(`声纹注册完成！\n\n本次提交: ${count}段\n成功融合: ${data.success}段\n累计融合: ${data.total_segments}段\n平均质量: ${qualityMsg}\n\n后续可继续选择该用户追加上传或录制，系统会继续融合。`);
                guideSegments = [];
                guideResults = [];
                loadUsers();
                checkModelStatus();
            }
        })
        .catch(err => alert('注册失败: ' + err))
        .finally(() => {
            document.getElementById('guide-record-btn').disabled = false;
            document.getElementById('guide-upload-btn').disabled = false;
            document.getElementById('guide-finish-btn').disabled = guideSegments.length === 0;
        });
}

// ========== 工具函数 ==========
function showStatus(elementId, message, type) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.className = 'status-msg ' + type;
}

function escHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ========== 系统重置 ==========
function showResetDialog() {
    document.getElementById('reset-modal').style.display = 'flex';
    document.getElementById('reset-user').value = '';
    document.getElementById('reset-pass').value = '';
    document.getElementById('reset-error').style.display = 'none';
}

function closeResetDialog() {
    document.getElementById('reset-modal').style.display = 'none';
}

function confirmReset() {
    const username = document.getElementById('reset-user').value.trim();
    const password = document.getElementById('reset-pass').value;
    const errEl = document.getElementById('reset-error');

    if (!username || !password) {
        errEl.textContent = '请输入账号和密码';
        errEl.style.display = 'block';
        return;
    }

    if (!confirm('确定要重置系统吗？所有用户、声纹和日志将被删除！')) return;

    fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            errEl.textContent = data.error;
            errEl.style.display = 'block';
        } else {
            closeResetDialog();
            alert(data.message);
            loadUsers();
            loadLogs();
        }
    })
    .catch(err => {
        errEl.textContent = '请求失败: ' + err;
        errEl.style.display = 'block';
    });
}
