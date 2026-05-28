"""
声纹注册引导对话框 - 分步朗读录音，覆盖全音素
用于高质量注册声纹
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import tempfile
from datetime import datetime
from config import SAMPLE_RATE


# ========== 朗读文本库 ==========
# 覆盖所有中文音素（元音、辅音、声调、语速）
# 分为不同类别，每段2-8秒，6-10段即可完整注册
REGISTRATION_TEXTS = [
    {
        "category": "元音覆盖",
        "text": "山丹丹花开红艳艳，嘀嗒嘀嗒嘀嗒",
        "tone": "平静、清晰",
    },
    {
        "category": "鼻音/边音",
        "text": "蓝蓝天上白云飘，白云下面马儿跑",
        "tone": "平静、清晰",
    },
    {
        "category": "声调覆盖",
        "text": "人工智能改变世界，科技创新引领未来",
        "tone": "缓慢、字正腔圆",
    },
    {
        "category": "唇齿音",
        "text": "粉红凤凰飞，红旗飘扬在风中",
        "tone": "正常语速",
    },
    {
        "category": "开口音",
        "text": "配电柜操作需谨慎，安全防护第一重要",
        "tone": "严肃、清晰",
    },
    {
        "category": "情感表达",
        "text": "您好，我是配电运维班班长张伟，现在为您演示声纹识别系统",
        "tone": "自然、亲切",
    },
    {
        "category": "数字与专业术语",
        "text": "电流电压正常，系统运行稳定，各项参数均在安全范围",
        "tone": "汇报语气",
    },
    {
        "category": "日常口语",
        "text": "喂，小张，今天的巡检工作准备好了吗",
        "tone": "自然对话语气",
    },
]

# 最短/最长录音时长（秒）
MIN_RECORD_SECONDS = 0.5
MAX_RECORD_SECONDS = 30


class VoiceprintRegistrationGuide:
    """声纹注册引导对话框"""

    def __init__(self, parent, db_manager, vp_database, user_id, user_name, on_success=None):
        self.parent = parent
        self.db = db_manager
        self.vp_db = vp_database
        self.user_id = user_id
        self.user_name = user_name
        self.on_success = on_success

        self._temp_audio_path = None
        self._audio_data = []
        self._stream = None
        self._is_recording = False
        self._timer_after_id = None
        self._record_start_time = None
        self._timer_start_time = None

        self._current_index = 0
        self._registration_results = []

        self._create_dialog()

    def _create_dialog(self):
        """创建注册引导对话框"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("声纹注册引导")
        self.dialog.geometry("700x520")
        self.dialog.resizable(True, True)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # 居中
        x = self.parent.winfo_x() + 50
        y = self.parent.winfo_y() + 50
        self.dialog.geometry(f"+{x}+{y}")

        # ========== 顶部区域：标题 ==========
        title_frame = ttk.Frame(self.dialog)
        title_frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        ttk.Label(
            title_frame,
            text="声纹注册引导",
            font=("微软雅黑", 16, "bold")
        ).pack(side=tk.LEFT)

        ttk.Label(
            title_frame,
            text=f"用户：{self.user_name}",
            font=("微软雅黑", 10),
            foreground="gray"
        ).pack(side=tk.RIGHT)

        # ========== 进度区域 ==========
        progress_frame = ttk.Frame(self.dialog)
        progress_frame.pack(fill=tk.X, padx=15, pady=5)

        self.progress_label = ttk.Label(
            progress_frame,
            text="0/8 完成",
            font=("微软雅黑", 10),
            foreground="#666"
        )
        self.progress_label.pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            length=650,
            maximum=len(REGISTRATION_TEXTS)
        )
        self.progress_bar.pack(fill=tk.X, pady=(3, 10))

        # ========== 主内容区域 ==========
        content_frame = ttk.Frame(self.dialog)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # 左侧：朗读文本区域
        left_frame = ttk.LabelFrame(content_frame, text="朗读文本", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 类别标签
        self.category_var = tk.StringVar(value="")
        ttk.Label(
            left_frame,
            textvariable=self.category_var,
            font=("微软雅黑", 9),
            foreground="#4CAF50"
        ).pack(anchor=tk.W)

        # 朗读文本（大字体）
        self.text_display = tk.Text(
            left_frame,
            height=4,
            width=50,
            font=("微软雅黑", 18),
            bg="#FFFDE7",
            fg="#333",
            relief=tk.SOLID,
            bd=1
        )
        self.text_display.pack(fill=tk.BOTH, expand=True, pady=(5, 5))

        # 语调提示
        self.tone_var = tk.StringVar(value="")
        ttk.Label(
            left_frame,
            textvariable=self.tone_var,
            font=("微软雅黑", 9),
            foreground="#999"
        ).pack(anchor=tk.W, pady=(0, 5))

        # 录音时长显示
        self.record_time_var = tk.StringVar(value="00:00")
        self.record_status_var = tk.StringVar(value="准备就绪")
        ttk.Label(
            left_frame,
            textvariable=self.record_status_var,
            font=("微软雅黑", 9),
            foreground="#666"
        ).pack(anchor=tk.W, pady=(0, 3))

        self.record_time_label = ttk.Label(
            left_frame,
            textvariable=self.record_time_var,
            font=("Consolas", 14),
            foreground="#333"
        )
        self.record_time_label.pack(anchor=tk.W, pady=(0, 10))

        # 质量指示器
        self.quality_var = tk.StringVar(value="待录音")
        quality_frame = ttk.Frame(left_frame)
        quality_frame.pack(fill=tk.X)
        ttk.Label(
            quality_frame,
            text="录音质量:",
            font=("微软雅黑", 9)
        ).pack(side=tk.LEFT)
        self.quality_label = ttk.Label(
            quality_frame,
            textvariable=self.quality_var,
            font=("微软雅黑", 9),
            foreground="#999"
        )
        self.quality_label.pack(side=tk.LEFT, padx=(10, 0))

        # 右侧：结果列表
        right_frame = ttk.LabelFrame(content_frame, text="注册进度", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))

        columns = ("#", "内容", "时长", "质量")
        self.result_tree = ttk.Treeview(
            right_frame,
            columns=columns,
            show="headings",
            height=12,
        )
        self.result_tree.heading("#", text="#")
        self.result_tree.heading("内容", text="内容")
        self.result_tree.heading("时长", text="时长")
        self.result_tree.heading("质量", text="质量")
        self.result_tree.column("#", width=30)
        self.result_tree.column("内容", width=150)
        self.result_tree.column("时长", width=50)
        self.result_tree.column("质量", width=50)

        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)

        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ========== 底部区域：按钮 ==========
        bottom_frame = ttk.Frame(self.dialog)
        bottom_frame.pack(fill=tk.X, padx=15, pady=(5, 10))

        # 录音控制：开始/停止 两个按钮
        self.record_start_btn = tk.Button(
            bottom_frame,
            text="▶ 开始录音",
            bg="#4CAF50",
            fg="white",
            font=("微软雅黑", 12, "bold"),
            cursor="hand2",
            width=14,
            height=2
        )
        self.record_start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.record_stop_btn = tk.Button(
            bottom_frame,
            text="⏹ 停止录音",
            bg="#f44336",
            fg="white",
            font=("微软雅黑", 12, "bold"),
            cursor="hand2",
            width=14,
            height=2
        )
        self.record_stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        # 绑定按钮事件
        self.record_start_btn.bind("<Button-1>", self._on_record_press)
        self.record_stop_btn.bind("<Button-1>", self._on_record_stop)

        # 跳过按钮
        self.skip_btn = ttk.Button(
            bottom_frame,
            text="跳过此段",
            command=self._skip_segment,
            width=12
        )
        self.skip_btn.pack(side=tk.LEFT, padx=(0, 5))

        # 取消按钮
        ttk.Button(
            bottom_frame,
            text="取消",
            command=self._cancel_registration,
            width=8
        ).pack(side=tk.RIGHT)

        # 完成按钮（初始隐藏）
        self.finish_btn = tk.Button(
            bottom_frame,
            text="完成注册",
            bg="#2196F3",
            fg="white",
            font=("微软雅黑", 11, "bold"),
            cursor="hand2",
            width=10,
            height=1
        )
        self.finish_btn.pack(side=tk.RIGHT, padx=(0, 10))
        self.finish_btn.pack_forget()

        # 加载第一段文本
        self._load_segment(0)

    def _set_recording_ui(self, recording):
        """更新录音按钮UI"""
        if recording:
            self.record_start_btn.config(state=tk.DISABLED, text="⏺ 录音中...", bg="#f44336")
            self.record_stop_btn.config(state=tk.NORMAL, bg="#e53935")
            self.record_status_var.set("正在录音...")
        else:
            self.record_start_btn.config(state=tk.DISABLED, text="▶ 开始录音", bg="#4CAF50")
            self.record_stop_btn.config(state=tk.DISABLED, bg="#f44336")
            self.record_status_var.set("准备就绪")

    def _load_segment(self, index):
        """加载当前朗读文本"""
        if index >= len(REGISTRATION_TEXTS):
            self._finish_registration()
            return

        text = REGISTRATION_TEXTS[index]
        self._current_index = index
        self.category_var.set(f"[{text['category']}]")
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, text["text"])
        self.text_display.config(state=tk.DISABLED)
        self.tone_var.set(f"语调提示：{text['tone']}")

        # 重置状态
        self.record_status_var.set("准备就绪")
        self.record_time_var.set("00:00")
        self.quality_var.set("待录音")

        # 重置录音按钮
        self.record_start_btn.config(state=tk.DISABLED, text="▶ 开始录音", bg="#4CAF50")
        self.record_stop_btn.config(state=tk.DISABLED, bg="#f44336")

        # 更新进度
        self.progress_label.config(text=f"{index}/{len(REGISTRATION_TEXTS)} 完成")
        self.progress_bar['value'] = index

        # 更新按钮状态
        if index == len(REGISTRATION_TEXTS) - 1:
            self.finish_btn.config(text="完成注册", state=tk.NORMAL)
            self.finish_btn.pack(side=tk.RIGHT, padx=(0, 5))
        else:
            self.finish_btn.pack_forget()

        # 高亮树视图当前行
        self.result_tree.selection_set("")

    def _on_record_press(self, event):
        """开始录音"""
        if self._is_recording:
            return

        self._is_recording = True
        self.record_start_btn.config(state=tk.DISABLED, text="⏺ 录音中...", bg="#f44336")
        self.record_stop_btn.config(state=tk.NORMAL, bg="#e53935")
        self.record_status_var.set("正在录音...")
        self.record_time_var.set("00:00")

        try:
            self._audio_data = []
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self._audio_callback,
                blocksize=1024
            )
            self._stream.start()

            # 更新计时器
            self._record_start_time = datetime.now()
            self._timer_start_time = self._record_start_time
            self._update_timer()

        except Exception as e:
            messagebox.showerror("错误", f"无法启动录音: {e}")
            self._is_recording = False
            self.record_start_btn.config(state=tk.DISABLED, text="▶ 开始录音", bg="#4CAF50")
            self.record_stop_btn.config(state=tk.DISABLED, bg="#f44336")
            self.record_status_var.set("准备就绪")

    def _on_record_stop(self, event):
        """停止录音"""
        if not self._is_recording:
            return

        self._is_recording = False

        # 停止录音流
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # 取消计时器更新
        if self._timer_after_id:
            self.dialog.after_cancel(self._timer_after_id)
            self._timer_after_id = None

        # 处理录音数据
        self._save_recording()

    def _audio_callback(self, indata, frames, time, status):
        """音频回调函数"""
        if status:
            print(f"录音状态: {status}")
        self._audio_data.append(indata.copy())

    def _update_timer(self):
        """更新录音计时器"""
        if self._is_recording:
            elapsed = (datetime.now() - self._timer_start_time).total_seconds()
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.record_time_var.set(f"{minutes:02d}:{seconds:02d}")

            if self._is_recording:
                self._timer_after_id = self.dialog.after(100, self._update_timer)

    def _save_recording(self):
        """保存录音并评估质量"""
        if not self._audio_data:
            messagebox.showwarning("警告", "录音内容为空，请重新录音")
            self._set_recording_ui(False)
            return

        # 合并录音数据
        audio_data = np.concatenate(self._audio_data, axis=0)
        duration = len(audio_data) / SAMPLE_RATE

        # 检查录音时长
        if duration < MIN_RECORD_SECONDS:
            messagebox.showwarning("警告", f"录音时间太短（{duration:.1f}秒），请重新录音")
            self._set_recording_ui(False)
            return

        if duration > MAX_RECORD_SECONDS:
            messagebox.showwarning("警告", f"录音时间过长（{duration:.1f}秒），已截取前{MAX_RECORD_SECONDS}秒")
            audio_data = audio_data[:int(MAX_RECORD_SECONDS * SAMPLE_RATE)]

        # 评估录音质量
        quality = self._evaluate_recording_quality(audio_data, duration)

        # 清理之前的临时文件
        if self._temp_audio_path and os.path.exists(self._temp_audio_path):
            try:
                os.remove(self._temp_audio_path)
            except:
                pass

        # 保存到临时文件
        self._temp_audio_path = os.path.join(
            tempfile.gettempdir(),
            f"vp_reg_{os.getpid()}_{self._current_index}.wav"
        )
        try:
            sf.write(self._temp_audio_path, audio_data, SAMPLE_RATE)
        except Exception as e:
            messagebox.showerror("错误", f"保存录音失败: {e}")
            self._set_recording_ui(False)
            return

        # 注册声纹
        self._register_voiceprint(audio_data, quality)

    def _evaluate_recording_quality(self, audio_data, duration):
        """评估录音质量"""
        # 音量评估
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 0.01:
            volume_score = 20  # 太安静
            volume_text = "太安静"
        elif rms < 0.05:
            volume_score = 60  # 偏安静
            volume_text = "偏安静"
        elif rms < 0.2:
            volume_score = 90  # 正常
            volume_text = "良好"
        elif rms < 0.4:
            volume_score = 80  # 偏响
            volume_text = "偏响"
        else:
            volume_score = 50  # 太响
            volume_text = "太响"

        # 时长评估
        if duration >= 2 and duration <= 8:
            duration_score = 95
        elif duration >= 1 and duration <= 12:
            duration_score = 70
        else:
            duration_score = 50

        # 总质量分
        total_score = int(volume_score * 0.6 + duration_score * 0.4)

        # 颜色标识
        if total_score >= 85:
            quality_text = "优秀"
            quality_color = "#4CAF50"
        elif total_score >= 70:
            quality_text = "良好"
            quality_color = "#FFC107"
        else:
            quality_text = "待改进"
            quality_color = "#f44336"

        return {
            "score": total_score,
            "volume": volume_text,
            "duration": duration,
            "quality_text": quality_text,
            "quality_color": quality_color,
        }

    def _register_voiceprint(self, audio_data, quality):
        """注册当前录音的声纹"""
        try:
            # 使用临时文件注册
            vp_id = self.vp_db.register_voiceprint(self.user_id, self._temp_audio_path)

            # 记录结果
            self._registration_results.append({
                "index": self._current_index + 1,
                "text": REGISTRATION_TEXTS[self._current_index]["text"],
                "quality": quality,
                "vp_id": vp_id,
            })

            # 更新树视图
            self.result_tree.insert("", tk.END, values=(
                self._current_index + 1,
                REGISTRATION_TEXTS[self._current_index]["text"][:15] + "...",
                f"{quality['duration']:.1f}s",
                quality["quality_text"]
            ))

            # 设置颜色
            item = self.result_tree.get_children()[-1]
            self.result_tree.item(item, tags=(quality["quality_color"],))
            self.result_tree.tag_configure("#4CAF50", foreground="#4CAF50")
            self.result_tree.tag_configure("#FFC107", foreground="#FFC107")
            self.result_tree.tag_configure("#f44336", foreground="#f44336")

            # 更新质量显示
            self.quality_var.set(quality["quality_text"])
            self.quality_label.config(foreground=quality["quality_color"])

            # 自动进入下一段（延迟800ms让用户看到结果）
            self.dialog.after(800, lambda: self._load_segment(self._current_index + 1))

        except Exception as e:
            messagebox.showerror("错误", f"声纹注册失败: {e}")
            self._set_recording_ui(False)

    def _skip_segment(self):
        """跳过当前段落"""
        if messagebox.askyesno("确认", "确定跳过这段录音吗？"):
            self._load_segment(self._current_index + 1)

    def _cancel_registration(self):
        """取消注册"""
        if messagebox.askyesno("确认", "确定要取消声纹注册吗？已录音的数据将被保留。"):
            self._cleanup_temp_files()
            self.dialog.destroy()

    def _finish_registration(self):
        """完成注册"""
        count = len(self._registration_results)
        if count == 0:
            messagebox.showwarning("警告", "没有录到任何有效数据，请至少录制一段")
            return

        avg_quality = sum(r["quality"]["score"] for r in self._registration_results) / count
        if avg_quality >= 75:
            quality_msg = "优秀"
        elif avg_quality >= 60:
            quality_msg = "良好"
        else:
            quality_msg = "一般"

        msg = (
            f"声纹注册完成！\n\n"
            f"录制段落: {count}/{len(REGISTRATION_TEXTS)}\n"
            f"平均质量: {quality_msg}\n\n"
            f"声纹已保存，后续识别将更加准确。"
        )

        messagebox.showinfo("成功", msg)
        self._cleanup_temp_files()
        self.dialog.destroy()
        if self.on_success:
            self.on_success()

    def _cleanup_temp_files(self):
        """清理临时文件"""
        if self._temp_audio_path and os.path.exists(self._temp_audio_path):
            try:
                os.remove(self._temp_audio_path)
            except:
                pass
