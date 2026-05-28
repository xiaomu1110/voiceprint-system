"""
身份验证界面 - 声纹识别系统
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from config import SAMPLE_RATE


class VerifyFrame(ttk.Frame):
    """身份验证界面"""
    
    def __init__(self, parent, db_manager, vp_database, main_window):
        """
        初始化身份验证界面
        
        Args:
            parent: 父窗口
            db_manager: 数据库管理器
            vp_database: 声纹数据库
            main_window: 主窗口
        """
        super().__init__(parent)
        self.db = db_manager
        self.vp_db = vp_database
        self.main_window = main_window
        
        self._create_widgets()
        self.refresh_user_list()
    
    def _create_widgets(self):
        """创建界面组件"""
        # 顶部标题
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            title_frame,
            text="身份验证",
            style="Title.TLabel"
        ).pack(side=tk.LEFT)
        
        # 主内容区域
        content_frame = ttk.Frame(self)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧：验证区域
        left_frame = ttk.LabelFrame(content_frame, text="语音验证", padding=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        
        self._create_verify_section(left_frame)
        
        # 右侧：验证结果
        right_frame = ttk.LabelFrame(content_frame, text="验证结果", padding=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self._create_result_section(right_frame)
    
    def _create_verify_section(self, parent):
        """创建验证区域"""
        # 验证模式选择
        ttk.Label(parent, text="验证模式:", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        self.verify_mode = tk.StringVar(value="identify")
        
        ttk.Radiobutton(
            parent,
            text="身份识别（识别是谁）",
            variable=self.verify_mode,
            value="identify",
            command=self._on_mode_change
        ).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Radiobutton(
            parent,
            text="身份验证（验证是否是指定用户）",
            variable=self.verify_mode,
            value="verify",
            command=self._on_mode_change
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # 用户选择（仅在身份验证模式下显示）
        self.user_frame = ttk.Frame(parent)
        self.user_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.user_frame, text="选择用户:").pack(anchor=tk.W, pady=(0, 5))
        self.user_combo = ttk.Combobox(self.user_frame, state="readonly", width=18)
        self.user_combo.pack(fill=tk.X)
        
        # 音频文件选择
        ttk.Label(parent, text="音频文件:", style="Header.TLabel").pack(anchor=tk.W, pady=(10, 5))

        file_frame = ttk.Frame(parent)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var, width=12).pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Button(
            btn_frame,
            text="浏览",
            command=self._browse_audio,
            width=5
        ).pack(side=tk.LEFT, padx=(0, 3))

        # 录音按钮
        self.record_btn = ttk.Button(
            btn_frame,
            text="🎤录音",
            command=self._start_recording,
            width=6
        )
        self.record_btn.pack(side=tk.LEFT)

        self._temp_audio_path = None
        
        # 相似度阈值设置
        ttk.Label(parent, text="相似度阈值:", style="Header.TLabel").pack(anchor=tk.W, pady=(10, 5))
        
        threshold_frame = ttk.Frame(parent)
        threshold_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.threshold_var = tk.DoubleVar(value=0.75)
        self.threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0.5,
            to=0.95,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            command=self._on_threshold_change
        )
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.threshold_label = ttk.Label(threshold_frame, text="0.75")
        self.threshold_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # 开始验证按钮
        ttk.Button(
            parent,
            text="开始验证",
            command=self._start_verify
        ).pack(fill=tk.X, pady=(0, 10))
        
        # 状态标签
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(parent, textvariable=self.status_var, style="Info.TLabel")
        self.status_label.pack(anchor=tk.W)
        
        # 初始状态下隐藏用户选择
        self._on_mode_change()
    
    def _create_result_section(self, parent):
        """创建结果区域"""
        # 结果图标
        self.result_icon_var = tk.StringVar(value="")
        self.result_icon_label = ttk.Label(
            parent,
            textvariable=self.result_icon_var,
            font=("微软雅黑", 48)
        )
        self.result_icon_label.pack(pady=(20, 10))
        
        # 结果文本
        self.result_var = tk.StringVar(value="等待验证...")
        self.result_label = ttk.Label(
            parent,
            textvariable=self.result_var,
            style="Header.TLabel"
        )
        self.result_label.pack(pady=(0, 10))
        
        # 详细信息
        self.detail_text = tk.Text(parent, height=10, width=40, state=tk.DISABLED)
        self.detail_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 清除按钮
        ttk.Button(
            parent,
            text="清除结果",
            command=self._clear_result
        ).pack(fill=tk.X)
    
    def _on_mode_change(self):
        """验证模式改变"""
        if self.verify_mode.get() == "verify":
            self.user_frame.pack(fill=tk.X, pady=(0, 10))
        else:
            self.user_frame.pack_forget()
    
    def _on_threshold_change(self, value):
        """阈值改变"""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
    
    def _browse_audio(self):
        """浏览音频文件"""
        filetypes = [
            ("音频文件", "*.wav *.mp3 *.flac *.ogg"),
            ("WAV文件", "*.wav"),
            ("MP3文件", "*.mp3"),
            ("所有文件", "*.*")
        ]

        filepath = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=filetypes
        )

        if filepath:
            self.file_var.set(filepath)

    def _start_recording(self):
        """开始录音"""
        if hasattr(self, '_is_recording') and self._is_recording:
            self._stop_recording()
            return
        self._create_recording_dialog()

    def _stop_recording(self):
        """停止录音（兼容旧方法名，实际不再使用）"""
        pass

    def _create_recording_dialog(self):
        """创建录音对话框"""
        # 检查是否有麦克风
        devices = sd.query_devices()
        input_devices = [
            i for i, dev in enumerate(devices)
            if dev['max_input_channels'] > 0
        ]

        if not input_devices:
            messagebox.showwarning("警告", "未检测到麦克风设备")
            return

        # 创建对话框
        self.record_dialog = tk.Toplevel(self)
        self.record_dialog.title("语音识别")
        self.record_dialog.geometry("400x200")
        self.record_dialog.resizable(False, False)
        self.record_dialog.transient(self.winfo_toplevel())
        self.record_dialog.grab_set()

        # 居中显示
        x = self.record_dialog.winfo_x()
        y = self.record_dialog.winfo_y()
        self.record_dialog.geometry(f"+{x}+{y}")

        # 标题
        ttk.Label(
            self.record_dialog,
            text="按住说话",
            style="Header.TLabel"
        ).pack(pady=(15, 10))

        # 录音时间
        self.record_time_var = tk.StringVar(value="00:00")
        ttk.Label(
            self.record_dialog,
            textvariable=self.record_time_var,
            font=("Consolas", 24)
        ).pack(pady=(0, 15))

        # 录音按钮 - 按住说话
        self.record_main_btn = tk.Button(
            self.record_dialog,
            text="按住说话",
            bg="#4CAF50",
            fg="white",
            font=("微软雅黑", 14, "bold"),
            relief=tk.RAISED,
            cursor="hand2",
            width=20,
            height=2
        )
        self.record_main_btn.pack(pady=(0, 15))
        self.record_main_btn.bind("<ButtonPress-1>", self._on_record_press)
        self.record_main_btn.bind("<ButtonRelease-1>", self._on_record_release)
        self.record_main_btn.bind("<Enter>", lambda e: self.record_main_btn.config(bg="#66BB6A"))
        self.record_main_btn.bind("<Leave>", lambda e: self.record_main_btn.config(bg="#4CAF50"))

        # 取消按钮
        ttk.Button(
            self.record_dialog,
            text="取消",
            command=self._cancel_recording
        ).pack(pady=(0, 10))

        # 状态
        self._is_recording = False
        self._audio_data = []
        self._stream = None

    def _on_record_press(self, event):
        """录音按钮按下 - 开始录音"""
        if self._is_recording:
            return

        self._is_recording = True
        self.record_main_btn.config(text="正在录音...", bg="#f44336")
        self.record_main_btn.config(relief=tk.SUNKEN)

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
            from datetime import datetime
            self._record_start_time = datetime.now()
            self._timer_start_time = self._record_start_time
            self._update_timer()

        except Exception as e:
            messagebox.showerror("错误", f"无法启动录音: {e}")
            self._is_recording = False
            self.record_main_btn.config(text="按住说话", bg="#4CAF50")
            self.record_main_btn.config(relief=tk.RAISED)

    def _on_record_release(self, event):
        """录音按钮释放 - 停止录音"""
        if not self._is_recording:
            return

        self._is_recording = False

        # 停止录音流
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # 取消计时器更新
        if hasattr(self, '_timer_after_id') and self._timer_after_id:
            self.record_dialog.after_cancel(self._timer_after_id)

        # 处理录音数据
        self._save_recording()

    def _audio_callback(self, indata, frames, time, status):
        """音频回调函数"""
        if status:
            print(f"录音状态: {status}")
        self._audio_data.append(indata.copy())

    def _update_timer(self):
        """更新录音计时器"""
        if self._is_recording and hasattr(self, 'record_dialog'):
            elapsed = (self._record_start_time - self._timer_start_time).total_seconds()
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.record_time_var.set(f"{minutes:02d}:{seconds:02d}")

            if self._is_recording:
                self._timer_after_id = self.record_dialog.after(100, self._update_timer)

    def _save_recording(self):
        """保存录音到临时文件"""
        if not self._audio_data:
            messagebox.showwarning("警告", "录音内容为空")
            self._close_recording_dialog()
            return

        # 合并录音数据
        audio_data = np.concatenate(self._audio_data, axis=0)

        # 检查录音时长
        duration = len(audio_data) / SAMPLE_RATE
        if duration < 0.5:
            messagebox.showwarning("警告", "录音时间太短，请重新录音")
            self._close_recording_dialog()
            return

        if duration > 30:
            messagebox.showwarning("警告", "录音时间过长，最长30秒")
            self._close_recording_dialog()
            return

        # 清理之前的临时文件
        if self._temp_audio_path and os.path.exists(self._temp_audio_path):
            try:
                os.remove(self._temp_audio_path)
            except:
                pass

        # 保存到临时文件
        self._temp_audio_path = os.path.join(tempfile.gettempdir(), f"voiceprint_rec_{os.getpid()}.wav")
        try:
            sf.write(self._temp_audio_path, audio_data, SAMPLE_RATE)
            self.file_var.set(self._temp_audio_path)
            messagebox.showinfo("提示", f"录音完成\n时长: {duration:.1f}秒")
        except Exception as e:
            messagebox.showerror("错误", f"保存录音失败: {e}")
            return

        self._close_recording_dialog()

    def _cancel_recording(self):
        """取消录音"""
        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if hasattr(self, '_timer_after_id') and self._timer_after_id:
            self.record_dialog.after_cancel(self._timer_after_id)
        self._close_recording_dialog()

    def _close_recording_dialog(self):
        """关闭录音对话框"""
        if hasattr(self, 'record_dialog'):
            self.record_dialog.destroy()
            self.record_dialog = None

    def _clear_result(self):
        """清除结果"""
        self.result_icon_var.set("")
        self.result_var.set("等待验证...")

        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.config(state=tk.DISABLED)

        self.status_var.set("")

        # 清理临时录音文件
        if self._temp_audio_path and os.path.exists(self._temp_audio_path):
            try:
                os.remove(self._temp_audio_path)
            except:
                pass
            self._temp_audio_path = None
    
    def _start_verify(self):
        """开始验证"""
        # 获取音频文件
        audio_path = self.file_var.get().strip()
        if not audio_path:
            messagebox.showwarning("警告", "请选择音频文件")
            return
        
        if not os.path.exists(audio_path):
            messagebox.showerror("错误", "音频文件不存在")
            return
        
        # 获取阈值
        threshold = self.threshold_var.get()
        
        # 检查是否有声纹数据
        all_voiceprints = self.db.get_all_voiceprints()
        if not all_voiceprints:
            messagebox.showwarning("警告", "声纹数据库为空，请先注册声纹")
            return
        
        # 根据验证模式执行验证
        mode = self.verify_mode.get()
        
        if mode == "verify":
            # 身份验证模式
            user_name = self.user_combo.get()
            if not user_name:
                messagebox.showwarning("警告", "请选择要验证的用户")
                return
            
            # 获取用户ID
            users = self.db.get_all_users()
            user_id = None
            for user in users:
                if user['name'] == user_name:
                    user_id = user['id']
                    break
            
            if not user_id:
                messagebox.showerror("错误", "未找到用户")
                return
            
            self._verify_user(audio_path, user_id, user_name, threshold)
        
        else:
            # 身份识别模式
            self._identify_speaker(audio_path, threshold)
    
    def _verify_user(self, audio_path, user_id, user_name, threshold):
        """验证是否是指定用户"""
        self.status_var.set("正在验证...")
        self.main_window.set_status("正在验证身份...")
        
        def verify_thread():
            try:
                result = self.vp_db.verify_speaker_by_id(
                    audio_path, user_id, threshold
                )
                self.winfo_toplevel().after(0, lambda r=result, u=user_name: self._show_verify_result(r, u))
            except Exception as e:
                error_msg = str(e)
                self.winfo_toplevel().after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=verify_thread, daemon=True).start()
    
    def _identify_speaker(self, audio_path, threshold):
        """识别说话人身份"""
        self.status_var.set("正在识别...")
        self.main_window.set_status("正在识别身份...")
        
        def identify_thread():
            try:
                result = self.vp_db.verify_speaker(audio_path, threshold)
                self.winfo_toplevel().after(0, lambda r=result: self._show_identify_result(r))
            except Exception as e:
                error_msg = str(e)
                self.winfo_toplevel().after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=identify_thread, daemon=True).start()
    
    def _show_verify_result(self, result, user_name):
        """显示验证结果"""
        self.status_var.set("验证完成")
        self.main_window.set_status("身份验证完成")
        
        # 更新结果图标和文本
        if result['matched']:
            self.result_icon_var.set("✓")
            self.result_icon_label.configure(foreground="green")
            self.result_var.set(f"验证通过: {user_name}")
        else:
            self.result_icon_var.set("✗")
            self.result_icon_label.configure(foreground="red")
            self.result_var.set(f"验证失败: 不是 {user_name}")
        
        # 显示详细信息
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        
        detail = f"验证模式: 身份验证\n"
        detail += f"目标用户: {user_name}\n"
        detail += f"相似度阈值: {self.threshold_var.get():.2f}\n"
        detail += f"实际相似度: {result['similarity']:.4f}\n"
        detail += f"验证结果: {'通过' if result['matched'] else '失败'}\n"
        
        self.detail_text.insert(tk.END, detail)
        self.detail_text.config(state=tk.DISABLED)
    
    def _show_identify_result(self, result):
        """显示识别结果"""
        self.status_var.set("识别完成")
        self.main_window.set_status("身份识别完成")
        
        # 更新结果图标和文本
        if result['matched']:
            self.result_icon_var.set("✓")
            self.result_icon_label.configure(foreground="green")
            self.result_var.set(f"识别成功: {result['user_name']}")
        else:
            self.result_icon_var.set("✗")
            self.result_icon_label.configure(foreground="red")
            self.result_var.set("识别失败: 未找到匹配身份")
        
        # 显示详细信息
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        
        detail = f"验证模式: 身份识别\n"
        detail += f"相似度阈值: {self.threshold_var.get():.2f}\n"
        detail += f"最高相似度: {result['similarity']:.4f}\n"
        
        if result['matched']:
            detail += f"匹配用户: {result['user_name']}\n"
            detail += f"用户部门: {result.get('department', '')}\n"
        
        detail += f"\n验证结果: {'成功' if result['matched'] else '失败'}\n"
        
        # 显示所有匹配分数
        if result.get('all_scores'):
            detail += f"\n--- 所有用户相似度 ---\n"
            for user_id, score in result['all_scores'][:5]:  # 只显示前5个
                user = self.db.get_user(user_id)
                if user:
                    detail += f"{user['name']}: {score:.4f}\n"
        
        self.detail_text.insert(tk.END, detail)
        self.detail_text.config(state=tk.DISABLED)
    
    def _show_error(self, error_msg):
        """显示错误"""
        self.status_var.set("验证失败")
        self.main_window.set_status("验证失败")
        
        self.result_icon_var.set("✗")
        self.result_icon_label.configure(foreground="red")
        self.result_var.set("验证失败")
        
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.insert(tk.END, f"错误: {error_msg}")
        self.detail_text.config(state=tk.DISABLED)
        
        messagebox.showerror("错误", f"验证失败: {error_msg}")
    
    def _clear_result(self):
        """清除结果"""
        self.result_icon_var.set("")
        self.result_var.set("等待验证...")
        
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.config(state=tk.DISABLED)
        
        self.status_var.set("")
    
    def refresh_user_list(self):
        """刷新用户列表"""
        users = self.db.get_all_users()
        user_names = [user['name'] for user in users]
        self.user_combo['values'] = user_names
        if user_names:
            self.user_combo.current(0)
