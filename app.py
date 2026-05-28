"""
主程序入口 - 配电柜智能语音识别系统
"""
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import threading

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SIMILARITY_THRESHOLD
from database import DatabaseManager


class VoiceprintApp:
    """声纹识别应用"""
    
    def __init__(self):
        """初始化应用"""
        # 创建根窗口
        self.root = tk.Tk()
        self.root.title("配电柜智能语音识别系统")
        self.root.geometry("900x650")
        
        # 居中显示
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 900) // 2
        y = (screen_height - 650) // 2
        self.root.geometry(f"900x650+{x}+{y}")
        
        # 初始化组件
        self.db_manager = DatabaseManager()
        self.extractor = None
        self.vp_database = None
        self.main_window = None
        
        # 显示加载界面
        self._show_loading()
        
        # 在后台线程加载模型
        threading.Thread(target=self._load_model, daemon=True).start()
    
    def _show_loading(self):
        """显示加载界面"""
        self.loading_frame = tk.Frame(self.root)
        self.loading_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        tk.Label(
            self.loading_frame,
            text="配电柜智能语音识别系统",
            font=("微软雅黑", 20, "bold")
        ).pack(pady=(100, 20))
        
        # 加载提示
        self.loading_label = tk.Label(
            self.loading_frame,
            text="正在加载声纹识别模型，请稍候...",
            font=("微软雅黑", 12)
        )
        self.loading_label.pack(pady=(0, 30))
        
        # 进度条
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar", troughcolor='#E0E0E0', background='#4CAF50')
        
        self.progress = ttk.Progressbar(
            self.loading_frame,
            mode='indeterminate',
            length=400,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress.pack()
        self.progress.start(15)
    
    def _load_model(self):
        """在后台线程加载模型"""
        try:
            from voiceprint import VoiceprintExtractor, VoiceprintDatabase
            
            # 更新状态
            self.root.after(0, lambda: self.loading_label.config(text="正在加载声纹识别模型..."))
            
            # 加载模型
            self.extractor = VoiceprintExtractor()
            
            # 初始化声纹数据库
            self.vp_database = VoiceprintDatabase(self.db_manager, self.extractor)
            
            # 模型加载完成，切换到主线程创建UI
            self.root.after(0, self._on_model_loaded)
            
        except Exception as e:
            self.root.after(0, lambda: self._on_model_error(str(e)))
    
    def _on_model_loaded(self):
        """模型加载完成回调"""
        # 停止进度条
        self.progress.stop()
        
        # 销毁加载界面
        self.loading_frame.destroy()
        
        # 创建主窗口
        from ui.main_window import MainWindow
        self.main_window = MainWindow(
            self.root,
            self.db_manager,
            self.vp_database
        )
        
        # 设置关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _on_model_error(self, error_msg):
        """模型加载错误回调"""
        self.progress.stop()
        self.loading_frame.destroy()
        
        # 显示错误信息
        error_frame = tk.Frame(self.root)
        error_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            error_frame,
            text="系统初始化失败",
            font=("微软雅黑", 16, "bold"),
            fg="red"
        ).pack(pady=(100, 20))
        
        tk.Label(
            error_frame,
            text=f"错误信息: {error_msg}",
            font=("微软雅黑", 10),
            wraplength=600
        ).pack(pady=(0, 20))
        
        tk.Button(
            error_frame,
            text="退出",
            command=self.root.destroy,
            font=("微软雅黑", 11)
        ).pack()
    
    def _on_closing(self):
        """关闭事件处理"""
        if messagebox.askyesno("退出", "确定要退出系统吗？"):
            self.root.destroy()
    
    def run(self):
        """运行应用"""
        self.root.mainloop()


def main():
    """主函数"""
    app = VoiceprintApp()
    app.run()


if __name__ == "__main__":
    main()
