"""
主窗口 - 声纹识别系统
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading


class MainWindow:
    """主窗口"""
    
    def __init__(self, root, db_manager, vp_database):
        """
        初始化主窗口
        
        Args:
            root: tkinter根窗口
            db_manager: 数据库管理器
            vp_database: 声纹数据库
        """
        self.root = root
        self.db = db_manager
        self.vp_db = vp_database
        
        self._setup_window()
        self._create_menu()
        self._create_notebook()
        self._create_status_bar()
    
    def _setup_window(self):
        """设置窗口属性"""
        self.root.title("配电柜智能语音识别系统")
        self.root.geometry("900x650")
        self.root.minsize(800, 600)
        
        # 设置样式
        style = ttk.Style()
        style.theme_use("clam")
        
        # 自定义样式
        style.configure("Title.TLabel", font=("微软雅黑", 16, "bold"))
        style.configure("Header.TLabel", font=("微软雅黑", 12, "bold"))
        style.configure("Info.TLabel", font=("微软雅黑", 10))
        style.configure("Success.TLabel", foreground="green", font=("微软雅黑", 10))
        style.configure("Error.TLabel", foreground="red", font=("微软雅黑", 10))
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self._show_about)
    
    def _create_notebook(self):
        """创建选项卡"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 导入并创建各个页面
        from ui.register_frame import RegisterFrame
        from ui.verify_frame import VerifyFrame
        from ui.log_frame import LogFrame
        
        # 用户管理页面
        self.register_frame = RegisterFrame(
            self.notebook, self.db, self.vp_db, self
        )
        self.notebook.add(self.register_frame, text="用户管理")
        
        # 身份验证页面
        self.verify_frame = VerifyFrame(
            self.notebook, self.db, self.vp_db, self
        )
        self.notebook.add(self.verify_frame, text="身份验证")
        
        # 日志查看页面
        self.log_frame = LogFrame(
            self.notebook, self.db, self
        )
        self.notebook.add(self.log_frame, text="验证日志")
    
    def _create_status_bar(self):
        """创建状态栏"""
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def set_status(self, message):
        """设置状态栏信息"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def _show_about(self):
        """显示关于对话框"""
        messagebox.showinfo(
            "关于",
            "配电柜智能语音识别系统 v1.0\n\n"
            "基于ECAPA-TDNN声纹识别模型\n"
            "用于配电柜操作人员身份认证\n\n"
            "技术栈：\n"
            "- 声纹识别：SpeechBrain ECAPA-TDNN\n"
            "- 前端界面：tkinter\n"
            "- 后端：Python"
        )
    
    def refresh_all(self):
        """刷新所有页面"""
        self.register_frame.refresh_user_list()
        self.verify_frame.refresh_user_list()
        self.log_frame.refresh_logs()
