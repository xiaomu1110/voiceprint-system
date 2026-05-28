"""
日志查看界面 - 声纹识别系统
"""
import tkinter as tk
from tkinter import ttk, messagebox


class LogFrame(ttk.Frame):
    """日志查看界面"""
    
    def __init__(self, parent, db_manager, main_window):
        """
        初始化日志查看界面
        
        Args:
            parent: 父窗口
            db_manager: 数据库管理器
            main_window: 主窗口
        """
        super().__init__(parent)
        self.db = db_manager
        self.main_window = main_window
        
        self._create_widgets()
        self.refresh_logs()
    
    def _create_widgets(self):
        """创建界面组件"""
        # 顶部标题和按钮
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            title_frame,
            text="验证日志",
            style="Title.TLabel"
        ).pack(side=tk.LEFT)
        
        # 按钮区域
        btn_frame = ttk.Frame(title_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(
            btn_frame,
            text="刷新",
            command=self.refresh_logs
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            btn_frame,
            text="清空日志",
            command=self._clear_logs
        ).pack(side=tk.LEFT)
        
        # 日志列表
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # 创建Treeview
        columns = ("ID", "用户", "相似度", "结果", "时间", "音频路径")
        self.log_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        # 设置列宽
        self.log_tree.heading("ID", text="ID")
        self.log_tree.heading("用户", text="用户")
        self.log_tree.heading("相似度", text="相似度")
        self.log_tree.heading("结果", text="结果")
        self.log_tree.heading("时间", text="时间")
        self.log_tree.heading("音频路径", text="音频路径")
        
        self.log_tree.column("ID", width=40)
        self.log_tree.column("用户", width=100)
        self.log_tree.column("相似度", width=80)
        self.log_tree.column("结果", width=80)
        self.log_tree.column("时间", width=150)
        self.log_tree.column("音频路径", width=200)
        
        # 滚动条
        scrollbar_y = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.log_tree.yview)
        scrollbar_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.log_tree.xview)
        self.log_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # 布局
        self.log_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # 统计信息
        stats_frame = ttk.Frame(self)
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.stats_var = tk.StringVar()
        ttk.Label(stats_frame, textvariable=self.stats_var).pack(side=tk.LEFT)
    
    def refresh_logs(self):
        """刷新日志列表"""
        # 清空列表
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)
        
        # 获取日志
        logs = self.db.get_verify_logs(limit=1000)
        
        # 统计
        total = len(logs)
        success = sum(1 for log in logs if log['result'] == '匹配')
        fail = total - success
        
        # 更新列表
        for log in logs:
            # 格式化相似度
            similarity = f"{log['similarity']:.4f}" if log['similarity'] else "N/A"
            
            # 截断音频路径
            audio_path = log.get('audio_path', '')
            if audio_path and len(audio_path) > 30:
                audio_path = "..." + audio_path[-27:]
            
            self.log_tree.insert("", tk.END, values=(
                log['id'],
                log.get('name', '未知'),
                similarity,
                log['result'],
                log['created_at'],
                audio_path
            ))
        
        # 更新统计信息
        self.stats_var.set(f"共 {total} 条记录 | 成功: {success} | 失败: {fail}")
    
    def _clear_logs(self):
        """清空日志"""
        if messagebox.askyesno("确认", "确定要清空所有验证日志吗？"):
            try:
                self.db.clear_verify_logs()
                self.refresh_logs()
                self.main_window.set_status("日志已清空")
            except Exception as e:
                messagebox.showerror("错误", f"清空日志失败: {e}")
