"""
用户管理界面 - 声纹识别系统
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
import threading
from config import AUDIO_DIR
from ui.registration_guide import VoiceprintRegistrationGuide


class RegisterFrame(ttk.Frame):
    """用户管理界面"""
    
    def __init__(self, parent, db_manager, vp_database, main_window):
        """
        初始化用户管理界面
        
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
            text="用户管理",
            style="Title.TLabel"
        ).pack(side=tk.LEFT)
        
        # 主内容区域
        content_frame = ttk.Frame(self)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧：用户添加区域
        left_frame = ttk.LabelFrame(content_frame, text="添加用户", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        
        self._create_add_user_section(left_frame)
        
        # 右侧：用户列表
        right_frame = ttk.LabelFrame(content_frame, text="用户列表", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self._create_user_list_section(right_frame)
    
    def _create_add_user_section(self, parent):
        """创建添加用户区域"""
        # 用户名
        ttk.Label(parent, text="用户名:").pack(anchor=tk.W, pady=(0, 5))
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(parent, textvariable=self.name_var, width=20)
        name_entry.pack(fill=tk.X, pady=(0, 10))
        
        # 部门
        ttk.Label(parent, text="部门:").pack(anchor=tk.W, pady=(0, 5))
        self.dept_var = tk.StringVar()
        dept_entry = ttk.Entry(parent, textvariable=self.dept_var, width=20)
        dept_entry.pack(fill=tk.X, pady=(0, 15))
        
        # 添加用户按钮
        ttk.Button(
            parent,
            text="添加用户",
            command=self._add_user
        ).pack(fill=tk.X, pady=(0, 10))
        
        # 分隔线
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 声纹注册区域
        ttk.Label(parent, text="声纹注册", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(parent, text="选择用户:").pack(anchor=tk.W, pady=(0, 5))
        self.user_combo = ttk.Combobox(parent, state="readonly", width=18)
        self.user_combo.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(parent, text="音频文件:").pack(anchor=tk.W, pady=(0, 5))
        
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var, width=15).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            file_frame,
            text="浏览",
            command=self._browse_audio,
            width=6
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 注册声纹按钮
        btn_frame_reg = ttk.Frame(parent)
        btn_frame_reg.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            btn_frame_reg,
            text="引导注册（推荐）",
            command=self._guide_register,
            width=12
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            btn_frame_reg,
            text="直接注册",
            command=self._register_voiceprint,
            width=8
        ).pack(side=tk.LEFT, padx=(0, 5))

        # 状态标签
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(parent, textvariable=self.status_var, style="Info.TLabel")
        self.status_label.pack(anchor=tk.W)
    
    def _create_user_list_section(self, parent):
        """创建用户列表区域"""
        # 用户列表
        columns = ("ID", "用户名", "部门", "声纹数", "创建时间")
        self.user_tree = ttk.Treeview(parent, columns=columns, show="headings", height=15)
        
        # 设置列宽
        self.user_tree.heading("ID", text="ID")
        self.user_tree.heading("用户名", text="用户名")
        self.user_tree.heading("部门", text="部门")
        self.user_tree.heading("声纹数", text="声纹数")
        self.user_tree.heading("创建时间", text="创建时间")
        
        self.user_tree.column("ID", width=40)
        self.user_tree.column("用户名", width=100)
        self.user_tree.column("部门", width=100)
        self.user_tree.column("声纹数", width=60)
        self.user_tree.column("创建时间", width=150)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.user_tree.yview)
        self.user_tree.configure(yscrollcommand=scrollbar.set)
        
        self.user_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右键菜单
        self.context_menu = tk.Menu(self.winfo_toplevel(), tearoff=0)
        self.context_menu.add_command(label="删除用户", command=self._delete_user)
        self.context_menu.add_command(label="删除声纹", command=self._delete_voiceprint)
        
        self.user_tree.bind("<Button-3>", self._show_context_menu)
        
        # 按钮区域
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            btn_frame,
            text="刷新",
            command=self.refresh_user_list
        ).pack(side=tk.LEFT)
    
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
    
    def _add_user(self):
        """添加用户"""
        name = self.name_var.get().strip()
        department = self.dept_var.get().strip()
        
        if not name:
            messagebox.showwarning("警告", "请输入用户名")
            return
        
        try:
            user_id = self.db.add_user(name, department)
            messagebox.showinfo("成功", f"用户添加成功\n用户ID: {user_id}")
            
            # 清空输入框
            self.name_var.set("")
            self.dept_var.set("")
            
            # 刷新列表
            self.refresh_user_list()
            self.main_window.set_status(f"用户 {name} 添加成功")
            
        except Exception as e:
            messagebox.showerror("错误", f"添加用户失败: {e}")
    
    def _register_voiceprint(self):
        """注册声纹"""
        # 获取选中的用户
        user_name = self.user_combo.get()
        if not user_name:
            messagebox.showwarning("警告", "请选择用户")
            return
        
        # 获取音频文件
        audio_path = self.file_var.get().strip()
        if not audio_path:
            messagebox.showwarning("警告", "请选择音频文件")
            return
        
        if not os.path.exists(audio_path):
            messagebox.showerror("错误", "音频文件不存在")
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
        
        # 在新线程中执行注册
        self.status_var.set("正在注册声纹...")
        self.main_window.set_status("正在注册声纹...")
        
        def register_thread():
            try:
                # 复制音频文件到数据目录
                filename = os.path.basename(audio_path)
                dest_path = os.path.join(AUDIO_DIR, f"{user_id}_{filename}")
                shutil.copy2(audio_path, dest_path)
                
                # 注册声纹
                vp_id = self.vp_db.register_voiceprint(user_id, dest_path)
                
                self.winfo_toplevel().after(0, lambda: self._register_success(user_name, vp_id))
                
            except Exception as e:
                error_msg = str(e)
                self.winfo_toplevel().after(0, lambda msg=error_msg: self._register_error(msg))
        
        threading.Thread(target=register_thread, daemon=True).start()
    
    def _register_success(self, user_name, vp_id):
        """注册成功回调"""
        self.status_var.set(f"声纹注册成功")
        self.main_window.set_status(f"用户 {user_name} 声纹注册成功")
        messagebox.showinfo("成功", f"声纹注册成功\n声纹ID: {vp_id}")
        
        # 清空文件选择
        self.file_var.set("")
        
        # 刷新列表
        self.refresh_user_list()
    
    def _register_error(self, error_msg):
        """注册失败回调"""
        self.status_var.set(f"注册失败: {error_msg}")
        self.main_window.set_status("声纹注册失败")
        messagebox.showerror("错误", f"声纹注册失败: {error_msg}")

    def _guide_register(self):
        """引导注册"""
        user_name = self.user_combo.get()
        if not user_name:
            messagebox.showwarning("警告", "请先选择用户")
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

        self.status_var.set("打开注册引导...")
        self.main_window.set_status("正在打开注册引导...")

        # 延迟创建对话框，避免阻塞
        self.winfo_toplevel().after(100, lambda: self._open_guide_dialog(user_id, user_name))

    def _open_guide_dialog(self, user_id, user_name):
        """打开引导注册对话框"""
        def on_success():
            self.winfo_toplevel().after(0, lambda: self._refresh_after_registration())

        guide = VoiceprintRegistrationGuide(
            parent=self.winfo_toplevel(),
            db_manager=self.db,
            vp_database=self.vp_db,
            user_id=user_id,
            user_name=user_name,
            on_success=on_success
        )

    def _refresh_after_registration(self):
        """注册完成后刷新"""
        self.status_var.set("注册完成")
        self.main_window.set_status("声纹注册完成")
        self.file_var.set("")
        self.refresh_user_list()
    
    def refresh_user_list(self):
        """刷新用户列表"""
        # 清空列表
        for item in self.user_tree.get_children():
            self.user_tree.delete(item)
        
        # 获取所有用户
        users = self.db.get_all_users()
        
        # 更新列表
        for user in users:
            # 获取用户的声纹数量
            voiceprints = self.db.get_voiceprints(user['id'])
            vp_count = len(voiceprints)
            
            self.user_tree.insert("", tk.END, values=(
                user['id'],
                user['name'],
                user['department'] or "",
                vp_count,
                user['created_at']
            ))
        
        # 更新下拉框
        user_names = [user['name'] for user in users]
        self.user_combo['values'] = user_names
        if user_names:
            self.user_combo.current(0)
    
    def _show_context_menu(self, event):
        """显示右键菜单"""
        item = self.user_tree.identify_row(event.y)
        if item:
            self.user_tree.selection_set(item)
            self.context_menu.post(event.x, event.y)
    
    def _delete_user(self):
        """删除用户"""
        selection = self.user_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要删除的用户")
            return
        
        item = selection[0]
        user_id = self.user_tree.item(item)['values'][0]
        user_name = self.user_tree.item(item)['values'][1]
        
        if messagebox.askyesno("确认", f"确定要删除用户 {user_name} 及其所有声纹吗？"):
            try:
                self.db.delete_user(user_id)
                messagebox.showinfo("成功", f"用户 {user_name} 已删除")
                self.refresh_user_list()
                self.main_window.set_status(f"用户 {user_name} 已删除")
            except Exception as e:
                messagebox.showerror("错误", f"删除用户失败: {e}")
    
    def _delete_voiceprint(self):
        """删除声纹"""
        selection = self.user_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择用户")
            return
        
        item = selection[0]
        user_id = self.user_tree.item(item)['values'][0]
        user_name = self.user_tree.item(item)['values'][1]
        
        # 获取用户的声纹
        voiceprints = self.db.get_voiceprints(user_id)
        if not voiceprints:
            messagebox.showinfo("提示", f"用户 {user_name} 没有声纹记录")
            return
        
        # 创建选择对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("删除声纹")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"用户 {user_name} 的声纹:").pack(pady=10)
        
        listbox = tk.Listbox(dialog, height=5)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for vp in voiceprints:
            listbox.insert(tk.END, f"ID: {vp['id']} - {vp['created_at']}")
        
        def delete_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("警告", "请选择要删除的声纹")
                return
            
            vp_id = voiceprints[selection[0]]['id']
            try:
                self.db.delete_voiceprint(vp_id)
                messagebox.showinfo("成功", "声纹已删除")
                dialog.destroy()
                self.refresh_user_list()
            except Exception as e:
                messagebox.showerror("错误", f"删除声纹失败: {e}")
        
        ttk.Button(dialog, text="删除选中", command=delete_selected).pack(pady=10)
