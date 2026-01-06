import tkinter as tk
from tkinter import ttk
import requests
import time
import threading

# 根據你的設定，預設是這個地址
SERVER_URL = "http://127.0.0.1:5000"

class RobotMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hil-Serl Robot Monitor")
        self.root.geometry("400x350")
        self.root.configure(bg="#f0f0f0")

        # 標題
        title_label = tk.Label(root, text="Robot State Monitor", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        # 建立數據顯示區
        self.frame = tk.Frame(root, bg="#f0f0f0")
        self.frame.pack(padx=20, pady=5)

        self.labels = {}
        # 定義我們要監控的變數
        self.vars_to_monitor = [
            ("X", "0.000"), ("Y", "0.000"), ("Z", "0.000"),
            ("RX (qx)", "0.000"), ("RY (qy)", "0.000"), ("RZ (qz)", "0.000"), ("RW (qw)", "0.000"),
            ("Gripper", "0.000")
        ]

        for i, (name, init_val) in enumerate(self.vars_to_monitor):
            row = tk.Frame(self.frame, bg="#f0f0f0")
            row.pack(fill="x", pady=2)
            
            lbl_name = tk.Label(row, text=f"{name}:", width=10, anchor="w", font=("Consolas", 12), bg="#f0f0f0")
            lbl_name.pack(side="left")
            
            lbl_val = tk.Label(row, text=init_val, width=15, anchor="e", font=("Consolas", 12, "bold"), bg="white", fg="blue")
            lbl_val.pack(side="right")
            
            self.labels[name] = lbl_val

        # 狀態燈
        self.status_label = tk.Label(root, text="Connecting...", fg="gray", bg="#f0f0f0", font=("Arial", 10))
        self.status_label.pack(side="bottom", pady=5)

        self.running = True
        self.update_data()

    def update_data(self):
        if not self.running:
            return

        try:
            # 模擬 franka_env.py 的請求方式
            response = requests.post(f"{SERVER_URL}/getstate", json={}, timeout=0.2)
            
            if response.status_code == 200:
                data = response.json()
                pose = data.get("pose", [0]*7) # [x, y, z, qx, qy, qz, qw]
                gripper = data.get("gripper_pos", 0)

                # 更新介面 (Pose)
                self.labels["X"].config(text=f"{pose[0]:.4f}")
                self.labels["Y"].config(text=f"{pose[1]:.4f}")
                self.labels["Z"].config(text=f"{pose[2]:.4f}")
                
                self.labels["RX (qx)"].config(text=f"{pose[3]:.4f}")
                self.labels["RY (qy)"].config(text=f"{pose[4]:.4f}")
                self.labels["RZ (qz)"].config(text=f"{pose[5]:.4f}")
                self.labels["RW (qw)"].config(text=f"{pose[6]:.4f}")

                # 更新介面 (Gripper)
                # 簡單判斷開合狀態文字
                g_text = f"{gripper:.3f}"
                if gripper > 0.8: g_text += " (OPEN)"
                elif gripper < 0.1: g_text += " (CLOSED)"
                
                self.labels["Gripper"].config(text=g_text)
                
                self.status_label.config(text="● Connected (Updating)", fg="green")
            else:
                self.status_label.config(text=f"Server Error: {response.status_code}", fg="red")

        except requests.exceptions.ConnectionError:
            self.status_label.config(text="● Disconnected (Is Robot Server running?)", fg="red")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")

        # 每 100ms 更新一次 (10Hz)
        self.root.after(100, self.update_data)

    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()