import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import threading
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import math
import pandas as pd

model = YOLO("best.pt")

class SmartParkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bãi Đỗ Xe Thông Minh")
        self.root.geometry("1100x700")

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

        self.label_title = ttk.Label(left_frame, text="BÃI ĐỖ XE THÔNG MINH", font=("Arial", 20, "bold"))
        self.label_title.pack(pady=(0,10))

        self.canvas = tk.Label(left_frame)
        self.canvas.pack()

        button_frame = tk.Frame(left_frame)
        button_frame.pack(pady=10)

        self.btn_start = ttk.Button(button_frame, text="Bắt đầu", command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(button_frame, text="Dừng", command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="n", padx=20, pady=10)

        self.time_label = ttk.Label(right_frame, text="", font=("Arial", 12))
        self.time_label.pack(pady=5)

        self.stats_label = ttk.Label(right_frame, text="Tổng chỗ: 0 | Trống: 0 | Đã có xe: 0", font=("Arial", 14))
        self.stats_label.pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 2))
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_chart.get_tk_widget().pack(pady=10, fill=tk.X)

        self.tree = ttk.Treeview(right_frame, columns=("slot", "start", "end", "duration", "fee"), show='headings', height=15)
        self.tree.pack(pady=10, fill=tk.BOTH, expand=True)

        self.tree.heading("slot", text="Chỗ")
        self.tree.heading("start", text="Thời gian vào")
        self.tree.heading("end", text="Thời gian rời")
        self.tree.heading("duration", text="Thời gian đỗ (phút)")
        self.tree.heading("fee", text="Phí (VNĐ)")

        self.tree.column("slot", width=50, anchor='center')
        self.tree.column("start", width=140, anchor='center')
        self.tree.column("end", width=140, anchor='center')
        self.tree.column("duration", width=120, anchor='center')
        self.tree.column("fee", width=100, anchor='center')

        scrollbar = tk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.time_data = []
        self.empty_data = []
        self.cap = None
        self.running = False
        self.slot_status = {}
        self.slot_entry_time = {}
        self.slot_numbering = {}
        self.records = {}

        self.update_time()

    def update_time(self):
        now = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
        self.time_label.config(text=f"Thời gian hiện tại: {now}")
        self.root.after(1000, self.update_time)

    def start(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self.update_frame, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.export_to_excel()

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.time_data, self.empty_data, color='green', marker='o')
        self.ax.set_title("Số lượng chỗ trống theo thời gian")
        self.ax.set_xlabel("Thời gian")
        self.ax.set_ylabel("Chỗ trống")
        self.ax.tick_params(axis='x', rotation=45)
        self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas_chart.draw()

    def export_to_excel(self):
        if not self.records:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu để xuất.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Chọn nơi lưu file Excel",
            initialfile=f"Parking_Records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        if not file_path:
            print("[INFO] Người dùng đã hủy lưu file Excel.")
            return

        df = pd.DataFrame(list(self.records.values()))
        df.to_excel(file_path, index=False)
        print(f"[INFO] Đã xuất file Excel: {file_path}")
        messagebox.showinfo("Thông báo", f"Dữ liệu đã được lưu tại:\n{file_path}")

    def update_frame(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (640, 480))
            results = model.predict(resized, conf=0.4, iou=0.3, verbose=False)[0]

            vacant_count = 0
            occupied_count = 0
            current_slots = {}
            slot_idx = 1

            for r in results.boxes:
                cls_id = int(r.cls[0])
                label = model.names[cls_id]
                conf = float(r.conf[0])
                x1, y1, x2, y2 = map(int, r.xyxy[0])

                # Làm tròn để ổn định slot ID
                slot_id = f"{round(x1/10)*10}_{round(y1/10)*10}"
                current_slots[slot_id] = label

                if slot_id not in self.slot_numbering:
                    self.slot_numbering[slot_id] = slot_idx
                    print(f"[DEBUG] Thêm slot mới: {slot_id} → #{slot_idx}")
                    slot_idx += 1

                slot_num = self.slot_numbering[slot_id]

                # Màu tím cho "occupied", đỏ cho "vacant"
                if label == "vacant":
                    color = (0, 0, 255)  # Red
                    vacant_count += 1
                else:  # label == "occupied"
                    color = (128, 0, 128)  # Purple
                    occupied_count += 1

                cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(resized, f"{label} {conf:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(resized, f"#{slot_num}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            for slot_id, label in current_slots.items():
                prev_status = self.slot_status.get(slot_id)
                if prev_status != label:
                    if label == "occupied":
                        self.slot_entry_time[slot_id] = datetime.now()
                        print(f"[Slot {slot_id}] Vào lúc: {self.slot_entry_time[slot_id]}")
                    elif label == "vacant" and slot_id in self.slot_entry_time:
                        start_time = self.slot_entry_time.pop(slot_id)
                        end_time = datetime.now()
                        duration = end_time - start_time
                        minutes = duration.total_seconds() / 60
                        fee_units = math.ceil(minutes / 5)
                        fee = fee_units * 2000
                        slot_num = self.slot_numbering.get(slot_id, -1)
                        record = {
                            "Slot": slot_num,
                            "Thời gian vào": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Thời gian rời": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Thời gian đỗ (phút)": round(minutes, 1),
                            "Phí (VNĐ)": fee
                        }
                        self.records[f"{slot_id}_{end_time.timestamp()}"] = record

                        print(f"[Slot {slot_num}] ĐỖ {minutes:.1f} phút → {fee:,} VNĐ")

                        self.tree.insert("", tk.END, values=(
                            record["Slot"],
                            record["Thời gian vào"],
                            record["Thời gian rời"],
                            record["Thời gian đỗ (phút)"],
                            f"{record['Phí (VNĐ)']:,}"
                        ))

                self.slot_status[slot_id] = label

            total_slots = vacant_count + occupied_count
            stats_text = f"Tổng chỗ: {total_slots} | Trống: {vacant_count} | Đã có xe: {occupied_count}"
            self.stats_label.config(text=stats_text)

            now = datetime.now().strftime("%H:%M:%S")
            self.time_data.append(now)
            self.empty_data.append(vacant_count)
            if len(self.time_data) > 20:
                self.time_data.pop(0)
                self.empty_data.pop(0)

            self.update_plot()

            img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

            time.sleep(0.2)

        self.canvas.config(image='')
        self.stats_label.config(text="Tổng chỗ: 0 | Trống: 0 | Đã có xe: 0")

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartParkingApp(root)
    root.mainloop()
