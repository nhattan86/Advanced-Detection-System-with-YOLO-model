import cv2
import time
import tkinter as tk
from tkinter import ttk, filedialog
from ultralytics import YOLO
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import numpy as np
import os

class YOLODetectionApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Advanced Detection System")
        self.root.geometry("1200x800")

        # Variables
        self.running = False
        self.video = None
        self.model = YOLO(r'best.pt')
        self.video_source = "camera"  # Default to camera
        self.video_path = ""
        self.current_frame = 0
        self.total_frames = 0
        self.frame_size = (640, 480)  # Default frame size
        self.detection_thread = None
        
        self.resolutions = [
            ("224x224", 224, 224),    
            ("320x320", 320, 320),    
            ("640x480", 640, 480),
            ("640x640", 640, 640),
            ("800x600", 800, 600),
            ("1280x720", 1280, 720),
            ("1920x1080", 1920, 1080)
        ]

        self.setup_gui()

    def setup_gui(self):
        # Create main frames
        self.control_frame = ctk.CTkFrame(self.root)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.video_frame = ctk.CTkFrame(self.root)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Video display
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True)

        # Controls
        ctk.CTkLabel(self.control_frame, text="DETECTION CONTROLS", font=("Arial", 16, "bold")).pack(pady=10)

        # Source Selection
        source_frame = ctk.CTkFrame(self.control_frame)
        source_frame.pack(pady=10, fill="x")
        
        self.source_var = tk.StringVar(value="camera")
        ctk.CTkRadioButton(source_frame, text="Camera", variable=self.source_var, 
                          value="camera", command=self.on_source_change).pack(pady=5)
        ctk.CTkRadioButton(source_frame, text="Video File", variable=self.source_var, 
                          value="video", command=self.on_source_change).pack(pady=5)

        # Video File Selection
        self.file_frame = ctk.CTkFrame(self.control_frame)
        self.file_button = ctk.CTkButton(self.file_frame, text="Select Video File", 
                                        command=self.select_video_file)
        self.file_button.pack(pady=5)
        self.file_label = ctk.CTkLabel(self.file_frame, text="No file selected", 
                                      wraplength=200)
        self.file_label.pack(pady=5)

        # Resolution Selection
        self.resolution_var = tk.StringVar(value=self.resolutions[2][0])
        ctk.CTkLabel(self.control_frame, text="Resolution:").pack(pady=5)
        resolution_dropdown = ctk.CTkComboBox(
            self.control_frame,
            values=[res[0] for res in self.resolutions],
            variable=self.resolution_var
        )
        resolution_dropdown.pack(pady=5)

        # Confidence Slider
        self.confidence_var = tk.DoubleVar(value=0.5)
        ctk.CTkLabel(self.control_frame, text="Confidence Threshold:").pack(pady=5)
        confidence_slider = ctk.CTkSlider(
            self.control_frame,
            from_=0.0,
            to=1.0,
            variable=self.confidence_var
        )
        confidence_slider.pack(pady=5)
        self.confidence_label = ctk.CTkLabel(self.control_frame, text="0.50")
        self.confidence_label.pack()

        # Video Progress (for video files)
        self.progress_frame = ctk.CTkFrame(self.control_frame)
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.set(0)
        self.time_label = ctk.CTkLabel(self.progress_frame, text="0:00 / 0:00")

        # Stats display
        self.stats_frame = ctk.CTkFrame(self.control_frame)
        self.stats_frame.pack(pady=20, fill="x")
        
        self.fps_label = ctk.CTkLabel(self.stats_frame, text="FPS: 0")
        self.fps_label.pack(pady=5)
        
        self.objects_label = ctk.CTkLabel(self.stats_frame, text="Objects: 0")
        self.objects_label.pack(pady=5)

        # Control buttons
        self.button_frame = ctk.CTkFrame(self.control_frame)
        self.button_frame.pack(pady=20, fill="x")
        
        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Detection",
            command=self.toggle_detection
        )
        self.start_button.pack(pady=5)

        self.restart_button = ctk.CTkButton(
            self.button_frame,
            text="Restart Video",
            command=self.restart_video,
            state="disabled"
        )
        self.restart_button.pack(pady=5)

        # Bind confidence slider update
        confidence_slider.configure(command=self.update_confidence_label)

        # Initial UI state
        self.update_ui_state()

    def on_source_change(self):
        self.video_source = self.source_var.get()
        self.update_ui_state()

    def update_ui_state(self):
        if self.video_source == "camera":
            self.file_frame.pack_forget()
            self.progress_frame.pack_forget()
            self.restart_button.configure(state="disabled")
        else:
            self.file_frame.pack(pady=10, fill="x")
            self.progress_frame.pack(pady=10, fill="x")
            self.progress_bar.pack(pady=5)
            self.time_label.pack(pady=5)
            if self.video_path:
                self.restart_button.configure(state="normal")

    def select_video_file(self):
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mov'),
            ('All files', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='Select a video file',
            filetypes=filetypes
        )
        if filename:
            self.video_path = filename
            self.file_label.configure(text=os.path.basename(filename))
            self.update_ui_state()

    def update_confidence_label(self, value):
        self.confidence_label.configure(text=f"{float(value):.2f}")

    def restart_video(self):
        if self.video and not self.running and self.video_source == "video":
            self.current_frame = 0
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.toggle_detection()

    def format_time(self, seconds):
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"

    def toggle_detection(self):
        if not self.running:
            # Clean up any existing video capture
            if hasattr(self, 'video') and self.video is not None:
                self.video.release()
                self.video = None

            try:
                if self.video_source == "camera":
                    self.video = cv2.VideoCapture(0)
                    if not self.video.isOpened():
                        raise Exception("Could not open camera")
                else:
                    if not self.video_path:
                        raise Exception("No video file selected")
                    self.video = cv2.VideoCapture(self.video_path)
                    if not self.video.isOpened():
                        raise Exception(f"Could not open video file: {self.video_path}")
                    self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

                # Set resolution
                selected_res = next((res for res in self.resolutions if res[0] == self.resolution_var.get()), None)
                if selected_res:
                    self.frame_size = (selected_res[1], selected_res[2])
                    self.video.set(cv2.CAP_PROP_FRAME_WIDTH, selected_res[1])
                    self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, selected_res[2])

                self.running = True
                self.start_button.configure(text="Stop Detection")
                
                # Start detection in a new thread
                if self.detection_thread is not None and self.detection_thread.is_alive():
                    self.detection_thread.join()
                self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
                self.detection_thread.start()

            except Exception as e:
                print(f"Error starting detection: {str(e)}")
                if self.video:
                    self.video.release()
                    self.video = None
                self.running = False
                return

        else:
            self.running = False
            self.start_button.configure(text="Start Detection")
            if self.video:
                self.video.release()
                self.video = None

    def run_detection(self):
        prev_frame_time = time.time()
        
        while self.running:
            try:
                if self.video is None or not self.video.isOpened():
                    break

                ret, frame = self.video.read()
                if not ret:
                    if self.video_source == "video":
                        self.running = False
                        self.root.after(0, lambda: self.start_button.configure(text="Start Detection"))
                    break

                # Resize frame
                frame = cv2.resize(frame, self.frame_size)

                # Update progress for video files
                if self.video_source == "video":
                    current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                    progress = current_frame / self.total_frames
                    current_time = current_frame / self.video.get(cv2.CAP_PROP_FPS)
                    total_time = self.total_frames / self.video.get(cv2.CAP_PROP_FPS)
                    
                    self.root.after(0, lambda p=progress: self.progress_bar.set(p))
                    self.root.after(0, lambda c=current_time, t=total_time: 
                        self.time_label.configure(text=f"{self.format_time(c)} / {self.format_time(t)}"))

                # Perform detection
                results = self.model(frame, conf=self.confidence_var.get())
                
                # Process results
                filtered_results = [r for r in results[0].boxes if r.conf > self.confidence_var.get()]
                
                # Draw detections
                for result in filtered_results:
                    box = result.xyxy[0].cpu().numpy().astype(int)
                    confidence = result.conf.item()
                    x1, y1, x2, y2 = box
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    conf_text = f"{confidence:.2f}"
                    cv2.putText(frame, conf_text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate and display FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_frame_time)
                prev_frame_time = current_time

                # Update stats
                self.root.after(0, lambda f=fps: self.fps_label.configure(text=f"FPS: {f:.1f}"))
                self.root.after(0, lambda c=len(filtered_results): 
                    self.objects_label.configure(text=f"Objects: {c}"))

                # Convert and display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.root.after(0, lambda i=imgtk: self.update_video_display(i))

            except Exception as e:
                print(f"Error during detection: {str(e)}")
                continue

        # Cleanup
        if self.video:
            self.video.release()
            self.video = None

    def update_video_display(self, imgtk):
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = YOLODetectionApp()
    app.run()
