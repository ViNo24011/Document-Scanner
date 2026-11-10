"""Mục đích: tạo giao diện Tkinter để quét tài liệu bằng thao tác chuột.
Người dùng tải ảnh, chỉnh bốn góc giấy, chọn bộ lọc, xem trước và lưu ảnh/PDF.
DocumentScannerApp giữ trạng thái ảnh gốc, ảnh hiển thị nhỏ và ảnh sau xử lý.
Tọa độ trên canvas được nhân scale_ratio để quay về ảnh gốc trước khi warp.
Các hàm on_* là callback (được Tkinter gọi khi có sự kiện), không phải thuật toán.
Khởi động bằng python gui.py; đọc load_image -> auto_detect_boundaries ->
warp_perspective -> apply_effects để theo dõi luồng xử lý chính.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import scanner
import image_algorithms as alg

class DocumentScannerApp:
    def __init__(self, root):
        """Khởi tạo cửa sổ, các biến ảnh, tham số lọc và bộ điều khiển Tkinter.
        """
        self.root = root
        self.root.title("Interactive Document Scanner & Perspective Tool")
        self.root.geometry("1250x780")
        self.root.minsize(1000, 600)
        
        # Style configuration
        self.setup_styles()

        # Image variables
        self.cv_img_original = None    # Original high-res OpenCV image (BGR)
        self.cv_img_display = None     # Resized OpenCV image for source canvas (BGR)
        self.cv_img_warped = None      # Current warped/corrected image (BGR/Gray/Binary)
        
        # Geometry ratios
        self.scale_ratio = 1.0         # cv_img_original / cv_img_display
        self.canvas_pts = []           # 4 corners in canvas coordinate space: [[x, y], ...]
        self.active_corner_idx = None  # Index of corner currently being dragged (0-3)
        self.drag_radius = 12          # Distance to detect clicks on corners

        # Default settings
        self.blur_val = tk.IntVar(value=5)
        self.canny_low_val = tk.IntVar(value=75)
        self.canny_high_val = tk.IntVar(value=200)
        self.mode_var = tk.StringVar(value="color")
        self.rotate_angle = 0          # Cumulative rotation (0, 90, 180, 270)
        self.sharpen_var = tk.BooleanVar(value=False)
        self.contrast_val = tk.IntVar(value=0)

        # UI Layout creation
        self.create_widgets()

    def setup_styles(self):
        """Thiết lập màu, font và trạng thái nút cho giao diện; không xử lý ảnh.
        """
        style = ttk.Style()
        style.theme_use("clam")
        
        # Custom colors for a sleek modern UI
        style.configure(".", font=("Helvetica", 10))
        style.configure("TFrame", background="#f0f2f5")
        style.configure("TLabel", background="#f0f2f5", foreground="#333333")
        style.configure("Header.TLabel", font=("Helvetica", 12, "bold"), background="#f0f2f5")
        
        # Primary Action Button styling
        style.configure("Action.TButton", font=("Helvetica", 10, "bold"), foreground="white", background="#1a73e8")
        style.map("Action.TButton", background=[("active", "#1557b0"), ("pressed", "#0d3c80")])
        
        # Standard Button styling
        style.configure("TButton", font=("Helvetica", 10), background="#e1e4e8")
        style.map("TButton", background=[("active", "#d1d5da")])
        
        # Radio button & checkbox background colors
        style.configure("TRadiobutton", background="#f0f2f5")
        style.configure("TCheckbutton", background="#f0f2f5")

    def create_widgets(self):
        # Base container
        """Tạo thanh công cụ, hai canvas và bộ điều chỉnh; gắn nút với callback.
        Bên trái hiển thị ảnh và bốn góc; bên phải hiển thị kết quả sau xử lý.
        """
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Top Control Bar (File actions)
        top_bar = ttk.Frame(main_frame, padding=10, relief=tk.RAISED)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        
        btn_load = ttk.Button(top_bar, text="📂 Load Image", command=self.load_image, style="Action.TButton")
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_auto = ttk.Button(top_bar, text="🔍 Auto Detect", command=self.auto_detect_boundaries)
        btn_auto.pack(side=tk.LEFT, padx=5)
        
        btn_reset = ttk.Button(top_bar, text="🔄 Reset Grid", command=self.reset_corners)
        btn_reset.pack(side=tk.LEFT, padx=5)

        # Spacer
        ttk.Frame(top_bar, width=30).pack(side=tk.LEFT)

        btn_save_img = ttk.Button(top_bar, text="💾 Save Image", command=self.save_image)
        btn_save_img.pack(side=tk.LEFT, padx=5)

        btn_save_pdf = ttk.Button(top_bar, text="📄 Export PDF", command=self.save_pdf)
        btn_save_pdf.pack(side=tk.LEFT, padx=5)

        # Status indicator
        self.lbl_status = ttk.Label(top_bar, text="Please load an image to begin...", font=("Helvetica", 10, "italic"))
        self.lbl_status.pack(side=tk.RIGHT, padx=15)

        # 2. Main Work Area (split panels: Source Canvas | Output Canvas)
        work_area = ttk.Frame(main_frame, padding=10)
        work_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Column: Source Image & Crop Handles
        left_pane = ttk.Frame(work_area)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left_pane, text="Original Image (Adjust Green Handles)", style="Header.TLabel").pack(anchor=tk.W, pady=2)
        
        # Source canvas holds the image + vector handles
        self.src_canvas = tk.Canvas(left_pane, bg="#d8dee9", highlightthickness=1, highlightbackground="#abb2bf")
        self.src_canvas.pack(fill=tk.BOTH, expand=True)
        self.src_canvas.bind("<Button-1>", self.on_canvas_press)
        self.src_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.src_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # Divider frame
        ttk.Frame(work_area, width=15).pack(side=tk.LEFT)

        # Right Column: Output / Corrected Preview
        right_pane = ttk.Frame(work_area)
        right_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_pane, text="Corrected Document Preview", style="Header.TLabel").pack(anchor=tk.W, pady=2)
        
        # Output canvas displays warped results
        self.dst_canvas = tk.Canvas(right_pane, bg="#d8dee9", highlightthickness=1, highlightbackground="#abb2bf")
        self.dst_canvas.pack(fill=tk.BOTH, expand=True)

        # 3. Bottom Adjustment Controls Panel
        ctrl_panel = ttk.Frame(main_frame, padding=10, relief=tk.SUNKEN)
        ctrl_panel.pack(side=tk.BOTTOM, fill=tk.X)

        # Preprocessing Slider column
        slider_frame = ttk.LabelFrame(ctrl_panel, text=" Preprocessing Edge Tuners ", padding=10)
        slider_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Blur slider
        ttk.Label(slider_frame, text="Blur Kernel Size (Odd):").grid(row=0, column=0, sticky=tk.W, padx=2)
        self.blur_scale = tk.Scale(slider_frame, from_=1, to=31, orient=tk.HORIZONTAL, variable=self.blur_val, 
                                   command=self.on_param_change)
        self.blur_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)

        # Canny Low slider
        ttk.Label(slider_frame, text="Canny Low Threshold:").grid(row=1, column=0, sticky=tk.W, padx=2)
        self.canny_low_scale = tk.Scale(slider_frame, from_=10, to=300, orient=tk.HORIZONTAL, variable=self.canny_low_val,
                                        command=self.on_param_change)
        self.canny_low_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)

        # Canny High slider
        ttk.Label(slider_frame, text="Canny High Threshold:").grid(row=2, column=0, sticky=tk.W, padx=2)
        self.canny_high_scale = tk.Scale(slider_frame, from_=50, to=400, orient=tk.HORIZONTAL, variable=self.canny_high_val,
                                         command=self.on_param_change)
        self.canny_high_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)

        # Filter mode / rotation column
        options_frame = ttk.LabelFrame(ctrl_panel, text=" Document Filters & Adjustments ", padding=10)
        options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, expand=True)

        # Output style options
        ttk.Label(options_frame, text="Color Filter Mode:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        mode_btn_frame = ttk.Frame(options_frame)
        mode_btn_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W)
        
        ttk.Radiobutton(mode_btn_frame, text="Full Color", value="color", variable=self.mode_var,
                        command=self.apply_effects).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_btn_frame, text="Grayscale", value="gray", variable=self.mode_var,
                        command=self.apply_effects).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_btn_frame, text="Scanned (B&W)", value="scanned", variable=self.mode_var,
                        command=self.apply_effects).pack(side=tk.LEFT, padx=5)

        # Sharpen toggle
        ttk.Checkbutton(options_frame, text="Sharpen Text", variable=self.sharpen_var,
                        command=self.apply_effects).grid(row=1, column=0, sticky=tk.W, pady=5)

        # Rotation controls
        ttk.Label(options_frame, text="Rotate Output:").grid(row=1, column=1, sticky=tk.W, padx=5)
        btn_rot_left = ttk.Button(options_frame, text="↺ Rotate L", command=lambda: self.rotate(270))
        btn_rot_left.grid(row=1, column=2, sticky=tk.W, padx=2)
        btn_rot_right = ttk.Button(options_frame, text="↻ Rotate R", command=lambda: self.rotate(90))
        btn_rot_right.grid(row=1, column=3, sticky=tk.W, padx=2)

        ttk.Label(options_frame, text="Contrast (%; 0 = Off):").grid(
            row=2, column=0, sticky=tk.W, pady=5)
        self.contrast_scale = tk.Scale(
            options_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.contrast_val, command=self.on_contrast_change)
        self.contrast_scale.grid(row=2, column=1, columnspan=3, sticky=tk.EW)

    def load_image(self):
        """Mở hộp thoại chọn ảnh, đọc ảnh gốc và tạo ảnh nhỏ vừa canvas.
        Lưu scale_ratio để chuyển tọa độ về ảnh gốc; sau đó tự tìm biên giấy.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return
            
        # Load high-res image via OpenCV
        self.cv_img_original = cv2.imread(file_path)
        if self.cv_img_original is None:
            messagebox.showerror("Error", "Could not open or read the image file.")
            return

        h_orig, w_orig = self.cv_img_original.shape[:2]
        
        # Calculate resize scale to fit canvas (target width & height around 550px)
        max_canvas_dim = 550.0
        self.scale_ratio = max(w_orig / max_canvas_dim, h_orig / max_canvas_dim)
        
        display_w = int(w_orig / self.scale_ratio)
        display_h = int(h_orig / self.scale_ratio)
        
        # Resize BGR image for canvas
        self.cv_img_display = cv2.resize(self.cv_img_original, (display_w, display_h))
        
        # Set up canvas dimensions
        self.src_canvas.config(width=display_w, height=display_h)
        self.dst_canvas.config(width=display_w, height=display_h)
        
        # Reset rotation angle
        self.rotate_angle = 0
        
        # Run Auto Edge detection
        self.auto_detect_boundaries()
        
        self.lbl_status.config(text=f"Loaded: {os.path.basename(file_path)} ({w_orig}x{h_orig})")

    def auto_detect_boundaries(self):
        """Tìm bốn góc trên ảnh hiển thị theo tham số Blur/Canny hiện tại.
        Chuẩn hóa kernel lẻ, tiền xử lý, tìm tứ giác; không thấy thì dùng vùng dự phòng.
        Cuối cùng vẽ góc và tính lại ảnh phối cảnh.
        """
        if self.cv_img_display is None:
            return

        h_disp, w_disp = self.cv_img_display.shape[:2]
        
        # Blur kernel size must be odd
        k_val = self.blur_val.get()
        if k_val % 2 == 0:
            k_val += 1
            self.blur_val.set(k_val)

        # Run preprocessing on canvas-sized image
        _, _, edged = scanner.preprocess_image(
            self.cv_img_display, 
            blur_kernel=(k_val, k_val),
            canny_low=self.canny_low_val.get(),
            canny_high=self.canny_high_val.get()
        )
        
        doc_contour = scanner.find_document_contour(edged)
        
        if doc_contour is not None:
            # Found a quadrilateral! Reshape to (4, 2)
            pts = doc_contour.reshape(4, 2).astype(np.float32)
            # Reorder them to the standard [TL, TR, BR, BL]
            ordered_pts = scanner.order_points(pts)
            self.canvas_pts = ordered_pts.tolist()
            self.lbl_status.config(text="Auto-detected document boundaries.")
        else:
            # No 4-sided shape found, fallback to margin boxes
            self.reset_corners()
            self.lbl_status.config(text="Edges not distinct. Standard grid applied.")

        self.draw_canvas_overlay()
        self.warp_perspective()

    def reset_corners(self):
        """Đặt lại vùng cắt giữa ảnh với lề 8%, rồi cập nhật kết quả.
        """
        if self.cv_img_display is None:
            return
        h_disp, w_disp = self.cv_img_display.shape[:2]
        # Vùng cắt dự phòng lùi vào 8% mỗi phía; không phải kết quả dò giấy.
        pts = scanner.get_default_corners(w_disp, h_disp, margin=0.08)
        self.canvas_pts = pts.tolist()
        self.draw_canvas_overlay()
        self.warp_perspective()

    def draw_canvas_overlay(self):
        """Vẽ ảnh nguồn và bốn tay nắm trên canvas để chỉnh góc.
        Giữ tham chiếu PhotoImage để Tkinter không giải phóng ảnh đang hiển thị.
        """
        if self.cv_img_display is None:
            return

        # Convert display image to Tk format
        img_rgb = alg.to_rgb(self.cv_img_display)
        img_pil = Image.fromarray(img_rgb)
        self.tk_img_src = ImageTk.PhotoImage(image=img_pil)
        
        # Redraw canvas background
        self.src_canvas.delete("all")
        self.src_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img_src)
        
        # Draw bounding lines (polygon shape)
        pts_flattened = [coord for pt in self.canvas_pts for coord in pt]
        self.src_canvas.create_polygon(
            pts_flattened, 
            outline="#2ecc71",  # Vibrant green
            fill="", 
            width=3, 
            tags="boundary"
        )
        
        # Draw draggable handle circles
        for idx, (x, y) in enumerate(self.canvas_pts):
            r = self.drag_radius
            color = "#16a085" if idx != self.active_corner_idx else "#e74c3c" # Red highlight when active
            self.src_canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color,
                outline="white",
                width=2,
                tags=("handle", f"handle-{idx}")
            )

    def on_canvas_press(self, event):
        """Khi nhấn chuột gần một góc, chọn góc đó làm góc đang kéo.
        """
        if not self.canvas_pts:
            return
            
        # Detect if click is near any handle
        mx, my = event.x, event.y
        for idx, (x, y) in enumerate(self.canvas_pts):
            dist = np.sqrt((x - mx)**2 + (y - my)**2)
            if dist < self.drag_radius + 4:
                self.active_corner_idx = idx
                self.draw_canvas_overlay()
                break

    def on_canvas_drag(self, event):
        """Cập nhật tọa độ góc theo chuột, giới hạn trong ảnh và vẽ lại khung.
        Chưa warp liên tục trong lúc kéo để giảm khối lượng tính toán.
        """
        if self.active_corner_idx is None:
            return
            
        h_disp, w_disp = self.cv_img_display.shape[:2]
        
        # Constrain handle positions inside canvas boundaries
        mx = max(0, min(event.x, w_disp - 1))
        my = max(0, min(event.y, h_disp - 1))
        
        self.canvas_pts[self.active_corner_idx] = [mx, my]
        self.draw_canvas_overlay()

    def on_canvas_release(self, event):
        """Khi thả chuột, kết thúc kéo góc và tính lại ảnh đã nắn.
        """
        if self.active_corner_idx is not None:
            self.active_corner_idx = None
            self.draw_canvas_overlay()
            self.warp_perspective()

    def on_param_change(self, *args):
        # Fired when blur/canny sliders change
        # Re-run auto edge detection if image exists
        """Khi thanh Blur/Canny thay đổi, chạy lại phát hiện giấy nếu đã có ảnh.
        """
        if self.cv_img_display is not None:
            self.auto_detect_boundaries()

    def warp_perspective(self):
        """Đổi góc canvas sang tọa độ ảnh gốc, nắn ảnh rồi áp dụng bộ lọc.
        Giữ ảnh đã nắn chưa lọc làm bản nền; báo trạng thái nếu bốn góc không hợp lệ.
        """
        if self.cv_img_original is None:
            return
            
        # Scale handles back to the high-res original coordinate system
        scaled_pts = np.array(self.canvas_pts, dtype=np.float32) * self.scale_ratio
        
        # Compute perspective warp on high resolution
        try:
            self.cv_img_warped_base = scanner.four_point_transform(self.cv_img_original, scaled_pts)
        except ValueError as error:
            self.lbl_status.config(text=str(error))
            return
        
        # Apply filters
        self.apply_effects()

    def on_contrast_change(self, _value: str) -> None:
        """Khi thay đổi Contrast, tính lại bộ lọc từ bản nền chưa lọc.
        """
        self.apply_effects()

    def apply_effects(self):
        """Tạo ảnh kết quả từ bản sao ảnh đã nắn, tránh cộng dồn hiệu ứng.
        Theo code hiện tại: chọn xám nếu cần -> tăng tương phản -> B&W nếu được chọn
        -> tăng nét tùy chọn -> xoay -> cập nhật preview. Không thay đổi ảnh gốc.
        """
        if not hasattr(self, 'cv_img_warped_base') or self.cv_img_warped_base is None:
            return

        img = self.cv_img_warped_base.copy()

        # 1. Color Filters
        mode = self.mode_var.get()
        if mode == "gray":
            img = alg.grayscale(img)
        if self.contrast_val.get() > 0:
            img = scanner.enhance_contrast(img, self.contrast_val.get() / 100.0)
        if mode == "scanned":
            img = scanner.binarize_image(img)

        # 2. Sharpening
        if self.sharpen_var.get():
            img = scanner.sharpen_image(img)

        # 3. Rotate
        if self.rotate_angle > 0:
            img = scanner.rotate_image(img, self.rotate_angle)

        self.cv_img_warped = img
        self.draw_warped_preview()

    def rotate(self, angle):
        """Cộng góc xoay, lấy modulo 360 rồi cập nhật lại kết quả.
        """
        if self.cv_img_original is None:
            return
        # Rotate angle accumulates in increments of 90 degrees (modulo 360)
        self.rotate_angle = (self.rotate_angle + angle) % 360
        self.apply_effects()

    def draw_warped_preview(self):
        """Thu phóng bản xem trước cho vừa canvas và hiển thị ở giữa.
        Chỉ resize ảnh hiển thị; ảnh lưu vẫn lấy từ cv_img_warped ở độ phân giải xử lý.
        """
        if self.cv_img_warped is None:
            return

        # Scale preview to fit destination canvas
        h_warp, w_warp = self.cv_img_warped.shape[:2]
        
        # Retrieve size of destination canvas layout
        disp_w = self.dst_canvas.winfo_width()
        disp_h = self.dst_canvas.winfo_height()
        if disp_w <= 1 or disp_h <= 1:
            # Fallback if canvas is not fully packed/mapped yet
            disp_w = self.src_canvas.winfo_width()
            disp_h = self.src_canvas.winfo_height()

        scale = max(w_warp / float(disp_w), h_warp / float(disp_h))
        
        preview_w = int(w_warp / scale)
        preview_h = int(h_warp / scale)
        
        preview_img = cv2.resize(self.cv_img_warped, (preview_w, preview_h))

        # Convert to tkinter compatible format
        if len(preview_img.shape) == 3:
            preview_rgb = alg.to_rgb(preview_img)
        else:
            preview_rgb = alg.to_rgb(preview_img)
            
        preview_pil = Image.fromarray(preview_rgb)
        self.tk_img_dst = ImageTk.PhotoImage(image=preview_pil)

        self.dst_canvas.delete("all")
        # Center the output image in the canvas
        cx = disp_w // 2
        cy = disp_h // 2
        self.dst_canvas.create_image(cx, cy, anchor=tk.CENTER, image=self.tk_img_dst)

    def save_image(self):
        """Cho chọn đường dẫn và ghi ảnh kết quả hiện tại; cảnh báo nếu chưa có ảnh.
        """
        if self.cv_img_warped is None:
            messagebox.showwarning("Warning", "No scanned document available to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("BMP Image", "*.bmp")]
        )
        if not file_path:
            return

        cv2.imwrite(file_path, self.cv_img_warped)
        messagebox.showinfo("Success", f"Scanned image successfully saved to:\n{file_path}")

    def save_pdf(self):
        """Đổi kết quả sang RGB rồi dùng Pillow lưu PDF một trang.
        Hộp thoại thông báo thành công hoặc lỗi xuất file.
        """
        if self.cv_img_warped is None:
            messagebox.showwarning("Warning", "No scanned document available to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Document", "*.pdf")]
        )
        if not file_path:
            return

        try:
            # Convert warped BGR/Gray image to RGB PIL image
            if len(self.cv_img_warped.shape) == 3:
                rgb_img = alg.to_rgb(self.cv_img_warped)
            else:
                rgb_img = alg.to_rgb(self.cv_img_warped)
                
            pil_img = Image.fromarray(rgb_img)
            pil_img.save(file_path, "PDF", resolution=100.0)
            messagebox.showinfo("Success", f"Document PDF successfully exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PDF:\n{str(e)}")

def main():
    """Tạo cửa sổ và DocumentScannerApp, đăng ký resize rồi chạy vòng lặp sự kiện.
    """
    root = tk.Tk()
    app = DocumentScannerApp(root)
    
    # Bind resize helper to update warped image rendering size correctly
    def on_resize(event):
        # Only handle global window resizing, avoid inner widget events
        """Khi cửa sổ chính đổi kích thước, hẹn vẽ preview sau 100 ms cho layout ổn định.
        """
        if event.widget == root:
            # Small delay to let tk settle geometry calculations
            root.after(100, app.draw_warped_preview)
            
    root.bind("<Configure>", on_resize)
    root.mainloop()

if __name__ == "__main__":
    main()

