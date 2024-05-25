import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim
import threading

class ImageComparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Comparator")

        self.frame = tk.Frame(root) #### Frame for image selection  ###
        self.frame.pack(pady=10)

        self.img1_path = tk.StringVar()
        self.img2_path = tk.StringVar()

        self.img1_label = tk.Label(self.frame, text="Vyber obrázok 1:")
        self.img1_label.grid(row=0, column=0, padx=5, pady=5)
        self.img1_entry = tk.Entry(self.frame, textvariable=self.img1_path, width=50)
        self.img1_entry.grid(row=0, column=1, padx=5, pady=5)
        self.img1_button = tk.Button(self.frame, text="Hľadaj", command=self.load_image1)
        self.img1_button.grid(row=0, column=2, padx=5, pady=5)

        self.img2_label = tk.Label(self.frame, text="Vyber obrázok 2:")
        self.img2_label.grid(row=1, column=0, padx=5, pady=5)
        self.img2_entry = tk.Entry(self.frame, textvariable=self.img2_path, width=50)
        self.img2_entry.grid(row=1, column=1, padx=5, pady=5)
        self.img2_button = tk.Button(self.frame, text="Hľadaj", command=self.load_image2)
        self.img2_button.grid(row=1, column=2, padx=5, pady=5)

        self.compare_button = tk.Button(self.frame, text="Porovnaj", command=self.compare_images)
        self.compare_button.grid(row=2, column=0, columnspan=3, pady=10)

        ####### Frame for buttons ######
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.grid(row=3, column=0, columnspan=3, pady=10)

        self.diff_button = tk.Button(self.button_frame, text="Zobraz obrázok rozdielu", command=self.show_difference_image)
        self.diff_button.pack(side=tk.LEFT, padx=5)

        self.harris_button = tk.Button(self.button_frame, text="Detekcia bodov a hrán", command=self.harris_corner_window)
        self.harris_button.pack(side=tk.LEFT, padx=5)

        self.ssim_button = tk.Button(self.button_frame, text="Výpočet SSIM", command=self.calculate_ssim)
        self.ssim_button.pack(side=tk.LEFT, padx=5)

        ######## Buttons for histogram calculation #######
        self.histogram_label = tk.Label(self.button_frame, text="Zobraz Histogramy:")
        self.histogram_label.pack(side=tk.LEFT, padx=5)

        self.histogram_rgb_button = tk.Button(self.button_frame, text="RGB", command=self.show_rgb_histograms)
        self.histogram_rgb_button.pack(side=tk.LEFT, padx=5)

        self.histogram_gray_button = tk.Button(self.button_frame, text="GRAY", command=self.show_combined_histogram)
        self.histogram_gray_button.pack(side=tk.LEFT, padx=5)

        ####### Frame for image display ##########
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=10)

        self.img1_canvas = tk.Canvas(self.image_frame, width=300, height=300, bg='white')
        self.img1_canvas.grid(row=0, column=0, padx=10)
        self.img2_canvas = tk.Canvas(self.image_frame, width=300, height=300, bg='white')
        self.img2_canvas.grid(row=0, column=1, padx=10)
        self.diff_canvas = tk.Canvas(self.image_frame, width=300, height=300, bg='white')
        self.diff_canvas.grid(row=0, column=2, padx=10)

        self.ssim_label = tk.Label(self.image_frame, text="SSIM: N/A", font=("Helvetica", 16))
        self.ssim_label.grid(row=1, column=0, columnspan=3, pady=10)

        ###### Frame for stats ######
        self.stats_frame = tk.Frame(root, pady=10)
        self.stats_frame.pack(fill=tk.BOTH, expand=True)

        self.stats_frame1 = tk.Frame(self.stats_frame)
        self.stats_frame1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.stats_frame2 = tk.Frame(self.stats_frame)
        self.stats_frame2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.stats_text1 = tk.Text(self.stats_frame1, height=15, width=50)
        self.stats_text1.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.stats_text2 = tk.Text(self.stats_frame2, height=15, width=50)
        self.stats_text2.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def load_image1(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img1_path.set(file_path)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.img1_data = img
            img = self.resize_image(img, 300)
            self.img1 = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.img1_canvas.create_image(150, 150, image=self.img1)

    def load_image2(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img2_path.set(file_path)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.img2_data = img
            img = self.resize_image(img, 300)
            self.img2 = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.img2_canvas.create_image(150, 150, image=self.img2)

    def resize_image(self, img, max_dim):
        if len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape

        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        return cv2.resize(img, (new_w, new_h))

    def resize_image_for_display(self, img, max_dim=800):
        if len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape

        if max(h, w) > max_dim:
            if h > w:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            else:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            return cv2.resize(img, (new_w, new_h))
        return img

    def compare_images(self):
        if not self.img1_path.get() or not self.img2_path.get():
            messagebox.showerror("Error", "Please select both images")
            return

        img1 = self.img1_data
        img2 = self.img2_data

        stats1 = self.calculate_stats(img1)
        stats2 = self.calculate_stats(img2)

        self.stats_text1.delete('1.0', tk.END)
        self.stats_text1.insert(tk.END, "Obrázok 1 štatistika:\n")
        self.stats_text1.insert(tk.END, self.format_stats(stats1) + "\n\n")

        self.stats_text2.delete('1.0', tk.END)
        self.stats_text2.insert(tk.END, "Obrázok 2 štatistika:\n")
        self.stats_text2.insert(tk.END, self.format_stats(stats2) + "\n\n")

    def calculate_stats(self, img):
        mean, stddev = cv2.meanStdDev(img)
        median = np.median(img, axis=(0, 1))
        minimum = np.min(img, axis=(0, 1))
        maximum = np.max(img, axis=(0, 1))

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_gray, stddev_gray = cv2.meanStdDev(img_gray)
        median_gray = np.median(img_gray)
        minimum_gray = np.min(img_gray)
        maximum_gray = np.max(img_gray)

        stats = {
            'mean': mean,
            'stddev': stddev,
            'median': median,
            'min': minimum,
            'max': maximum,
            'mean_gray': mean_gray,
            'stddev_gray': stddev_gray,
            'median_gray': median_gray,
            'min_gray': minimum_gray,
            'max_gray': maximum_gray
        }
        return stats

    def format_stats(self, stats):
        formatted = (
            f"Priemer (R, G, B): {stats['mean'].ravel()}\n"
            f"Štd. od. (R, G, B): {stats['stddev'].ravel()}\n"
            f"Medián (R, G, B): {stats['median'].ravel()}\n"
            f"Minimum (R, G, B): {stats['min'].ravel()}\n"
            f"Maximum (R, G, B): {stats['max'].ravel()}\n"
            f"Priemer (Gray): {stats['mean_gray'].ravel()}\n"
            f"Štd. od. (Gray): {stats['stddev_gray'].ravel()}\n"
            f"Medián (Gray): {stats['median_gray']}\n"
            f"Minimum (Gray): {stats['min_gray']}\n"
            f"Maximum (Gray): {stats['max_gray']}"
        )
        return formatted

    def show_difference_image(self, diff=None):
        if not self.img1_path.get() or not self.img2_path.get():
            messagebox.showerror("Error", "Please select both images")
            return

        img1 = self.img1_data
        img2 = self.img2_data

        if img1.shape != img2.shape:
            messagebox.showerror("Error", "Obrázky nemajú rovnakú veľkosť")
            return

        if diff is None:
            diff_img = cv2.absdiff(img1, img2)
            diff_img = 255 - diff_img
        else:
            diff_img = diff

        diff_img = self.resize_image(diff_img, 300)
        self.diff_img = ImageTk.PhotoImage(image=Image.fromarray(diff_img))
        self.diff_canvas.create_image(150, 150, image=self.diff_img)


    def show_rgb_histograms(self):
        if not self.img1_path.get() or not self.img2_path.get():
            messagebox.showerror("Error", "Please select both images")
            return

        img1 = self.img1_data
        img2 = self.img2_data

        img1_resized = self.resize_image(img1, 300)
        img2_resized = self.resize_image(img2, 300)

        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        for i, img in enumerate([img1_resized, img2_resized]):
            ax = axes[i]
            colors = ('r', 'g', 'b')
            for j, color in enumerate(colors):
                hist, bins = np.histogram(img[:, :, j], bins=256, range=(0, 256))
                sns.lineplot(x=bins[:-1], y=hist, color=color, ax=ax, label=color.upper())
            ax.set_xlim([0, 256])
            ax.set_title(f'Obrázok {i+1} Histogram (R, G, B)')
            ax.set_xlabel('Intenzita Pixela')
            ax.set_ylabel('Frekvencia')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def show_combined_histogram(self):
        if not self.img1_path.get() or not self.img2_path.get():
            messagebox.showerror("Error", "Please select both images")
            return

        img1 = self.img1_data
        img2 = self.img2_data

        img1_resized = self.resize_image(img1, 300)
        img2_resized = self.resize_image(img2, 300)

        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        for i, img in enumerate([img1_resized, img2_resized]):
            ax = axes[i]
            sns.histplot(img.ravel(), bins=256, color='gray', kde=False, ax=ax)
            ax.set_title(f'Image {i+1} Histogram (grayscale)')
            ax.set_xlabel('Intenzita Pixela')
            ax.set_ylabel('Frekvencia')

        plt.tight_layout()
        plt.show()

    def non_maximum_suppression(self, coords, dist_thresh):
        if len(coords) == 0:
            return []
        coords = np.array(coords)
        clustering = DBSCAN(eps=dist_thresh, min_samples=1).fit(coords)
        labels = clustering.labels_

        unique_labels = set(labels)
        centers = np.array([coords[labels == label].mean(axis=0) for label in unique_labels])

        return [(int(i[0]), int(i[1])) for i in centers]

    def harris_corner_window(self):
        if not self.img1_path.get() or not self.img2_path.get():
            messagebox.showerror("Error", "Please select both images")
            return

        self.harris_window = tk.Toplevel(self.root)
        self.harris_window.title("Harris Corner Detection")

        ##### Frame for parameters and progress bar #####
        params_frame = tk.Frame(self.harris_window)
        params_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(params_frame, text="Margin:").grid(row=0, column=0, padx=5, pady=5)
        self.margin_entry = tk.Entry(params_frame)
        self.margin_entry.grid(row=0, column=1, padx=5, pady=5)
        self.margin_entry.insert(0, "50")

        tk.Label(params_frame, text="Block Size:").grid(row=1, column=0, padx=5, pady=5)
        self.block_size_entry = tk.Entry(params_frame)
        self.block_size_entry.grid(row=1, column=1, padx=5, pady=5)
        self.block_size_entry.insert(0, "4")

        tk.Label(params_frame, text="K Size:").grid(row=2, column=0, padx=5, pady=5)
        self.ksize_entry = tk.Entry(params_frame)
        self.ksize_entry.grid(row=2, column=1, padx=5, pady=5)
        self.ksize_entry.insert(0, "3")

        tk.Label(params_frame, text="K:").grid(row=3, column=0, padx=5, pady=5)
        self.k_entry = tk.Entry(params_frame)
        self.k_entry.grid(row=3, column=1, padx=5, pady=5)
        self.k_entry.insert(0, "0.04")

        tk.Label(params_frame, text="Threshold:").grid(row=4, column=0, padx=5, pady=5)
        self.threshold_entry = tk.Entry(params_frame)
        self.threshold_entry.grid(row=4, column=1, padx=5, pady=5)
        self.threshold_entry.insert(0, "0.01")

        self.start_button = tk.Button(params_frame, text="Spusti Detekciu", command=self.start_harris_corner_detection)
        self.start_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.reset_button = tk.Button(params_frame, text="Obnov pôvodné nastavenia", command=self.reset_parameters)
        self.reset_button.grid(row=6, column=0, columnspan=2, pady=10)

        ######## Canvas for the result image #######
        self.result_canvas = tk.Canvas(self.harris_window, bg='white')
        self.result_canvas.pack(side=tk.RIGHT, padx=10, pady=10)

    def reset_parameters(self):
        self.margin_entry.delete(0, tk.END)
        self.margin_entry.insert(0, "50")
        self.block_size_entry.delete(0, tk.END)
        self.block_size_entry.insert(0, "4")
        self.ksize_entry.delete(0, tk.END)
        self.ksize_entry.insert(0, "3")
        self.k_entry.delete(0, tk.END)
        self.k_entry.insert(0, "0.04")
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, "0.01")

    def start_harris_corner_detection(self):
        ######## New window for the progress bar #######
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Processing...")
        tk.Label(self.progress_window, text="Please wait...").pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.progress_window, orient='horizontal', mode='indeterminate')
        self.progress_bar.pack(padx=20, pady=20)
        self.progress_bar.start()

        margin = int(self.margin_entry.get())
        block_size = int(self.block_size_entry.get())
        ksize = int(self.ksize_entry.get())
        k = float(self.k_entry.get())
        threshold = float(self.threshold_entry.get())

        ####### Harris Corner Detection in a separate thread #####
        threading.Thread(target=self.harris_corner_detection, args=(margin, block_size, ksize, k, threshold)).start()

    def harris_corner_detection(self, margin, block_size, ksize, k, threshold):
        img1 = self.img1_data
        img2 = self.img2_data

        inverse_difference = cv2.absdiff(img1, img2)
        inverse_difference = 255 - inverse_difference

        coords, img_with_corners = self.detect_harris_corners(inverse_difference, margin, block_size, ksize, k, threshold)

        if coords:
            filtered_corners = self.non_maximum_suppression(coords, dist_thresh=10)
            self.draw_corners(img_with_corners, filtered_corners, (255, 0, 0))

            connected_image = self.connect_regions(img_with_corners, filtered_corners)
            self.update_image_on_canvas(connected_image)
        else:
            messagebox.showinfo("Info", "Neboli nájdené žiadne body.")

        self.progress_bar.stop()
        self.progress_window.destroy()

    def detect_harris_corners(self, image, margin, block_size, ksize, k, threshold, point_size=2):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        corners = cv2.cornerHarris(gray, block_size, ksize, k)
        corners = cv2.dilate(corners, None)
        threshold_value = threshold * corners.max()

        height, width = image.shape[:2]
        limited_corners = corners[margin:height-margin, margin:width-margin]

        corner_coordinates = np.column_stack(np.where(limited_corners > threshold_value))
        coordinates = []

        for coord in corner_coordinates:
            x, y = coord + margin
            coordinates.append((y, x))
            cv2.circle(image, (y, x), point_size, (0, 255, 0), -1)

        return coordinates, image

    def draw_corners(self, image, corners, color):
        point_size = 2
        for x, y in corners:
            cv2.circle(image, (int(x), int(y)), point_size, color, -1)

    def connect_regions(self, image, regions):
        distances = np.zeros((len(regions), len(regions)))
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i != j:
                    distances[i, j] = np.linalg.norm(np.array(region1) - np.array(region2))
                else:
                    distances[i, j] = np.inf

        row_ind, col_ind = linear_sum_assignment(distances)

        for i, j in zip(row_ind, col_ind):
            cv2.line(image, regions[i], regions[j], (0, 255, 0), 2)

        return image

    def update_image_on_canvas(self, image):
        self.harris_window.after(0, self.show_image_on_canvas, image)

    def show_image_on_canvas(self, image):
        image_resized = self.resize_image_for_display(image)
        img = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img)

        width, height = img.size
        self.result_canvas.config(width=width, height=height)
        self.result_canvas.create_image(width // 2, height // 2, image=img_tk)
        self.result_canvas.image = img_tk

    def calculate_ssim(self):
        if not self.img1_path.get() or not self.img2_path.get():
            messagebox.showerror("Error", "Please select both images")
            return

        img1 = self.img1_data
        img2 = self.img2_data

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        score, diff = ssim(img1_gray, img2_gray, full=True)
        diff = (diff * 255).astype("uint8")

        self.ssim_label.config(text=f"SSIM: {score:.4f}")
        self.show_difference_image(diff)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparatorApp(root)
    root.mainloop()