import sys
import os
import cv2
from huggingface_hub import snapshot_download
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__))

def sort_boxes_reading_order(boxes):
    centers = sorted([(box, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes], key=lambda p: p[2])

    gaps = [centers[i+1][2] - centers[i][2] for i in range(len(centers) - 1)]
    if len(gaps) == 0:
        return boxes
    
    mean_gap = sum(gaps) / len(gaps)
    std_gap = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5
    row_threshold = mean_gap + std_gap
    rows, cur = [], [centers[0]]

    for i, pt in enumerate(centers[1:]):
        if gaps[i] <= row_threshold:
            cur.append(pt)
        else:
            rows.append(sorted(cur, key=lambda p: -p[1]))
            cur = [pt]

    rows.append(sorted(cur, key=lambda p: -p[1]))

    return [item[0] for row in rows for item in row]

def detect_bubbles(model, image_input, shrink_percent=6):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input
        
    if img is None:
        return

    h_img, w_img = img.shape[:2]

    results = model.predict(img, imgsz=1280, conf=0.25, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    boxes = sort_boxes_reading_order(boxes)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        box_w = x2 - x1
        box_h = y2 - y1
        shrink_x = int(box_w * (shrink_percent / 100.0))
        shrink_y = int(box_h * (shrink_percent / 100.0))
        x1 = max(0, x1 + shrink_x)
        y1 = max(0, y1 + shrink_y)
        x2 = min(w_img, x2 - shrink_x)
        y2 = min(h_img, y2 - shrink_y)
        
        if x1 < x2 and y1 < y2:
            cropped_bubble = img[y1:y2, x1:x2]
            yield cropped_bubble

def clean_bubble_edges(cropped_bubble):
    margin = 3
    h_img, w_img = cropped_bubble.shape[:2]

    gray = cv2.cvtColor(cropped_bubble, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dark_pixels = sum([h[0] for h in hist[0:128]])
    dark_ratio = dark_pixels / (h_img * w_img)
    if dark_ratio > 0.60:
        processed_img = cv2.bitwise_not(cropped_bubble)
        gray = cv2.bitwise_not(gray)
    else:
        processed_img = cropped_bubble.copy()

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cleaned_bubble = processed_img.copy()
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        touching_left   = (x <= margin)
        touching_top    = (y <= margin)
        touching_right  = (x + w >= w_img - margin)
        touching_bottom = (y + h >= h_img - margin)

        if touching_left or touching_top or touching_right or touching_bottom:
            cv2.drawContours(cleaned_bubble, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)

    return cleaned_bubble

class OCRWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, target_files, bubble_model, ocr_model, feature_extractor, tokenizer, output_file):
        super().__init__()
        self.target_files, self.bubble_model, self.ocr_model = target_files, bubble_model, ocr_model
        self.feature_extractor, self.tokenizer, self.output_file = feature_extractor, tokenizer, output_file

    def run(self):
        total = len(self.target_files)
        for i, img_path in enumerate(self.target_files):
            filename = os.path.basename(img_path)
            self.progress_signal.emit(f'Processing {i+1}/{total}: {filename}')

            img = cv2.imread(img_path)
            if img is None: 
                continue

            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f'\n--- {filename} ---\n')

                for bubble_img in detect_bubbles(self.bubble_model, img):
                    cleaned_img = clean_bubble_edges(bubble_img)

                    image_rgb = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
                    pixel_values = self.feature_extractor(images=image_rgb, return_tensors='pt').pixel_values.to(device)
                    with torch.no_grad():
                        output_ids = self.ocr_model.generate(pixel_values)[0]
                    
                    text = self.tokenizer.decode(output_ids, skip_special_tokens=True).replace(' ', '')
                    f.write(f'{text.strip()}\n')
            
        self.finished_signal.emit()

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.init_models()
        self.target_files = []
        self.initUI()

    def init_models(self):
        ocr_model = snapshot_download(repo_id = 'kha-white/manga-ocr-base', local_dir = f'{current_dir}/model/manga-ocr-base')
        self.feature_extractor = AutoImageProcessor.from_pretrained(ocr_model)
        self.tokenizer = AutoTokenizer.from_pretrained(ocr_model)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model).to(device)

        bubble_detector_model_dir = snapshot_download(repo_id = 'ogkalu/comic-speech-bubble-detector-yolov8m', local_dir = f'{current_dir}/model/comic-speech-bubble-detector-yolov8m')
        self.bubble_detector_model = YOLO(f'{bubble_detector_model_dir}/comic-speech-bubble-detector.pt')
        pass

    def initUI(self):
        self.setWindowTitle('Manga Text Extractor')
        self.setFixedSize(400, 130)
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout()

        path_layout = QHBoxLayout()

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText('Select file or folder, or drag it here.')
        path_layout.addWidget(self.path_input)

        self.btn_browse = QPushButton('Browse')
        self.btn_browse.clicked.connect(self.open_dialog)
        path_layout.addWidget(self.btn_browse)

        main_layout.addLayout(path_layout)

        self.support_label = QLabel('Supports: .jpeg, .jpg, .png')
        self.support_label.setStyleSheet('color: gray; font-size: 11px; margin-left: 2px;')
        main_layout.addWidget(self.support_label)

        self.btn_start = QPushButton('Start')
        self.btn_start.setFixedWidth(100)
        self.btn_start.clicked.connect(self.run_process)

        main_layout.addWidget(self.btn_start, alignment=Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel('Ready')
        self.status_label.setStyleSheet('color: gray; font-size: 12px; margin-left: 2px;')

        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

    def open_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select Files', '', 'Image Files (*.png *.jpg *.jpeg)')
        
        if files:
            self.target_files = files
            self.path_input.setText(files[0] if len(files) == 1 else f'{len(files)} files selected')
            self.status_label.setText(f'Selected: {len(self.target_files)} files')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()

            is_folder = os.path.isdir(path)
            is_image = path.lower().endswith(('.png', '.jpg', '.jpeg'))

            if is_folder or is_image:
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()

        if not urls:
            return

        path = urls[0].toLocalFile()
        self.path_input.setText(path)

        valid_ext = ('.png', '.jpg', '.jpeg')

        if os.path.isdir(path):
            self.target_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_ext)])
            count = len(self.target_files)
            self.status_label.setText(f'Loaded folder: {count} images.')

        elif path.lower().endswith(valid_ext):
            self.target_files = [path]
            self.status_label.setText(f'Loaded file: {os.path.basename(path)}')

        event.acceptProposedAction()

    def run_process(self):
        if not self.target_files:
            self.status_label.setText('Error: No images loaded!')
            return
        
        self.output_file = os.path.join(current_dir, 'extracted_text.txt')
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            pass

        self.target_files.sort()

        self.btn_start.setEnabled(False) 
        self.btn_browse.setEnabled(False)
        
        self.worker = OCRWorker(
            self.target_files, 
            self.bubble_detector_model,
            self.ocr_model, 
            self.feature_extractor, 
            self.tokenizer, 
            self.output_file
        )
        
        self.worker.progress_signal.connect(self.status_label.setText)
        self.worker.finished_signal.connect(self.on_finished)
        
        self.worker.start()

    def on_finished(self):
        self.status_label.setText('Task Completed!')
        self.btn_start.setEnabled(True)
        self.btn_browse.setEnabled(True)
  
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec())