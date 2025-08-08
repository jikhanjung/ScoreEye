#!/usr/bin/env python3
"""
ScoreEye GUI - Desktop application for measure detection in sheet music
"""

import sys
import os
import time
import threading
import logging
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QSlider, QGroupBox,
    QMessageBox, QProgressBar, QCheckBox, QSplitter, QDialog, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QAction, QImage
import fitz  # PyMuPDF
import cv2
import numpy as np
from detect_measure import MeasureDetector
import json
from pathlib import Path
import os
import math
import wave
import struct
import tempfile
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLOv8 detection features will be disabled.")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Metronome features will be disabled.")

def setup_logger():
    """Setup logger to save debug messages to dated files in logs/ directory"""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create logger
    logger = logging.getLogger('ScoreEye')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler with dated filename
    current_date = datetime.now().strftime("%Y%m%d")
    log_filename = os.path.join(logs_dir, f"scoreeye_{current_date}.log")
    
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()


class DetectionThread(QThread):
    """Thread for running detection without freezing the GUI"""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, pdf_path, page_num, dpi, use_alternative_preprocessing=False, system_group_threshold=8.0):
        super().__init__()
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.dpi = dpi
        self.use_alternative_preprocessing = use_alternative_preprocessing
        self.system_group_threshold = system_group_threshold
        
    def run(self):
        try:
            self.progress.emit("Initializing detector...")
            detector = MeasureDetector(debug=False)
            
            # Apply custom system group threshold
            detector.config.system_group_clustering_ratio = self.system_group_threshold
            logger.debug(f"DetectionThread: Setting system_group_clustering_ratio = {self.system_group_threshold}")
            
            self.progress.emit(f"Loading page {self.page_num + 1}...")
            results = detector.detect_measures_from_pdf(
                self.pdf_path, self.page_num, self.dpi, self.use_alternative_preprocessing
            )
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class DetectAllThread(QThread):
    """Thread for running detection on all pages of PDF"""
    progress = pyqtSignal(str)
    page_completed = pyqtSignal(int, dict)  # page_num, results
    finished = pyqtSignal(dict)  # all_results
    error = pyqtSignal(str)
    
    def __init__(self, pdf_path, dpi=300, yolo_confidence=0.5, use_alt_preprocessing=False):
        super().__init__()
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.yolo_confidence = yolo_confidence
        self.use_alt_preprocessing = use_alt_preprocessing
        
    def run(self):
        try:
            import fitz
            from ultralytics import YOLO
            import json
            from datetime import datetime
            import os
            
            # Load PDF
            pdf_document = fitz.open(self.pdf_path)
            total_pages = len(pdf_document)
            
            # Load YOLO model
            self.progress.emit("Loading YOLOv8 model...")
            yolo_model = YOLO('stage4_best.pt')
            
            # Load class mapping
            with open('stage4_class_mapping.json', 'r') as f:
                class_data = json.load(f)
                class_mapping = class_data['class_mapping']
            
            all_results = {
                'metadata': {
                    'pdf_path': self.pdf_path,
                    'total_pages': total_pages,
                    'detection_timestamp': datetime.now().isoformat(),
                    'dpi': self.dpi,
                    'yolo_confidence': self.yolo_confidence,
                    'model_info': class_data.get('model_info', {})
                },
                'pages': {}
            }
            
            detector = MeasureDetector(debug=False)
            
            # Track time signature across pages
            global_time_signature = {'signature': '4/4', 'beats_per_measure': 4}  # Default
            
            for page_num in range(total_pages):
                self.progress.emit(f"Processing page {page_num + 1}/{total_pages}...")
                
                try:
                    # Measure detection
                    measure_results = detector.detect_measures_from_pdf(
                        self.pdf_path, page_num, self.dpi, self.use_alt_preprocessing
                    )
                    
                    # YOLO detection for music symbols
                    page = pdf_document.load_page(page_num)
                    mat = fitz.Matrix(self.dpi/72, self.dpi/72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_data = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                    img_rgb = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    
                    self.progress.emit(f"Running YOLO detection on page {page_num + 1}...")
                    yolo_results = yolo_model(img_rgb, conf=self.yolo_confidence, verbose=False)
                    
                    # Process YOLO detections
                    yolo_detections = []
                    time_signatures = []  # Track time signatures for analysis
                    
                    for result in yolo_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                cls = int(box.cls[0].cpu().numpy())
                                class_name = class_mapping.get(str(cls), f'class_{cls}')
                                
                                # Convert bbox coordinates to ratios (0-1)
                                img_height, img_width = img_rgb.shape[:2]
                                detection = {
                                    'class_id': cls,
                                    'class_name': class_name,
                                    'confidence': float(conf),
                                    'bbox': [float(x1)/img_width, float(y1)/img_height, float(x2)/img_width, float(y2)/img_height]
                                }
                                
                                yolo_detections.append(detection)
                                
                                # Collect time signatures for analysis
                                if 'timeSig' in class_name:
                                    time_signatures.append(detection)
                    
                    # Convert numpy types to Python types for JSON serialization
                    def convert_numpy_types(obj):
                        """Recursively convert numpy types to Python types"""
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, list):
                            return [convert_numpy_types(item) for item in obj]
                        elif isinstance(obj, dict):
                            return {key: convert_numpy_types(value) for key, value in obj.items()}
                        return obj
                    
                    # Analyze time signature for this page
                    page_time_signature = self._analyze_time_signature(time_signatures) if time_signatures else None
                    
                    # Find time signature positions (which measures have time signatures)
                    time_sig_positions = []
                    if time_signatures:
                        # Map time signature detections to measures using their x-coordinates
                        barlines = measure_results.get('barlines', [])
                        if barlines:
                            for time_sig in time_signatures:
                                x_coord = time_sig.get('x', 0)
                                # Find which measure this time signature belongs to
                                for measure_idx, barline in enumerate(barlines[:-1]):  # Exclude last barline
                                    next_barline = barlines[measure_idx + 1] if measure_idx + 1 < len(barlines) else float('inf')
                                    if barline <= x_coord < next_barline:
                                        time_sig_positions.append({
                                            'measure_index': measure_idx,
                                            'time_signature': self._analyze_time_signature([time_sig])
                                        })
                                        break
                    
                    # Generate measure-specific time signature info with change tracking
                    measure_count = measure_results.get('measure_count', 0)
                    measure_time_signatures = []
                    current_time_sig = global_time_signature.copy()
                    
                    for i in range(measure_count):
                        # Check if this measure has a time signature change
                        for time_sig_pos in time_sig_positions:
                            if time_sig_pos['measure_index'] == i and time_sig_pos['time_signature']:
                                current_time_sig = time_sig_pos['time_signature']
                                global_time_signature = current_time_sig  # Update global for subsequent pages
                                logger.debug(f"Page {page_num + 1}, Measure {i + 1} - Time signature changed to: {current_time_sig['signature']}")
                                break
                        
                        measure_time_signatures.append({
                            'measure_number': i + 1,
                            'time_signature': current_time_sig.copy()
                        })
                    
                    if page_time_signature:
                        logger.debug(f"Page {page_num + 1} - Found time signature: {page_time_signature['signature']} at {len(time_sig_positions)} position(s)")
                    else:
                        logger.debug(f"Page {page_num + 1} - Using previous time signature: {global_time_signature['signature']}")
                    
                    # Combine results for this page
                    page_results = {
                        'page_number': page_num + 1,
                        'measures': convert_numpy_types({
                            'staff_lines': measure_results.get('staff_lines', []),
                            'barlines': measure_results.get('barlines', []),
                            'measure_count': measure_count,
                            'staff_systems': measure_results.get('staff_systems', []),
                            'system_groups': measure_results.get('system_groups', []),
                            'barlines_with_systems': measure_results.get('barlines_with_systems', []),
                            'detected_brackets': measure_results.get('detected_brackets', []),
                            'bracket_candidates': measure_results.get('bracket_candidates', []),
                            'measure_time_signatures': measure_time_signatures  # Per-measure time signature info
                        }),
                        'symbols': {
                            'detections': yolo_detections,
                            'detection_count': len(yolo_detections),
                            'time_signatures': time_signatures,
                            'detected_time_signature': page_time_signature,
                            'effective_time_signature': global_time_signature  # Time signature used for this page
                        }
                    }
                    
                    all_results['pages'][str(page_num + 1)] = page_results
                    self.page_completed.emit(page_num, page_results)
                    
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()
                    self.progress.emit(f"Error on page {page_num + 1}: {str(e)}")
                    logger.error(f"DETAILED ERROR for page {page_num + 1}:")
                    logger.error(error_detail)
                    logger.error("-" * 50)
                    continue
            
            pdf_document.close()
            
            # Save results to JSON (same directory and name as PDF, but with .json extension)
            pdf_dir = os.path.dirname(self.pdf_path)
            pdf_basename = os.path.splitext(os.path.basename(self.pdf_path))[0]
            output_filename = os.path.join(pdf_dir, f"{pdf_basename}.json")
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            self.progress.emit(f"Results saved to {output_filename}")
            self.finished.emit(all_results)
            
        except Exception as e:
            self.error.emit(f"Detection failed: {str(e)}")
    
    def _analyze_time_signature(self, time_signatures):
        """Analyze detected time signatures and return the most likely one"""
        if not time_signatures:
            return None
        
        # Count occurrences of each time signature type
        sig_counts = {}
        for sig in time_signatures:
            class_name = sig['class_name']
            if class_name not in sig_counts:
                sig_counts[class_name] = []
            sig_counts[class_name].append(sig)
        
        # Find the most common time signature
        if not sig_counts:
            return None
        
        most_common = max(sig_counts.keys(), key=lambda k: len(sig_counts[k]))
        
        # Map class names to time signature info
        time_sig_mapping = {
            'timeSig4': {'signature': '4/4', 'beats_per_measure': 4},
            'timeSigCommon': {'signature': '4/4', 'beats_per_measure': 4}  # Common time = 4/4
        }
        
        return time_sig_mapping.get(most_common, {
            'signature': most_common.replace('timeSig', ''),
            'beats_per_measure': 4  # Default to 4 if unknown
        })


class YOLODetectionThread(QThread):
    """Thread for running YOLOv8 music symbol detection"""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_array, model_path="stage4_best.pt", conf_threshold=0.25):
        super().__init__()
        self.image_array = image_array
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
    def run(self):
        try:
            if not YOLO_AVAILABLE:
                self.error.emit("YOLOv8 is not installed. Please install ultralytics package.")
                return
            
            self.progress.emit("Loading YOLOv8 model...")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                self.error.emit(f"Model file not found: {self.model_path}")
                return
            
            model = YOLO(self.model_path)
            
            self.progress.emit("Detecting music symbols...")
            
            # Convert image array to proper format if needed
            if len(self.image_array.shape) == 2:
                # Grayscale to RGB
                image_rgb = cv2.cvtColor(self.image_array, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = self.image_array
            
            # Run detection
            results = model(image_rgb, conf=self.conf_threshold)
            
            # Process results
            detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    class_id = int(box.cls)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf)
                    
                    detections.append({
                        "class_id": class_id,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf
                    })
            
            # Load class names
            class_names = {}
            class_file = Path("stage4_classes.txt")
            if class_file.exists():
                with open(class_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(': ')
                        if len(parts) == 2:
                            class_names[int(parts[0])] = parts[1]
            
            # Add class names to detections
            for det in detections:
                det["class_name"] = class_names.get(det["class_id"], f"class_{det['class_id']}")
            
            result_dict = {
                "detections": detections,
                "num_detections": len(detections),
                "class_names": class_names
            }
            
            self.progress.emit(f"Detected {len(detections)} music symbols")
            self.finished.emit(result_dict)
            
        except Exception as e:
            self.error.emit(str(e))


class YOLOClassFilterDialog(QDialog):
    """Dialog for filtering YOLO detection classes"""
    
    def __init__(self, class_names, visible_classes, parent=None):
        super().__init__(parent)
        self.class_names = class_names  # {id: name}
        self.visible_classes = visible_classes.copy()  # Set of visible class IDs
        
        self.setWindowTitle("Music Symbol Class Filter")
        self.setModal(True)
        self.setMinimumSize(400, 500)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Select which music symbols to display:")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        self.checkboxes = {}
        
        # Group classes by category
        categories = {
            "Clefs": ["clef"],
            "Time Signatures": ["timeSig"],
            "Note Heads": ["notehead"],
            "Flags": ["flag"],
            "Rests": ["rest"],
            "Accidentals & Keys": ["accidental", "key"],
            "Structure": ["staff", "brace"],
            "Other": ["tuplet"]
        }
        
        # Add checkboxes by category
        for category, keywords in categories.items():
            # Category header
            category_label = QLabel(f"{category}:")
            category_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            category_label.setStyleSheet("color: #333; margin-top: 10px;")
            scroll_layout.addWidget(category_label)
            
            # Find classes for this category
            category_classes = []
            for class_id, class_name in self.class_names.items():
                if any(keyword.lower() in class_name.lower() for keyword in keywords):
                    category_classes.append((class_id, class_name))
            
            # Sort classes by name
            category_classes.sort(key=lambda x: x[1])
            
            # Add checkboxes
            for class_id, class_name in category_classes:
                checkbox = QCheckBox(f"{class_name} (ID: {class_id})")
                checkbox.setChecked(class_id in self.visible_classes)
                checkbox.setStyleSheet("margin-left: 20px;")
                self.checkboxes[class_id] = checkbox
                scroll_layout.addWidget(checkbox)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_none)
        button_layout.addWidget(select_none_btn)
        
        button_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def select_all(self):
        """Select all classes"""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)
    
    def select_none(self):
        """Deselect all classes"""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
    
    def get_visible_classes(self):
        """Get set of selected class IDs"""
        visible = set()
        for class_id, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                visible.add(class_id)
        return visible


class Metronome:
    """Simple metronome class with fallback to system bell"""
    
    def __init__(self):
        self.use_pygame = False
        self.use_system_bell = False
        self.tick_sound = None
        self.temp_files = []  # Track temporary files for cleanup
        
        # Try pygame first
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
                self.use_pygame = True
                self._generate_tick_sound()
                logger.info("Metronome: Using pygame audio")
            except Exception as e:
                logger.warning(f"Failed to initialize pygame mixer: {e}")
                self.use_pygame = False
        
        # Fallback to system bell
        if not self.use_pygame:
            self.use_system_bell = True
            logger.info("Metronome: Using system bell as fallback")
    
    @property
    def is_initialized(self):
        return self.use_pygame or self.use_system_bell
    
    def _generate_tick_sound(self):
        """Generate a clean wood block-like tick sound"""
        if not self.use_pygame:
            return
        
        # Generate a short, percussive tick
        duration = 0.05  # Very short
        sample_rate = 44100
        
        # Mix of frequencies for wood block-like sound
        frequencies = [800, 1200, 2000]  # Harmonics
        weights = [1.0, 0.5, 0.25]      # Volume weights
        
        frames = int(duration * sample_rate)
        samples = []
        
        for i in range(frames):
            t = float(i) / sample_rate
            
            # Exponential decay envelope (percussive)
            envelope = math.exp(-t * 50)  # Fast decay
            
            # Mix frequencies
            sample_value = 0
            for freq, weight in zip(frequencies, weights):
                sample_value += weight * math.sin(2 * math.pi * freq * t)
            
            # Apply envelope and scale
            amplitude = 8192  # Conservative amplitude
            sample = int(amplitude * envelope * sample_value)
            
            # Clamp to prevent overflow
            sample = max(-32767, min(32767, sample))
            samples.extend([sample, sample])  # Stereo
        
        # Create temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.temp_files.append(temp_file.name)
        
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
        
        # Load sound with conservative volume
        try:
            self.tick_sound = pygame.mixer.Sound(temp_file.name)
            self.tick_sound.set_volume(0.4)  # Conservative volume to prevent distortion
        except Exception as e:
            logger.warning(f"Failed to load tick sound: {e}")
            self.tick_sound = None
    
    def play_tick(self):
        """Play a single tick sound"""
        if self.use_pygame and self.tick_sound:
            try:
                self.tick_sound.play()
            except Exception as e:
                logger.warning(f"Failed to play pygame tick: {e}")
        elif self.use_system_bell:
            try:
                # Use system bell as fallback
                import sys
                if sys.platform == "win32":
                    import winsound
                    winsound.Beep(800, 100)  # 800Hz, 100ms
                else:
                    # Unix/Linux systems
                    logger.debug("Terminal bell played")  # Terminal bell
            except Exception as e:
                logger.warning(f"Failed to play system bell: {e}")
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        self.temp_files.clear()


class ScoreImageWidget(QLabel):
    """Custom widget for displaying score with overlay"""
    
    def __init__(self):
        super().__init__()
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.staff_lines = []
        self.barlines = []
        self.barline_candidates = []
        self.show_staff_lines = True
        self.show_barlines = True
        self.show_candidates = False
        self.show_system_groups = True  # Show system group clustering
        self.show_measure_boxes = True  # Show measure bounding boxes
        self.show_bracket_candidates = False  # Show bracket candidates
        self.show_brackets = True  # Show verified brackets
        self.show_yolo_detections = False  # Show YOLO music symbol detections
        self.show_current_measure = False  # Show current playing measure highlight
        self.measure_boxes = []  # List of measure bounding boxes
        self.bracket_candidates = []  # List of bracket candidate lines
        self.verified_brackets = []  # List of verified brackets
        self.yolo_detections = []  # List of YOLO detections
        self.yolo_class_names = {}  # YOLO class ID to name mapping
        self.yolo_visible_classes = set()  # Set of visible class IDs
        self.current_measure_index = -1  # Currently playing measure index
        self.measure_count = 0
        self.staff_systems = []
        self.system_groups = []
        
        # Manual barline addition
        self.add_barline_mode = False  # Manual barline addition mode
        self.auto_fit = True  # Track if auto-fit is enabled
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
    def set_image(self, image_array):
        """Set the image from numpy array"""
        height, width = image_array.shape[:2]
        
        # Convert to QPixmap
        if len(image_array.shape) == 2:
            # Grayscale
            bytes_per_line = width
            q_image = QImage(image_array.data.tobytes(), width, height, 
                            bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # Color
            bytes_per_line = 3 * width
            q_image = QImage(image_array.data.tobytes(), width, height,
                            bytes_per_line, QImage.Format.Format_RGB888)
            
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.update()
        
    def set_detection_results(self, staff_lines, barlines, measure_count, barline_candidates=None, 
                             staff_lines_with_ranges=None, barlines_with_systems=None,
                             staff_systems=None, system_groups=None, measure_boxes=None,
                             bracket_candidates=None, verified_brackets=None):
        """Set the detection results for overlay"""
        self.staff_lines = staff_lines
        self.barlines = barlines
        self.measure_count = measure_count
        self.barline_candidates = barline_candidates or []
        self.staff_lines_with_ranges = staff_lines_with_ranges or []
        self.barlines_with_systems = barlines_with_systems or []
        self.staff_systems = staff_systems or []
        self.system_groups = system_groups or []
        self.measure_boxes = measure_boxes or []
        self.bracket_candidates = bracket_candidates or []
        self.verified_brackets = verified_brackets or []
        self.update()
        
    def set_scale(self, scale_factor):
        """Set the scale factor for display"""
        self.scale_factor = scale_factor
        self.update()
        
    def toggle_staff_lines(self, show):
        """Toggle staff line overlay"""
        self.show_staff_lines = show
        self.update()
        
    def toggle_barlines(self, show):
        """Toggle barline overlay"""
        self.show_barlines = show
        self.update()
        
    def toggle_candidates(self, show):
        """Toggle barline candidates overlay"""
        self.show_candidates = show
        self.update()
        
    def toggle_system_groups(self, show):
        """Toggle system group clustering overlay"""
        self.show_system_groups = show
        self.update()
        
    def toggle_bracket_candidates(self, show):
        """Toggle bracket candidate overlay"""
        self.show_bracket_candidates = show
        self.update()
        
    def toggle_brackets(self, show):
        """Toggle verified bracket overlay"""
        self.show_brackets = show
        logger.debug(f"toggle_brackets: show={show}, verified_brackets count={len(self.verified_brackets) if self.verified_brackets else 0}")
        self.update()
    
    def clear_detections(self):
        """Clear all detection results and overlays"""
        self.staff_lines = []
        self.barlines = []
        self.measure_count = 0
        self.barline_candidates = []
        self.staff_lines_with_ranges = []
        self.barlines_with_systems = []
        self.staff_systems = []
        self.system_groups = []
        self.measure_boxes = []
        self.bracket_candidates = []
        self.verified_brackets = []
        self.yolo_detections = []
        self.yolo_class_names = {}
        
        # Clear highlighting
        self.current_measure_index = None
        self.show_current_measure_highlight = False
        
        # Force update
        self.update()
    
    def set_yolo_detections(self, detections, class_names):
        """Set YOLO detection results"""
        self.yolo_detections = detections
        self.yolo_class_names = class_names
        # Initially show all classes
        self.yolo_visible_classes = set(class_names.keys())
        self.update()
    
    def set_yolo_visible_classes(self, visible_classes):
        """Set which YOLO classes are visible"""
        self.yolo_visible_classes = visible_classes
        self.update()
    
    def toggle_yolo_detections(self, show):
        """Toggle YOLO detections overlay"""
        self.show_yolo_detections = show
        self.update()
    
    def set_current_measure(self, measure_index):
        """Set the currently playing measure"""
        self.current_measure_index = measure_index
        self.update()
    
    def toggle_current_measure_highlight(self, show):
        """Toggle current measure highlight"""
        self.show_current_measure = show
        self.update()
        
    def calculate_fit_scale(self):
        """Calculate scale to fit image in widget"""
        if self.original_pixmap:
            # Get available space from parent widget
            parent_widget = self.parent()
            if parent_widget:
                available_size = parent_widget.size()
            else:
                available_size = self.size()
                
            img_size = self.original_pixmap.size()
            
            # Calculate scale to fit with margin
            margin = 40
            scale_x = (available_size.width() - margin) / img_size.width()
            scale_y = (available_size.height() - margin) / img_size.height()
            
            return min(scale_x, scale_y, 2.0)  # Max 200% zoom
        return 1.0
        
    def set_auto_fit(self, auto_fit):
        """Set auto-fit mode"""
        self.auto_fit = auto_fit
        if auto_fit and self.original_pixmap:
            self.apply_auto_fit()
            
    def apply_auto_fit(self):
        """Apply auto-fit scaling"""
        if self.original_pixmap:
            fit_scale = self.calculate_fit_scale()
            self.scale_factor = fit_scale
            self.update()
            return fit_scale
        return 1.0
        
    def paintEvent(self, event):
        """Paint event with overlay"""
        # Create painter for the widget
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.original_pixmap is None:
            return
            
        # Scale the pixmap
        scaled_pixmap = self.original_pixmap.scaled(
            int(self.original_pixmap.width() * self.scale_factor),
            int(self.original_pixmap.height() * self.scale_factor),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Calculate position to center the image
        widget_rect = self.rect()
        pixmap_rect = scaled_pixmap.rect()
        
        x_offset = (widget_rect.width() - pixmap_rect.width()) // 2
        y_offset = (widget_rect.height() - pixmap_rect.height()) // 2
        
        # 1. Draw the base image first
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        # 2. Draw overlays on top of the image
        # Translate painter coordinate system to image coordinate system
        painter.save()  # Save current state
        painter.translate(x_offset, y_offset)  # Move origin to image top-left
        
        # Draw staff lines (only in their actual ranges)
        if self.show_staff_lines and hasattr(self, 'staff_lines_with_ranges') and self.staff_lines_with_ranges:
            pen = QPen(QColor(0, 0, 255, 128), 2)  # Blue with transparency
            painter.setPen(pen)
            for line_info in self.staff_lines_with_ranges:
                y_scaled = int(line_info['y'] * self.scale_factor)
                x_start_scaled = int(line_info['x_start'] * self.scale_factor)
                x_end_scaled = int(line_info['x_end'] * self.scale_factor)
                painter.drawLine(x_start_scaled, y_scaled, x_end_scaled, y_scaled)
        elif self.show_staff_lines and self.staff_lines:
            # Fallback to full-width lines if range info not available
            pen = QPen(QColor(0, 0, 255, 128), 2)  # Blue with transparency
            painter.setPen(pen)
            for y in self.staff_lines:
                y_scaled = int(y * self.scale_factor)
                painter.drawLine(0, y_scaled, scaled_pixmap.width(), y_scaled)
                
        # Draw barline candidates
        if self.show_candidates and self.barline_candidates:
            pen = QPen(QColor(255, 165, 0, 100), 2)  # Orange with transparency
            painter.setPen(pen)
            for x in self.barline_candidates:
                x_scaled = int(x * self.scale_factor)
                painter.drawLine(x_scaled, 0, x_scaled, scaled_pixmap.height())
                
        # Draw barlines (within staff system boundaries)  
        if self.show_barlines and self.barlines:
            pen = QPen(QColor(255, 0, 0, 180), 3)  # Red with transparency
            painter.setPen(pen)
            font = QFont("Arial", 12)
            painter.setFont(font)
            
            # Check if we have staff system information
            if hasattr(self, 'barlines_with_systems') and self.barlines_with_systems:
                logger.debug(f"Total barlines_with_systems: {len(self.barlines_with_systems)}")
                # Cluster barline, 일반 barline, manual barline 구분해서 그리기
                cluster_barlines = []
                regular_barlines = []
                manual_barlines = []
                
                for bl in self.barlines_with_systems:
                    if bl.get('type') == 'manual':
                        manual_barlines.append(bl)
                    elif bl.get('is_cluster_barline', False):
                        cluster_barlines.append(bl)
                    else:
                        regular_barlines.append(bl)
                
                # Debug: Print barline counts
                logger.debug(f"Barlines count: {len(cluster_barlines)} cluster, {len(regular_barlines)} regular, {len(manual_barlines)} manual")
                if manual_barlines:
                    logger.debug(f"Manual barlines: {manual_barlines}")
                
                # 1. Cluster barlines 먼저 그리기 (cluster 전체를 관통하는 긴 barline)
                if cluster_barlines:
                    pen = QPen(QColor(255, 0, 0, 220), 4)  # 더 진한 빨간색, 더 굵은 선
                    painter.setPen(pen)
                    font = QFont("Arial", 14, QFont.Weight.Bold)
                    painter.setFont(font)
                    
                    cluster_barline_count = 0
                    for bl in cluster_barlines:
                        cluster_barline_count += 1
                        x_scaled = int(bl['x'] * self.scale_factor)
                        y_start = max(0, int(bl['y_start'] * self.scale_factor))
                        y_end = min(scaled_pixmap.height(), int(bl['y_end'] * self.scale_factor))
                        
                        # Cluster 전체를 관통하는 긴 barline 그리기
                        painter.drawLine(x_scaled, y_start, x_scaled, y_end)
                
                # 2. 일반 barlines 그리기 (system별 개별 barline)
                if regular_barlines:
                    pen = QPen(QColor(255, 100, 100, 150), 2)  # 연한 빨간색, 얇은 선
                    painter.setPen(pen)
                    font = QFont("Arial", 10)
                    painter.setFont(font)
                    
                    system_barline_counts = {}
                    for bl in regular_barlines:
                        x_scaled = int(bl['x'] * self.scale_factor)
                        y_start = max(0, int(bl['y_start'] * self.scale_factor))
                        y_end = min(scaled_pixmap.height(), int(bl['y_end'] * self.scale_factor))
                        
                        # 개별 system barline 그리기
                        painter.drawLine(x_scaled, y_start, x_scaled, y_end)
                        
            else:
                # 기존 방식 (호환성)
                staff_top = min(self.staff_lines) if self.staff_lines else 0
                staff_bottom = max(self.staff_lines) if self.staff_lines else scaled_pixmap.height()
                margin = 20  # Small margin beyond staff
                
                for i, x in enumerate(self.barlines):
                    x_scaled = int(x * self.scale_factor)
                    y_start = int((staff_top - margin) * self.scale_factor)
                    y_end = int((staff_bottom + margin) * self.scale_factor)
                    
                    # Ensure we don't go beyond image bounds
                    y_start = max(0, y_start)
                    y_end = min(scaled_pixmap.height(), y_end)
                    
                    painter.drawLine(x_scaled, y_start, x_scaled, y_end)
                    
                    # Draw barline number
                    painter.drawText(x_scaled - 15, y_start + 30, f"{i+1}")
        
        # Draw manual barlines (same style as regular barlines but in blue)
        if self.show_barlines and manual_barlines:
            logger.debug(f"Drawing {len(manual_barlines)} manual barlines")
            painter.setPen(QPen(QColor(0, 150, 255), 2))  # Blue color for manual barlines
            
            for i, bl in enumerate(manual_barlines):
                logger.debug(f"Drawing manual barline {i}: {bl}")
                # Use the same drawing logic as regular barlines
                if 'y_start' in bl and 'y_end' in bl:
                    # Manual barline with proper y_start/y_end coordinates
                    x_scaled = int(bl['x'] * self.scale_factor)
                    y_start = max(0, int(bl['y_start'] * self.scale_factor))
                    y_end = min(scaled_pixmap.height(), int(bl['y_end'] * self.scale_factor))
                    
                    logger.debug(f"Drawing manual barline at x={x_scaled}, y={y_start}-{y_end}")
                    painter.drawLine(x_scaled, y_start, x_scaled, y_end)
                    
                    # Draw manual barline indicator (M)
                    painter.drawText(x_scaled - 15, y_start + 15, "M")
                else:
                    # Fallback: manual barline without proper coordinates, try to find system group
                    x_ratio = bl.get('x', bl.get('x_ratio', 0))
                    y_ratio = bl.get('y', bl.get('y_ratio', 0))
                    
                    # Find which system group this manual barline belongs to
                    target_group_idx = None
                    if self.system_groups:
                        for group_idx, system_indices in enumerate(self.system_groups):
                            for sys_idx in system_indices:
                                if sys_idx < len(self.staff_systems):
                                    system = self.staff_systems[sys_idx]
                                    if system['top'] <= y_ratio <= system['bottom']:
                                        target_group_idx = group_idx
                                        break
                            if target_group_idx is not None:
                                break
                    
                    # Draw manual barline only in the target system group
                    if target_group_idx is not None:
                        x_scaled = int(x_ratio * original_width * scale_factor)
                        
                        # Get Y range for the target system group
                        group_systems = [self.staff_systems[i] for i in self.system_groups[target_group_idx] 
                                       if i < len(self.staff_systems)]
                        if group_systems:
                            y_start = int(min(sys['top'] for sys in group_systems) * original_height * scale_factor)
                            y_end = int(max(sys['bottom'] for sys in group_systems) * original_height * scale_factor)
                            
                            # Ensure coordinates are within bounds
                            y_start = max(0, y_start)
                            y_end = min(scaled_pixmap.height(), y_end)
                            
                            painter.drawLine(x_scaled, y_start, x_scaled, y_end)
                            
                            # Draw manual barline indicator (M)
                            painter.drawText(x_scaled - 15, y_start + 15, "M")
        
        # Draw system group clustering visualization
        if self.show_system_groups and self.staff_systems and self.system_groups:
            # Define colors for different system groups
            group_colors = [
                QColor(255, 100, 100, 100),  # Light red
                QColor(100, 255, 100, 100),  # Light green  
                QColor(100, 100, 255, 100),  # Light blue
                QColor(255, 255, 100, 100),  # Light yellow
                QColor(255, 100, 255, 100),  # Light magenta
                QColor(100, 255, 255, 100),  # Light cyan
            ]
            
            for group_idx, system_indices in enumerate(self.system_groups):
                if not system_indices:
                    continue
                    
                color = group_colors[group_idx % len(group_colors)]
                pen = QPen(color, 3)
                painter.setPen(pen)
                
                # Calculate group bounding box
                group_top = float('inf')
                group_bottom = float('-inf')
                
                for sys_idx in system_indices:
                    if sys_idx < len(self.staff_systems):
                        system = self.staff_systems[sys_idx]
                        group_top = min(group_top, system['top'])
                        group_bottom = max(group_bottom, system['bottom'])
                
                if group_top == float('inf'):
                    continue
                
                # Scale coordinates
                group_top_scaled = int(group_top * self.scale_factor)
                group_bottom_scaled = int(group_bottom * self.scale_factor)
                
                # Draw group boundary rectangle
                margin = 10
                painter.drawRect(
                    margin, 
                    group_top_scaled - margin,
                    scaled_pixmap.width() - 2 * margin,
                    group_bottom_scaled - group_top_scaled + 2 * margin
                )
                
                # Draw group label
                font = QFont("Arial", 14, QFont.Weight.Bold)
                painter.setFont(font)
                painter.setPen(QPen(color.darker(150), 2))
                
                group_label = f"Group {group_idx + 1} ({len(system_indices)} systems)"
                painter.drawText(
                    margin + 10, 
                    group_top_scaled - margin + 20, 
                    group_label
                )
                
        # Draw measure bounding boxes
        logger.debug(f"paintEvent: show_measure_boxes={self.show_measure_boxes}, measure_boxes count={len(self.measure_boxes) if self.measure_boxes else 0}")
        if self.show_measure_boxes and self.measure_boxes:
            logger.debug(f"paintEvent: Drawing {len(self.measure_boxes)} measure boxes")
            pen = QPen(QColor(0, 255, 0, 150), 2)  # Green with transparency
            painter.setPen(pen)
            font = QFont("Arial", 10)
            painter.setFont(font)
            
            for i, box in enumerate(self.measure_boxes):
                # Convert ratio coordinates to actual pixel coordinates, then scale
                if self.original_pixmap:
                    orig_width = self.original_pixmap.width()
                    orig_height = self.original_pixmap.height()
                    
                    x_pixel = box['x'] * orig_width
                    y_pixel = box['y'] * orig_height
                    width_pixel = box['width'] * orig_width
                    height_pixel = box['height'] * orig_height
                    
                    x_scaled = int(x_pixel * self.scale_factor)
                    y_scaled = int(y_pixel * self.scale_factor)
                    width_scaled = int(width_pixel * self.scale_factor)
                    height_scaled = int(height_pixel * self.scale_factor)
                else:
                    # Fallback: treat as pixel coordinates
                    x_scaled = int(box['x'] * self.scale_factor)
                    y_scaled = int(box['y'] * self.scale_factor)
                    width_scaled = int(box['width'] * self.scale_factor)
                    height_scaled = int(box['height'] * self.scale_factor)
                
                if i < 3:  # Log first 3 boxes for debugging
                    logger.debug(f"  Box {i}: orig=({box['x']},{box['y']},{box['width']}x{box['height']}), "
                          f"scaled=({x_scaled},{y_scaled},{width_scaled}x{height_scaled}), scale={self.scale_factor}")
                
                # Draw rectangle
                rect = QRect(x_scaled, y_scaled, width_scaled, height_scaled)
                painter.drawRect(rect)
                
                # Draw measure ID
                measure_id = box.get('measure_id', f'M{i+1}')
                painter.drawText(x_scaled + 5, y_scaled + 15, measure_id)
        
        # Draw bracket candidates
        if self.show_bracket_candidates and self.bracket_candidates:
            pen = QPen(QColor(255, 255, 0, 120), 3)  # Yellow with transparency
            painter.setPen(pen)
            for candidate in self.bracket_candidates:
                try:
                    if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                        x1, y1, x2, y2 = candidate
                        x1_scaled = int(x1 * self.scale_factor)
                        y1_scaled = int(y1 * self.scale_factor)
                        x2_scaled = int(x2 * self.scale_factor)
                        y2_scaled = int(y2 * self.scale_factor)
                        painter.drawLine(x1_scaled, y1_scaled, x2_scaled, y2_scaled)
                    else:
                        logger.debug(f"Skipping invalid candidate: {candidate}")
                except Exception as e:
                    logger.debug(f"Error drawing candidate {candidate}: {e}")
        
        # Draw verified brackets
        logger.debug(f"paintEvent: show_brackets={self.show_brackets}, verified_brackets count={len(self.verified_brackets) if self.verified_brackets else 0}")
        if self.show_brackets and self.verified_brackets:
            logger.debug(f"paintEvent: Drawing {len(self.verified_brackets)} verified brackets")
            pen = QPen(QColor(255, 0, 255, 180), 4)  # Magenta with transparency
            painter.setPen(pen)
            font = QFont("Arial", 10)
            painter.setFont(font)
            
            for i, bracket in enumerate(self.verified_brackets):
                if isinstance(bracket, dict):
                    # Bracket info dictionary format
                    x = bracket.get('x', 0)
                    y_start = bracket.get('y_start', 0)
                    y_end = bracket.get('y_end', 0)
                    
                    x_scaled = int(x * self.scale_factor)
                    y_start_scaled = int(y_start * self.scale_factor)
                    y_end_scaled = int(y_end * self.scale_factor)
                    
                    # Draw main vertical line
                    painter.drawLine(x_scaled, y_start_scaled, x_scaled, y_end_scaled)
                    
                    # Draw top horizontal line (bracket corner)
                    painter.drawLine(x_scaled, y_start_scaled, x_scaled + 15, y_start_scaled)
                    
                    # Draw bottom horizontal line (bracket corner)
                    painter.drawLine(x_scaled, y_end_scaled, x_scaled + 15, y_end_scaled)
                    
                    # Draw bracket label
                    systems = bracket.get('covered_staff_system_indices', [])
                    label = f"B{i+1}({len(systems)})"
                    painter.drawText(x_scaled + 20, y_start_scaled + 15, label)
                else:
                    # Raw coordinate format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = bracket
                    x1_scaled = int(x1 * self.scale_factor)
                    y1_scaled = int(y1 * self.scale_factor)
                    x2_scaled = int(x2 * self.scale_factor)
                    y2_scaled = int(y2 * self.scale_factor)
                    painter.drawLine(x1_scaled, y1_scaled, x2_scaled, y2_scaled)
        
        # Draw measure count
        if self.measure_count > 0:
            pen = QPen(QColor(0, 255, 0), 3)  # Green
            painter.setPen(pen)
            font = QFont("Arial", 16, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(20, scaled_pixmap.height() - 20, 
                           f"Measures: {self.measure_count}")
        
        # Draw YOLO detections
        if self.show_yolo_detections and self.yolo_detections:
            # High saturation colors for better visibility
            category_colors = {
                "notehead": QColor(0, 100, 255, 220),      # Bright Blue
                "flag": QColor(0, 255, 0, 220),            # Bright Green  
                "rest": QColor(255, 0, 100, 220),          # Bright Red-Pink
                "clef": QColor(255, 200, 0, 220),          # Bright Gold
                "key": QColor(255, 0, 255, 220),           # Bright Magenta
                "accidental": QColor(0, 255, 255, 220),    # Bright Cyan
                "timeSig": QColor(255, 128, 0, 220),       # Bright Orange
                "staff": QColor(150, 255, 0, 220),         # Bright Lime
                "brace": QColor(255, 100, 0, 220),         # Bright Orange-Red
                "tuplet": QColor(200, 0, 255, 220)         # Bright Purple
            }
            
            for det in self.yolo_detections:
                class_id = det["class_id"]
                
                # Skip if this class is not visible
                if class_id not in self.yolo_visible_classes:
                    continue
                
                x1, y1, x2, y2 = det["bbox"]
                class_name = det.get("class_name", f"class_{class_id}")
                
                # Scale coordinates
                x1_scaled = int(x1 * self.scale_factor)
                y1_scaled = int(y1 * self.scale_factor)
                x2_scaled = int(x2 * self.scale_factor)
                y2_scaled = int(y2 * self.scale_factor)
                
                # Determine color based on class name
                color = QColor(0, 150, 255, 220)  # Default bright blue
                for category, cat_color in category_colors.items():
                    if category.lower() in class_name.lower():
                        color = cat_color
                        break
                
                # Draw bounding box with thicker line for better visibility
                pen = QPen(color, 3)
                painter.setPen(pen)
                rect = QRect(x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled)
                painter.drawRect(rect)
        
        # Draw current measure highlight using measure_boxes
        if (self.show_current_measure and 
            self.current_measure_index >= 0 and 
            hasattr(self, 'measure_boxes') and self.measure_boxes):
            
            # Find the target measure number from the current measure index
            if self.current_measure_index < len(self.measure_boxes):
                current_box = self.measure_boxes[self.current_measure_index]
                target_measure_number = current_box.get('measure_index', 1)
                target_group_index = current_box.get('system_group_index', 0)
                
                # Find all measure boxes with the same measure number in the SAME system group
                matching_measures = [box for box in self.measure_boxes 
                                   if (box.get('measure_index') == target_measure_number and 
                                       box.get('system_group_index') == target_group_index)]
                
                if matching_measures:
                    print(f"DEBUG: Highlighting {len(matching_measures)} measures with number {target_measure_number} in group {target_group_index + 1}")
                    
                    # Calculate combined bounding box for all matching measures
                    min_x = min(box['x'] for box in matching_measures)
                    min_y = min(box['y'] for box in matching_measures)
                    max_x = max(box['x'] + box['width'] for box in matching_measures)
                    max_y = max(box['y'] + box['height'] for box in matching_measures)
                    
                    # Scale coordinates for the combined bounding box
                    x_start_scaled = int(min_x * self.scale_factor)
                    y_top_scaled = int(min_y * self.scale_factor)
                    x_end_scaled = int(max_x * self.scale_factor)
                    y_bottom_scaled = int(max_y * self.scale_factor)
                    
                    width_scaled = x_end_scaled - x_start_scaled
                    height_scaled = y_bottom_scaled - y_top_scaled
                    
                    # Ensure reasonable bounds
                    x_start_scaled = max(0, x_start_scaled)
                    y_top_scaled = max(0, y_top_scaled)
                    width_scaled = min(width_scaled, scaled_pixmap.width() - x_start_scaled)
                    height_scaled = min(height_scaled, scaled_pixmap.height() - y_top_scaled)
                    
                    if width_scaled > 10 and height_scaled > 10:  # Only draw if reasonable size
                        # Draw single combined highlight overlay
                        highlight_color = QColor(255, 255, 0, 60)  # Yellow with transparency
                        painter.fillRect(x_start_scaled, y_top_scaled, width_scaled, height_scaled, highlight_color)
                        
                        # Draw single combined border
                        border_pen = QPen(QColor(255, 140, 0, 220), 6)  # Orange border, thick
                        painter.setPen(border_pen)
                        rect = QRect(x_start_scaled, y_top_scaled, width_scaled, height_scaled)
                        painter.drawRect(rect)
                        
                        # Draw measure number label at top-left of combined box
                        font = QFont("Arial", 16, QFont.Weight.Bold)
                        painter.setFont(font)
                        
                        measure_text = f"♪ G{target_group_index + 1} M{target_measure_number}"
                        text_rect = painter.fontMetrics().boundingRect(measure_text)
                        
                        # Draw text background
                        text_bg_rect = QRect(x_start_scaled + 10, y_top_scaled + 10, 
                                            text_rect.width() + 20, text_rect.height() + 10)
                        painter.fillRect(text_bg_rect, QColor(255, 255, 255, 200))
                        
                        # Draw text
                        painter.setPen(QPen(QColor(255, 0, 0), 2))  # Red text
                        painter.drawText(x_start_scaled + 20, y_top_scaled + 30, measure_text)
                        
                        print(f"DEBUG: Combined bounding box: x:{x_start_scaled}-{x_end_scaled}, y:{y_top_scaled}-{y_bottom_scaled} (covers {len(matching_measures)} systems)")
                else:
                    print(f"DEBUG: No measures found with measure number {target_measure_number}")
            else:
                print(f"DEBUG: Measure index {self.current_measure_index} out of range (total: {len(self.measure_boxes)})")
            
        # Restore painter coordinate system
        painter.restore()
        painter.end()
        
    def mousePressEvent(self, event):
        """Handle mouse press events for manual barline addition"""
        logger.debug(f"mousePressEvent called: add_barline_mode={self.add_barline_mode}, button={event.button()}")
        
        if not self.add_barline_mode or not self.original_pixmap:
            logger.debug(f"mousePressEvent: ignoring - mode={self.add_barline_mode}, pixmap={self.original_pixmap is not None}")
            super().mousePressEvent(event)
            return
        
        # Handle left mouse button for adding, right for removing
        if event.button() not in [Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton]:
            logger.debug(f"mousePressEvent: unsupported button {event.button()}")
            super().mousePressEvent(event)
            return
        
        # Get click position relative to the image
        click_pos = event.position().toPoint()
        logger.debug(f"mousePressEvent: click_pos={click_pos.x()},{click_pos.y()}")
        
        # Convert click coordinates to image coordinates
        widget_rect = self.rect()
        
        # Calculate scaled pixmap size (same as in paintEvent)
        scaled_width = int(self.original_pixmap.width() * self.scale_factor)
        scaled_height = int(self.original_pixmap.height() * self.scale_factor)
        pixmap_rect = QRect(0, 0, scaled_width, scaled_height)
        
        logger.debug(f"mousePressEvent: widget_rect={widget_rect}, scaled_pixmap_rect={pixmap_rect}")
        
        if pixmap_rect.isValid():
            # Calculate image position within widget (centered) - same as in paintEvent
            x_offset = (widget_rect.width() - pixmap_rect.width()) // 2
            y_offset = (widget_rect.height() - pixmap_rect.height()) // 2
            logger.debug(f"mousePressEvent: offsets=({x_offset},{y_offset})")
            
            # Adjust click position relative to image
            img_x = click_pos.x() - x_offset
            img_y = click_pos.y() - y_offset
            logger.debug(f"mousePressEvent: img coordinates=({img_x},{img_y})")
            
            # Check if click is within image bounds
            if 0 <= img_x <= pixmap_rect.width() and 0 <= img_y <= pixmap_rect.height():
                logger.debug(f"mousePressEvent: click is within image bounds")
                # Convert to original image coordinates (unscaled)
                orig_width = self.original_pixmap.width()
                orig_height = self.original_pixmap.height()
                
                orig_x = img_x / self.scale_factor
                orig_y = img_y / self.scale_factor
                logger.debug(f"mousePressEvent: original coordinates=({orig_x},{orig_y})")
                
                # Convert to ratio coordinates (0-1)
                x_ratio = orig_x / orig_width
                y_ratio = orig_y / orig_height
                logger.debug(f"mousePressEvent: ratio coordinates=({x_ratio:.3f},{y_ratio:.3f})")
                
                if event.button() == Qt.MouseButton.LeftButton:
                    logger.debug(f"mousePressEvent: adding manual barline at x_ratio={x_ratio:.3f}")
                    # Add manual barline
                    self.add_manual_barline(x_ratio, y_ratio)
                elif event.button() == Qt.MouseButton.RightButton:
                    logger.debug(f"mousePressEvent: removing barline near x_ratio={x_ratio:.3f}")
                    # Remove nearby barline (manual or regular)
                    self.remove_barline(x_ratio)
            else:
                logger.debug(f"mousePressEvent: click outside image bounds")
        else:
            logger.debug(f"mousePressEvent: invalid pixmap_rect")
        
        super().mousePressEvent(event)
        
    def add_manual_barline(self, x_ratio, y_ratio):
        """Add a manual barline at the specified position"""
        logger.info(f"Adding manual barline at x={x_ratio:.3f}, y={y_ratio:.3f}")
        
        # Notify parent to add to detection results and update display
        parent = self.parent()
        while parent and not hasattr(parent, 'add_manual_barline_to_detection'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'add_manual_barline_to_detection'):
            parent.add_manual_barline_to_detection(x_ratio, y_ratio)
    
    def remove_manual_barline(self, x_ratio):
        """Remove manual barline near the specified x position"""
        logger.info(f"Attempting to remove manual barline near x={x_ratio:.3f}")
        
        # Notify parent to remove from detection results
        parent = self.parent()
        while parent and not hasattr(parent, 'remove_manual_barline_from_detection'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'remove_manual_barline_from_detection'):
            parent.remove_manual_barline_from_detection(x_ratio)
    
    def remove_barline(self, x_ratio):
        """Remove any barline (manual or regular) near the specified x position"""
        logger.info(f"Attempting to remove barline near x={x_ratio:.3f}")
        
        # Notify parent to remove from detection results
        parent = self.parent()
        while parent and not hasattr(parent, 'remove_barline_from_detection'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'remove_barline_from_detection'):
            parent.remove_barline_from_detection(x_ratio)


class ScoreEyeGUI(QMainWindow):
    """Main application window"""
    
    # Signal for high-precision metronome
    metronome_beat_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.pdf_path = None
        
        # Connect precision metronome signal
        self.metronome_beat_signal.connect(self.metronome_tick)
        self.pdf_document = None
        self.current_page = 0
        self.total_pages = 0
        self.detection_results = None
        self.all_pages_results = None  # Store results from "Detect All"
        
        # Metronome - High-precision threaded approach
        self.metronome = Metronome()
        self.is_playing = False
        self.metronome_thread = None
        self.metronome_stop_event = threading.Event()
        
        # Timing tracking
        self.metronome_start_time = None
        self.expected_beat_interval_ms = 500  # Will be set based on BPM
        
        # Beat tracking
        self.current_beat = 0
        self.current_measure = 0
        self.beats_per_measure = 4  # Default 4/4 time
        self.current_time_signature = None
        
        # Global time signature tracking across pages
        self.global_time_signature = {'signature': '4/4', 'beats_per_measure': 4}  # Default
        
        # System group tracking for metronome
        self.current_group_index = 0
        self.current_measure_in_group = 0
        
        # Timing accuracy tracking
        self.total_beats_played = 0
        self.expected_beat_interval_ms = 0
        
        self.init_ui()
        
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Apply auto-fit if enabled (direct update for smooth resizing)
        if (hasattr(self, 'image_widget') and 
            hasattr(self, 'zoom_slider') and 
            self.image_widget.auto_fit and 
            self.image_widget.original_pixmap):
            
            fit_scale = self.image_widget.apply_auto_fit()
            # Update zoom slider to reflect new scale
            self.zoom_slider.blockSignals(True)  # Prevent recursive updates
            self.zoom_slider.setValue(int(fit_scale * 100))
            self.zoom_label.setText(f"Zoom: {int(fit_scale * 100)}%")
            self.zoom_slider.blockSignals(False)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("ScoreEye - Measure Detection")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)
        
        # File controls
        file_group = QGroupBox("File Controls")
        file_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Load PDF")
        self.load_btn.clicked.connect(self.load_pdf)
        file_layout.addWidget(self.load_btn)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # Page controls
        page_group = QGroupBox("Page Navigation")
        page_layout = QVBoxLayout()
        
        page_control_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_page)
        self.prev_btn.setEnabled(False)
        page_control_layout.addWidget(self.prev_btn)
        
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.valueChanged.connect(self.go_to_page)
        self.page_spin.setEnabled(False)
        page_control_layout.addWidget(self.page_spin)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        page_control_layout.addWidget(self.next_btn)
        
        page_layout.addLayout(page_control_layout)
        
        self.page_label = QLabel("Page: 0 / 0")
        page_layout.addWidget(self.page_label)
        
        page_group.setLayout(page_layout)
        left_layout.addWidget(page_group)
        
        # Detection controls
        detection_group = QGroupBox("Detection")
        detection_layout = QVBoxLayout()
        
        self.detect_btn = QPushButton("Detect Measures")
        self.detect_btn.clicked.connect(self.detect_measures)
        self.detect_btn.setEnabled(False)
        detection_layout.addWidget(self.detect_btn)
        
        self.detect_all_btn = QPushButton("🔍 Detect All Pages")
        self.detect_all_btn.clicked.connect(self.detect_all_pages)
        self.detect_all_btn.setEnabled(False)
        self.detect_all_btn.setStyleSheet("QPushButton { background-color: #2E7D32; color: white; font-weight: bold; }")
        detection_layout.addWidget(self.detect_all_btn)
        
        self.dpi_label = QLabel("DPI: 300")
        detection_layout.addWidget(self.dpi_label)
        
        self.dpi_slider = QSlider(Qt.Orientation.Horizontal)
        self.dpi_slider.setMinimum(150)
        self.dpi_slider.setMaximum(600)
        self.dpi_slider.setValue(300)
        self.dpi_slider.setTickInterval(50)
        self.dpi_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.dpi_slider.valueChanged.connect(self.update_dpi_label)
        detection_layout.addWidget(self.dpi_slider)
        
        # 대안 전처리 옵션
        self.alt_preprocessing_check = QCheckBox("Use Alternative Preprocessing")
        self.alt_preprocessing_check.setToolTip("Use fixed threshold instead of Otsu for better thin line detection")
        detection_layout.addWidget(self.alt_preprocessing_check)
        
        # System Group Clustering Threshold
        self.system_group_threshold_label = QLabel("System Group Threshold: 8.0")
        detection_layout.addWidget(self.system_group_threshold_label)
        
        self.system_group_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.system_group_threshold_slider.setMinimum(10)  # 1.0
        self.system_group_threshold_slider.setMaximum(300)  # 30.0
        self.system_group_threshold_slider.setValue(80)  # 8.0 (default)
        self.system_group_threshold_slider.setTickInterval(10)
        self.system_group_threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.system_group_threshold_slider.valueChanged.connect(self.update_system_group_threshold_label)
        detection_layout.addWidget(self.system_group_threshold_slider)
        
        # Re-detect current page button
        self.redetect_page_btn = QPushButton("🔄 Re-detect Current Page")
        self.redetect_page_btn.clicked.connect(self.redetect_current_page)
        self.redetect_page_btn.setEnabled(False)
        self.redetect_page_btn.setToolTip("Re-run detection on current page with new threshold")
        detection_layout.addWidget(self.redetect_page_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        detection_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        detection_layout.addWidget(self.status_label)
        
        detection_group.setLayout(detection_layout)
        left_layout.addWidget(detection_group)
        
        # YOLO Music Symbol Detection controls
        yolo_group = QGroupBox("Music Symbol Detection (YOLOv8)")
        yolo_layout = QVBoxLayout()
        
        self.yolo_detect_btn = QPushButton("Detect Music Symbols")
        self.yolo_detect_btn.clicked.connect(self.detect_music_symbols)
        self.yolo_detect_btn.setEnabled(False)
        self.yolo_detect_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        yolo_layout.addWidget(self.yolo_detect_btn)
        
        self.yolo_conf_label = QLabel("Confidence: 0.25")
        yolo_layout.addWidget(self.yolo_conf_label)
        
        self.yolo_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.yolo_conf_slider.setMinimum(10)
        self.yolo_conf_slider.setMaximum(90)
        self.yolo_conf_slider.setValue(25)
        self.yolo_conf_slider.setTickInterval(10)
        self.yolo_conf_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.yolo_conf_slider.valueChanged.connect(self.update_yolo_conf_label)
        yolo_layout.addWidget(self.yolo_conf_slider)
        
        self.yolo_status_label = QLabel("")
        self.yolo_status_label.setWordWrap(True)
        yolo_layout.addWidget(self.yolo_status_label)
        
        self.yolo_filter_btn = QPushButton("Filter Classes...")
        self.yolo_filter_btn.clicked.connect(self.open_yolo_filter)
        self.yolo_filter_btn.setEnabled(False)
        self.yolo_filter_btn.setToolTip("Choose which music symbol classes to display")
        yolo_layout.addWidget(self.yolo_filter_btn)
        
        yolo_group.setLayout(yolo_layout)
        left_layout.addWidget(yolo_group)
        
        # Metronome controls
        metronome_group = QGroupBox("Metronome")
        metronome_layout = QVBoxLayout()
        
        # BPM input
        bpm_layout = QHBoxLayout()
        bpm_layout.addWidget(QLabel("BPM:"))
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setMinimum(40)
        self.bpm_spin.setMaximum(220)
        self.bpm_spin.setValue(120)
        self.bpm_spin.setSuffix(" BPM")
        self.bpm_spin.valueChanged.connect(self.update_metronome_bpm)
        bpm_layout.addWidget(self.bpm_spin)
        metronome_layout.addLayout(bpm_layout)
        
        # Time signature display
        self.time_sig_label = QLabel("Time Signature: Not detected")
        self.time_sig_label.setStyleSheet("color: #666; font-size: 10px;")
        self.time_sig_label.setWordWrap(True)
        metronome_layout.addWidget(self.time_sig_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_metronome)
        self.play_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        control_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_metronome)
        control_layout.addWidget(self.stop_btn)
        
        metronome_layout.addLayout(control_layout)
        
        # Status
        self.metronome_status = QLabel("Ready")
        self.metronome_status.setStyleSheet("color: #666;")
        metronome_layout.addWidget(self.metronome_status)
        
        # Check metronome initialization
        if not self.metronome.is_initialized:
            metronome_group.setEnabled(False)
            self.metronome_status.setText("Metronome disabled (no audio available)")
            self.metronome_status.setStyleSheet("color: red; font-size: 10px;")
            logger.debug("metronome not initialized")
        else:
            # Show which audio method is being used
            if self.metronome.use_pygame:
                self.metronome_status.setText("Ready (using pygame audio)")
                print("DEBUG: metronome using pygame")
            else:
                self.metronome_status.setText("Ready (using system bell)")
                print("DEBUG: metronome using system bell")
        
        metronome_group.setLayout(metronome_layout)
        left_layout.addWidget(metronome_group)
        
        # Display controls
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.show_staff_cb = QCheckBox("Show Staff Lines")
        self.show_staff_cb.setChecked(True)
        self.show_staff_cb.toggled.connect(self.toggle_staff_lines)
        display_layout.addWidget(self.show_staff_cb)
        
        self.show_barlines_cb = QCheckBox("Show Barlines")
        self.show_barlines_cb.setChecked(True)
        self.show_barlines_cb.toggled.connect(self.toggle_barlines)
        display_layout.addWidget(self.show_barlines_cb)
        
        self.show_candidates_cb = QCheckBox("Show Barline Candidates")
        self.show_candidates_cb.setChecked(False)
        self.show_candidates_cb.toggled.connect(self.toggle_candidates)
        display_layout.addWidget(self.show_candidates_cb)
        
        self.show_system_groups_cb = QCheckBox("Show System Groups")
        self.show_system_groups_cb.setChecked(True)
        self.show_system_groups_cb.setToolTip("Show clustering of staff systems (for quartet/orchestra scores)")
        self.show_system_groups_cb.toggled.connect(self.toggle_system_groups)
        display_layout.addWidget(self.show_system_groups_cb)
        
        self.show_measure_boxes_cb = QCheckBox("Show Measure Boxes")
        self.show_measure_boxes_cb.setChecked(True)
        self.show_measure_boxes_cb.setToolTip("Preview measure extraction bounding boxes")
        self.show_measure_boxes_cb.toggled.connect(self.toggle_measure_boxes)
        display_layout.addWidget(self.show_measure_boxes_cb)
        
        self.show_bracket_candidates_cb = QCheckBox("Show Bracket Candidates")
        self.show_bracket_candidates_cb.setChecked(False)
        self.show_bracket_candidates_cb.setToolTip("Show bracket candidate vertical lines")
        self.show_bracket_candidates_cb.toggled.connect(self.toggle_bracket_candidates)
        display_layout.addWidget(self.show_bracket_candidates_cb)
        
        self.show_brackets_cb = QCheckBox("Show Verified Brackets")
        self.show_brackets_cb.setChecked(True)
        self.show_brackets_cb.setToolTip("Show verified bracket detections")
        self.show_brackets_cb.toggled.connect(self.toggle_brackets)
        display_layout.addWidget(self.show_brackets_cb)
        
        self.show_yolo_cb = QCheckBox("Show Music Symbols")
        self.show_yolo_cb.setChecked(False)
        self.show_yolo_cb.setToolTip("Show YOLOv8 detected music symbols")
        self.show_yolo_cb.toggled.connect(self.toggle_yolo_detections)
        display_layout.addWidget(self.show_yolo_cb)
        
        # Manual barline addition
        self.add_barline_btn = QPushButton("Add Barline (Click Mode)")
        self.add_barline_btn.setCheckable(True)
        self.add_barline_btn.setChecked(False)
        self.add_barline_btn.setToolTip("Click to enable barline addition mode.\n"
                                      "Left click on image: Add barline\n"
                                      "Right click on image: Remove nearby barline")
        self.add_barline_btn.toggled.connect(self.toggle_add_barline_mode)
        display_layout.addWidget(self.add_barline_btn)
        
        self.fit_window_btn = QPushButton("Fit to Window")
        self.fit_window_btn.clicked.connect(self.fit_to_window)
        display_layout.addWidget(self.fit_window_btn)
        
        self.zoom_label = QLabel("Zoom: 100%")
        display_layout.addWidget(self.zoom_label)
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(25)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        display_layout.addWidget(self.zoom_slider)
        
        display_group.setLayout(display_layout)
        left_layout.addWidget(display_group)
        
        # Results
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        
        self.results_label = QLabel("No results yet")
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)
        
        self.extract_measures_btn = QPushButton("Extract Measures")
        self.extract_measures_btn.clicked.connect(self.extract_measures)
        self.extract_measures_btn.setEnabled(False)
        self.extract_measures_btn.setToolTip("Extract individual measure images")
        results_layout.addWidget(self.extract_measures_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        results_layout.addWidget(self.export_btn)
        
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)
        
        left_layout.addStretch()
        
        # Right panel - Image display
        self.image_widget = ScoreImageWidget()
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(self.image_widget)
        splitter.setStretchFactor(1, 1)
        
        # Store splitter reference for resize events
        self.splitter = splitter
        
        # Create menu bar
        self.create_menu_bar()
        
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open PDF", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_pdf)
        file_menu.addAction(open_action)
        
        export_action = QAction("Export Results", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        zoom_reset_action = QAction("Reset Zoom", self)
        zoom_reset_action.setShortcut("Ctrl+0")
        zoom_reset_action.triggered.connect(self.zoom_reset)
        view_menu.addAction(zoom_reset_action)
        
    def load_pdf(self):
        """Load a PDF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            try:
                self.pdf_path = file_path
                self.pdf_document = fitz.open(file_path)
                self.total_pages = len(self.pdf_document)
                self.current_page = 0
                
                # Update UI
                self.file_label.setText(os.path.basename(file_path))
                self.page_spin.setMaximum(self.total_pages)
                self.page_spin.setValue(1)
                self.page_spin.setEnabled(True)
                self.prev_btn.setEnabled(False)
                self.next_btn.setEnabled(self.total_pages > 1)
                self.detect_btn.setEnabled(True)
                self.detect_all_btn.setEnabled(True)
                self.redetect_page_btn.setEnabled(True)
                
                # Check for existing JSON file and load it
                self.check_and_load_existing_json()
                
                # Load first page
                self.load_current_page()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load PDF: {str(e)}")
    
    def check_and_load_existing_json(self):
        """Check for existing JSON file with same name as PDF and load it"""
        if not self.pdf_path:
            return
            
        # Generate JSON file path
        pdf_dir = os.path.dirname(self.pdf_path)
        pdf_basename = os.path.splitext(os.path.basename(self.pdf_path))[0]
        json_path = os.path.join(pdf_dir, f"{pdf_basename}.json")
        
        if os.path.exists(json_path):
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.all_pages_results = json.load(f)
                
                print(f"Loaded existing detection results from: {json_path}")
                
                # Update UI to show that data is loaded
                total_pages = len(self.all_pages_results.get('pages', {}))
                self.status_label.setText(f"Loaded detection results for {total_pages} pages from JSON")
                
                # Enable relevant buttons
                self.export_btn.setEnabled(True)
                self.extract_measures_btn.setEnabled(True)
                self.yolo_filter_btn.setEnabled(True)
                
                # Load current page results if available, or go to first available page
                current_page_key = str(self.current_page + 1)
                available_pages = list(self.all_pages_results.get('pages', {}).keys())
                
                if current_page_key in self.all_pages_results.get('pages', {}):
                    self.load_stored_page_results()
                elif available_pages:
                    # Go to first available page
                    first_page = int(available_pages[0])
                    print(f"Current page {self.current_page + 1} not available, switching to page {first_page}")
                    self.current_page = first_page - 1
                    self.page_spin.setValue(first_page)
                    self.load_current_page()
                    self.load_stored_page_results()
                    
            except Exception as e:
                logger.error(f"Failed to load existing JSON file: {str(e)}")
                self.all_pages_results = None
        else:
            logger.debug(f"No existing JSON file found at: {json_path}")
                
    def load_current_page(self):
        """Load and display the current page"""
        if self.pdf_document is None:
            return
            
        try:
            # Get page
            page = self.pdf_document[self.current_page]
            
            # Render page to pixmap
            mat = fitz.Matrix(self.dpi_slider.value() / 72.0, 
                            self.dpi_slider.value() / 72.0)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img_data = img_data.reshape(pix.height, pix.width, pix.n)
            
            # Convert to RGB if necessary
            if pix.n == 4:  # RGBA
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:  # Grayscale
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
                
            # Display image
            self.image_widget.set_image(img_data)
            
            # Enable auto-fit and apply it
            QApplication.processEvents()  # Ensure widget is updated
            self.image_widget.set_auto_fit(True)
            fit_scale = self.image_widget.apply_auto_fit()
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(int(fit_scale * 100))
            self.zoom_label.setText(f"Zoom: {int(fit_scale * 100)}%")
            self.zoom_slider.blockSignals(False)
            
            # Update page label
            self.page_label.setText(f"Page: {self.current_page + 1} / {self.total_pages}")
            
            # Clear previous detection results
            self.detection_results = None
            self.image_widget.set_detection_results([], [], 0)
            self.image_widget.set_yolo_detections([], {})  # Clear YOLO detections
            self.results_label.setText("No results yet")
            self.export_btn.setEnabled(False)
            self.yolo_detect_btn.setEnabled(True)  # Enable YOLO detection button
            self.yolo_filter_btn.setEnabled(False)  # Disable filter button
            self.yolo_status_label.setText("")  # Clear YOLO status
            self.time_sig_label.setText("Time Signature: Not detected")  # Reset time signature
            self.time_sig_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Load stored results if available
            self.load_stored_page_results()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load page: {str(e)}")
            
    def prev_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.page_spin.setValue(self.current_page + 1)
            
    def next_page(self):
        """Go to next page"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.page_spin.setValue(self.current_page + 1)
            
    def go_to_page(self, page_num):
        """Go to specific page"""
        self.current_page = page_num - 1
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < self.total_pages - 1)
        self.load_current_page()
        
        # Load stored results for this page if available
        if hasattr(self, 'all_pages_results') and self.all_pages_results:
            current_page_key = str(self.current_page + 1)
            if current_page_key in self.all_pages_results.get('pages', {}):
                self.load_stored_page_results()
            else:
                # Clear any previous detection results for pages without data
                self.image_widget.clear_detections()
                self.results_label.setText("No detection results available for this page")
        else:
            # Clear any previous detection results when no stored data exists
            self.image_widget.clear_detections()
            self.results_label.setText("Load a PDF and run detection")
        
    def detect_measures(self):
        """Run measure detection on current page"""
        if self.pdf_path is None:
            return
            
        # Disable controls during detection
        self.detect_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Create and start detection thread
        self.detection_thread = DetectionThread(
            self.pdf_path, self.current_page, self.dpi_slider.value(),
            self.alt_preprocessing_check.isChecked()
        )
        self.detection_thread.progress.connect(self.update_status)
        self.detection_thread.finished.connect(self.detection_finished)
        self.detection_thread.error.connect(self.detection_error)
        self.detection_thread.start()
        
    def update_status(self, message):
        """Update status message"""
        self.status_label.setText(message)
        
    def detection_finished(self, results):
        """Handle detection completion"""
        self.detection_results = results
        
        # Store binary image for consensus validation
        if 'original_image' in results:
            detector = MeasureDetector()
            self._current_binary_img = detector.preprocess_image(results['original_image'])
        
        # Extract bracket data from results
        detected_brackets = results.get('detected_brackets', [])
        raw_bracket_candidates = results.get('bracket_candidates', [])  # Get raw candidates
        verified_brackets = detected_brackets  # Detected brackets are already verified
        
        logger.debug(f"Debug GUI: detected_brackets count: {len(detected_brackets)}")
        logger.debug(f"Debug GUI: raw_bracket_candidates count: {len(raw_bracket_candidates)}")
        if raw_bracket_candidates:
            logger.debug(f"Debug GUI: First raw candidate type: {type(raw_bracket_candidates[0])}")
            logger.debug(f"Debug GUI: First raw candidate: {raw_bracket_candidates[0]}")
        
        # Filter and store only valid coordinate lists for candidates
        self.bracket_candidates = []
        for candidate in raw_bracket_candidates:
            if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                self.bracket_candidates.append(candidate)
        
        self.verified_brackets = detected_brackets  # These are verified bracket dicts
        
        logger.debug(f"Debug GUI: Filtered bracket_candidates count: {len(self.bracket_candidates)}")
        logger.debug(f"Debug GUI: Stored verified_brackets count: {len(self.verified_brackets)}")
        if self.bracket_candidates:
            logger.debug(f"Debug GUI: First filtered candidate: {self.bracket_candidates[0]}")
        
        # Update display
        self.image_widget.set_detection_results(
            results['staff_lines'],
            results['barlines'],
            results['measure_count'],
            results.get('barline_candidates', []),
            results.get('staff_lines_with_ranges', []),
            results.get('barlines_with_systems', []),
            results.get('staff_systems', []),
            results.get('system_groups', []),
            None,  # measure_boxes - not used yet
            self.bracket_candidates,  # bracket_candidates - raw coordinates
            self.verified_brackets    # verified_brackets - dict objects
        )
        
        # Update results label with system-specific info
        candidates_count = len(results.get('barline_candidates', []))
        barlines_with_systems = results.get('barlines_with_systems', [])
        staff_systems = results.get('staff_systems', [])
        system_groups = results.get('system_groups', [])
        
        # Add bracket info
        bracket_count = len(detected_brackets)
        
        results_text = (
            f"Detected:\n"
            f"- {len(results['staff_lines'])} staff lines\n"
            f"- {len(staff_systems)} staff systems\n"
            f"- {bracket_count} brackets\n"
            f"- {candidates_count} barline candidates\n"
            f"- {len(results['barlines'])} valid barlines\n"
            f"- {results['measure_count']} measures\n"
        )
        
        # Add system group clustering info
        if system_groups:
            results_text += f"\nSystem Clustering:\n"
            results_text += f"- {len(system_groups)} system group(s)\n"
            for group_idx, system_indices in enumerate(system_groups):
                results_text += f"  Group {group_idx + 1}: {len(system_indices)} systems\n"
        
        # Add system-specific breakdown if available
        if barlines_with_systems:
            system_counts = {}
            for bl in barlines_with_systems:
                system_idx = bl['system_idx']
                system_counts[system_idx] = system_counts.get(system_idx, 0) + 1
            
            results_text += f"\nPer System:\n"
            for system_idx in sorted(system_counts.keys()):
                results_text += f"- System {system_idx + 1}: {system_counts[system_idx]} barlines\n"
        
        self.results_label.setText(results_text)
        
        # Auto-generate measure boxes for metronome functionality
        self.generate_measure_boxes()
        logger.debug(f"DEBUG: Auto-generated {len(self.image_widget.measure_boxes)} measure boxes for metronome")
        
        # Re-enable controls
        self.detect_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.extract_measures_btn.setEnabled(True)  # Enable measure extraction
        self.progress_bar.hide()
        self.status_label.setText("Detection complete!")
        
    def detection_error(self, error_msg):
        """Handle detection error"""
        QMessageBox.critical(self, "Detection Error", error_msg)
        self.detect_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText("Detection failed")
        
    def update_dpi_label(self, value):
        """Update DPI label"""
        self.dpi_label.setText(f"DPI: {value}")
    
    def update_system_group_threshold_label(self, value):
        """Update system group threshold label when slider changes"""
        threshold_value = value / 10.0  # Convert back to decimal (30 -> 3.0, 80 -> 8.0)
        self.system_group_threshold_label.setText(f"System Group Threshold: {threshold_value}")
    
    def get_system_group_threshold(self):
        """Get current system group threshold value"""
        return self.system_group_threshold_slider.value() / 10.0
    
    def redetect_current_page(self):
        """Re-detect measures on current page with current settings"""
        if self.pdf_path is None:
            return
            
        # Disable controls during detection
        self.redetect_page_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Re-detecting current page...")
        
        # Create detection thread for current page only
        self.detection_thread = DetectionThread(
            self.pdf_path, self.current_page, self.dpi_slider.value(),
            self.alt_preprocessing_check.isChecked(),
            self.get_system_group_threshold()
        )
        self.detection_thread.finished.connect(self.on_current_page_detection_complete)
        self.detection_thread.error.connect(self.detection_error)
        self.detection_thread.start()
    
    def on_current_page_detection_complete(self, results):
        """Handle completion of current page re-detection"""
        # Store the results (keep original ratio coordinates)
        self.detection_results = results
        
        # Convert ratio coordinates to pixels for display
        pixel_results = self._convert_results_to_pixels(results)
        
        # Update the image widget with pixel coordinates
        self.image_widget.set_detection_results(
            pixel_results.get('staff_lines', []),
            pixel_results.get('barlines', []), 
            pixel_results.get('measure_count', 0),
            pixel_results.get('barline_candidates', []),
            pixel_results.get('staff_lines_with_ranges', []),
            pixel_results.get('barlines_with_systems', []),
            pixel_results.get('staff_systems', []),
            pixel_results.get('system_groups', []),
            None,  # measure_boxes - will be auto-generated
            pixel_results.get('bracket_candidates', []),
            pixel_results.get('detected_brackets', [])
        )
        
        # Auto-generate measure boxes
        self.generate_measure_boxes()
        
        # Update results display
        bracket_count = len(results.get('detected_brackets', []))
        results_text = (
            f"Page {self.current_page + 1} Results (Re-detected):\n"
            f"- {len(results.get('staff_lines', []))} staff lines\n"
            f"- {len(results.get('staff_systems', []))} staff systems\n"
            f"- {len(results.get('barlines', []))} valid barlines\n"
            f"- {results.get('measure_count', 0)} measures\n"
            f"- {bracket_count} brackets"
        )
        self.results_label.setText(results_text)
        
        # Debug bracket information
        logger.debug(f"DEBUG: Re-detection results - brackets: {bracket_count}")
        if bracket_count > 0:
            logger.debug(f"DEBUG: First bracket: {results.get('detected_brackets', [])[0]}")
        
        # Update stored results if they exist
        if hasattr(self, 'all_pages_results') and self.all_pages_results:
            current_page_key = str(self.current_page + 1)
            if current_page_key in self.all_pages_results['pages']:
                # Update the stored page data
                self.all_pages_results['pages'][current_page_key]['measures'] = {
                    'staff_lines': results.get('staff_lines', []),
                    'barlines': results.get('barlines', []),
                    'measure_count': results.get('measure_count', 0),
                    'staff_systems': results.get('staff_systems', []),
                    'system_groups': results.get('system_groups', []),
                    'barlines_with_systems': results.get('barlines_with_systems', []),
                    'detected_brackets': results.get('detected_brackets', []),
                    'bracket_candidates': results.get('bracket_candidates', [])
                }
                
                # Save updated results to JSON file
                self.save_results_to_json()
        
        # Re-enable controls
        self.redetect_page_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText("Current page re-detection complete!")
    
    def save_results_to_json(self):
        """Save current detection results to JSON file"""
        if not hasattr(self, 'all_pages_results') or not self.all_pages_results:
            return
            
        try:
            # Generate JSON file path (same directory and name as PDF, but with .json extension)
            pdf_dir = os.path.dirname(self.pdf_path)
            pdf_basename = os.path.splitext(os.path.basename(self.pdf_path))[0]
            json_path = os.path.join(pdf_dir, f"{pdf_basename}.json")
            
            import json
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_pages_results, f, indent=2, ensure_ascii=False, default=self.convert_numpy_types)
            
            logger.debug(f"Re-detection results saved to JSON file: {json_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results to JSON: {str(e)}")
    
    def convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj
    
    def _ratio_to_pixels(self, coords_dict, width, height):
        """Convert ratio coordinates (0-1) back to pixel coordinates
        
        Args:
            coords_dict: Dictionary containing ratio coordinates
            width: Page width in pixels
            height: Page height in pixels
            
        Returns:
            dict: Coordinates converted to pixels
        """
        converted = coords_dict.copy()
        
        # Convert x coordinates from ratios to pixels
        if 'x' in converted:
            converted['x'] = converted['x'] * width
        if 'x1' in converted:
            converted['x1'] = converted['x1'] * width
        if 'x2' in converted:
            converted['x2'] = converted['x2'] * width
        if 'center_x' in converted:
            converted['center_x'] = converted['center_x'] * width
        if 'left' in converted:
            converted['left'] = converted['left'] * width
        if 'right' in converted:
            converted['right'] = converted['right'] * width
            
        # Convert y coordinates from ratios to pixels
        if 'y' in converted:
            converted['y'] = converted['y'] * height
        if 'y1' in converted:
            converted['y1'] = converted['y1'] * height
        if 'y2' in converted:
            converted['y2'] = converted['y2'] * height
        if 'center_y' in converted:
            converted['center_y'] = converted['center_y'] * height
        if 'top' in converted:
            converted['top'] = converted['top'] * height
        if 'bottom' in converted:
            converted['bottom'] = converted['bottom'] * height
        if 'y_start' in converted:
            converted['y_start'] = converted['y_start'] * height
        if 'y_end' in converted:
            converted['y_end'] = converted['y_end'] * height
            
        # Convert lists of coordinates
        if 'lines' in converted and isinstance(converted['lines'], list):
            converted['lines'] = [line * height for line in converted['lines']]
            
        return converted
    
    def _convert_results_to_pixels(self, results):
        """Convert detection results from ratios back to pixels for display
        
        Args:
            results: Detection results with ratio coordinates
            
        Returns:
            dict: Results with pixel coordinates
        """
        if not results or not hasattr(self, 'image_widget') or not self.image_widget.original_pixmap:
            return results
            
        # Get current page dimensions
        width = self.image_widget.original_pixmap.width()
        height = self.image_widget.original_pixmap.height()
        
        converted_results = results.copy()
        
        # Convert barlines (simple x coordinates)
        if 'barlines' in converted_results and converted_results['barlines']:
            converted_results['barlines'] = [x * width for x in converted_results['barlines']]
        
        # Convert barline candidates (x coordinates)
        if 'barline_candidates' in converted_results and converted_results['barline_candidates']:
            converted_results['barline_candidates'] = [x * width for x in converted_results['barline_candidates']]
            
        # Convert staff lines (y coordinates)
        if 'staff_lines' in converted_results and converted_results['staff_lines']:
            converted_results['staff_lines'] = [y * height for y in converted_results['staff_lines']]
            
        # Convert complex coordinate structures
        for key in ['staff_lines_with_ranges', 'barlines_with_systems', 'staff_systems', 
                   'system_groups', 'detected_brackets', 'bracket_candidates']:
            if key in converted_results and converted_results[key]:
                converted_results[key] = [self._ratio_to_pixels(item, width, height) 
                                        for item in converted_results[key]]
        
        return converted_results
    
    def _convert_yolo_detections_to_pixels(self, yolo_detections):
        """Convert YOLO detections from ratios back to pixels for display
        
        Args:
            yolo_detections: List of YOLO detections with ratio bbox coordinates
            
        Returns:
            list: YOLO detections with pixel bbox coordinates
        """
        if not yolo_detections or not hasattr(self, 'image_widget') or not self.image_widget.original_pixmap:
            return yolo_detections
            
        # Get current page dimensions
        width = self.image_widget.original_pixmap.width()
        height = self.image_widget.original_pixmap.height()
        
        converted_detections = []
        for det in yolo_detections:
            converted_det = det.copy()
            
            # Convert bbox coordinates from ratios to pixels
            if 'bbox' in converted_det:
                x1_ratio, y1_ratio, x2_ratio, y2_ratio = converted_det['bbox']
                converted_det['bbox'] = [
                    x1_ratio * width,
                    y1_ratio * height,
                    x2_ratio * width,
                    y2_ratio * height
                ]
            
            converted_detections.append(converted_det)
            
        return converted_detections
        
    def toggle_staff_lines(self, checked):
        """Toggle staff line display"""
        self.image_widget.toggle_staff_lines(checked)
        
    def toggle_barlines(self, checked):
        """Toggle barline display"""
        self.image_widget.toggle_barlines(checked)
        
    def toggle_candidates(self, checked):
        """Toggle barline candidates display"""
        self.image_widget.toggle_candidates(checked)
        
    def toggle_system_groups(self, checked):
        """Toggle system group clustering display"""
        self.image_widget.toggle_system_groups(checked)
        
    def fit_to_window(self):
        """Fit image to window and enable auto-fit"""
        if self.image_widget.original_pixmap:
            self.image_widget.set_auto_fit(True)
            fit_scale = self.image_widget.apply_auto_fit()
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(int(fit_scale * 100))
            self.zoom_label.setText(f"Zoom: {int(fit_scale * 100)}%")
            self.zoom_slider.blockSignals(False)
        
    def update_zoom(self, value):
        """Update zoom level"""
        # Disable auto-fit when user manually changes zoom
        self.image_widget.set_auto_fit(False)
        self.zoom_label.setText(f"Zoom: {value}%")
        self.image_widget.set_scale(value / 100.0)
        
    def zoom_in(self):
        """Zoom in"""
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(min(current + 25, 200))
        
    def zoom_out(self):
        """Zoom out"""
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(max(current - 25, 25))
        
    def zoom_reset(self):
        """Reset zoom to 100%"""
        self.image_widget.set_auto_fit(False)
        self.zoom_slider.setValue(100)
        
    def export_results(self):
        """Export detection results"""
        if self.detection_results is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", 
            f"{os.path.splitext(self.pdf_path)[0]}_page{self.current_page+1}_results.png",
            "PNG Files (*.png)"
        )
        
        if file_path:
            try:
                # Get current displayed pixmap
                pixmap = self.image_widget.pixmap()
                if pixmap:
                    pixmap.save(file_path)
                    QMessageBox.information(self, "Success", 
                                          f"Results exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
    
    def toggle_measure_boxes(self, checked):
        """Toggle measure box display"""
        self.image_widget.show_measure_boxes = checked
        if checked and hasattr(self, 'detection_results') and self.detection_results:
            # Generate measure boxes from current results
            self.generate_measure_boxes()
        else:
            self.image_widget.measure_boxes = []
        self.image_widget.update()
    
    def toggle_bracket_candidates(self, checked):
        """Toggle bracket candidate display"""
        self.image_widget.toggle_bracket_candidates(checked)
    
    def toggle_brackets(self, checked):
        """Toggle verified bracket display"""
        logger.debug(f"GUI toggle_brackets: checked={checked}")
        self.image_widget.toggle_brackets(checked)
    
    def toggle_yolo_detections(self, checked):
        """Toggle YOLO detections display"""
        self.image_widget.toggle_yolo_detections(checked)
    
    def toggle_add_barline_mode(self, checked):
        """Toggle manual barline addition mode"""
        logger.debug(f"toggle_add_barline_mode called with checked={checked}")
        self.image_widget.add_barline_mode = checked
        if checked:
            self.add_barline_btn.setText("Add Barline (ACTIVE)")
            self.add_barline_btn.setStyleSheet("background-color: lightgreen;")
            logger.info("Manual barline addition mode enabled - click on image to add barlines")
            logger.debug(f"image_widget.add_barline_mode set to {self.image_widget.add_barline_mode}")
        else:
            self.add_barline_btn.setText("Add Barline (Click Mode)")
            self.add_barline_btn.setStyleSheet("")
            logger.info("Manual barline addition mode disabled")
            logger.debug(f"image_widget.add_barline_mode set to {self.image_widget.add_barline_mode}")
    
    def add_manual_barline_to_detection(self, x_ratio, y_ratio):
        """Add manual barline to detection results and regenerate measure boxes"""
        if not hasattr(self, 'detection_results') or not self.detection_results:
            logger.warning("No detection results available to add manual barline")
            return
        
        logger.info(f"Adding manual barline at x_ratio={x_ratio:.3f}, y_ratio={y_ratio:.3f} to detection results")
        
        # Find which system group the Y coordinate belongs to
        staff_systems = self.detection_results.get('staff_systems', [])
        system_groups = self.detection_results.get('system_groups', [])
        
        if not staff_systems or not system_groups:
            logger.warning("No staff systems or system groups available")
            return
        
        target_group_idx = None
        
        # Find the system group that contains the Y coordinate
        for group_idx, system_indices in enumerate(system_groups):
            for sys_idx in system_indices:
                if sys_idx < len(staff_systems):
                    system = staff_systems[sys_idx]
                    # Check if Y coordinate is within this system's range
                    if system['top'] <= y_ratio <= system['bottom']:
                        target_group_idx = group_idx
                        logger.debug(f"Y coordinate {y_ratio:.3f} found in system {sys_idx} (group {group_idx})")
                        break
            if target_group_idx is not None:
                break
        
        if target_group_idx is None:
            logger.warning(f"Could not find system group for Y coordinate {y_ratio:.3f}")
            return
        
        # Add to barlines_with_systems for the target system group only
        barlines_with_systems = self.detection_results.get('barlines_with_systems', [])
        
        # Calculate Y range for the target system group (same as regular barlines)
        group_systems = [staff_systems[i] for i in system_groups[target_group_idx] 
                        if i < len(staff_systems)]
        
        if group_systems:
            y_start = min(sys['top'] for sys in group_systems)
            y_end = max(sys['bottom'] for sys in group_systems)
            
            # Add manual barline to the target system group with same structure as regular barlines
            manual_barline_info = {
                'x': x_ratio,
                'y_start': y_start,
                'y_end': y_end,
                'y': y_ratio,  # Store original click Y coordinate for reference
                'system_idx': target_group_idx,  # This is actually system group index
                'type': 'manual'
            }
            logger.debug(f"Created manual barline info: {manual_barline_info}")
        else:
            logger.warning("No systems found in target group")
            return
        barlines_with_systems.append(manual_barline_info)
        logger.info(f"Added manual barline to system group {target_group_idx} only")
        
        # Update detection results
        self.detection_results['barlines_with_systems'] = barlines_with_systems
        
        # Save to JSON file
        self.save_results_to_json()
        
        # Regenerate measure boxes
        self.generate_measure_boxes()
        
        # Update display
        self.image_widget.update()
        
        logger.info(f"Manual barline added and {len(barlines_with_systems)} barline entries updated")
    
    def remove_manual_barline_from_detection(self, x_ratio):
        """Remove manual barline from detection results and regenerate measure boxes"""
        if not hasattr(self, 'detection_results') or not self.detection_results:
            logger.warning("No detection results available to remove manual barline")
            return
        
        logger.info(f"Removing manual barline at x_ratio={x_ratio:.3f} from detection results")
        
        # Remove from barlines_with_systems
        barlines_with_systems = self.detection_results.get('barlines_with_systems', [])
        tolerance = 0.02  # 2% tolerance
        
        original_count = len(barlines_with_systems)
        barlines_with_systems = [bl for bl in barlines_with_systems 
                                if not (bl.get('type') == 'manual' and abs(bl.get('x', 0) - x_ratio) < tolerance)]
        
        removed_count = original_count - len(barlines_with_systems)
        
        # Update detection results
        self.detection_results['barlines_with_systems'] = barlines_with_systems
        
        # Save to JSON file
        self.save_results_to_json()
        
        # Regenerate measure boxes
        self.generate_measure_boxes()
        
        # Update display
        self.image_widget.update()
        
        logger.info(f"Removed {removed_count} manual barline entries")
    
    def remove_barline_from_detection(self, x_ratio):
        """Remove any barline (manual or regular) from detection results and regenerate measure boxes"""
        if not hasattr(self, 'detection_results') or not self.detection_results:
            logger.warning("No detection results available to remove barline")
            return
        
        logger.info(f"Removing barline at x_ratio={x_ratio:.3f} from detection results")
        
        # Remove from barlines_with_systems (any type)
        barlines_with_systems = self.detection_results.get('barlines_with_systems', [])
        tolerance = 0.02  # 2% tolerance
        
        original_count = len(barlines_with_systems)
        
        # Find closest barline within tolerance
        closest_barline = None
        closest_distance = float('inf')
        
        for bl in barlines_with_systems:
            bl_x = bl.get('x', bl.get('x_ratio', 0))
            distance = abs(bl_x - x_ratio)
            if distance < tolerance and distance < closest_distance:
                closest_distance = distance
                closest_barline = bl
        
        # Remove the closest barline if found
        if closest_barline:
            barlines_with_systems.remove(closest_barline)
            barline_type = closest_barline.get('type', 'regular')
            logger.info(f"Removed {barline_type} barline at x={closest_barline.get('x', closest_barline.get('x_ratio', 0)):.3f}")
        else:
            logger.info(f"No barline found near x={x_ratio:.3f}")
            return
        
        # Update detection results
        self.detection_results['barlines_with_systems'] = barlines_with_systems
        
        # Save to JSON file
        self.save_results_to_json()
        
        # Regenerate measure boxes
        self.generate_measure_boxes()
        
        # Update display
        self.image_widget.update()
        
        logger.info(f"Removed 1 barline entry, {len(barlines_with_systems)} barlines remaining")
    
    def update_yolo_conf_label(self, value):
        """Update YOLO confidence threshold label"""
        self.yolo_conf_label.setText(f"Confidence: {value/100:.2f}")
    
    def detect_music_symbols(self):
        """Run YOLOv8 music symbol detection"""
        if self.pdf_document is None:
            return
        
        # Disable button during detection
        self.yolo_detect_btn.setEnabled(False)
        self.yolo_status_label.setText("Detecting music symbols...")
        
        # Get current page image
        try:
            page = self.pdf_document[self.current_page]
            mat = fitz.Matrix(self.dpi_slider.value() / 72.0, self.dpi_slider.value() / 72.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Convert to RGB if necessary
            if pix.n == 4:  # RGBA
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:  # Grayscale
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            
            # Create and start YOLO detection thread
            self.yolo_thread = YOLODetectionThread(
                img_data,
                model_path="stage4_best.pt",
                conf_threshold=self.yolo_conf_slider.value() / 100.0
            )
            self.yolo_thread.progress.connect(self.on_yolo_progress)
            self.yolo_thread.finished.connect(self.on_yolo_finished)
            self.yolo_thread.error.connect(self.on_yolo_error)
            self.yolo_thread.start()
            
        except Exception as e:
            self.yolo_status_label.setText(f"Error: {str(e)}")
            self.yolo_detect_btn.setEnabled(True)
    
    def on_yolo_progress(self, message):
        """Handle YOLO detection progress"""
        self.yolo_status_label.setText(message)
    
    def on_yolo_finished(self, results):
        """Handle YOLO detection completion"""
        self.yolo_detect_btn.setEnabled(True)
        
        detections = results.get("detections", [])
        class_names = results.get("class_names", {})
        num_detections = results.get("num_detections", 0)
        
        # Convert YOLO detections from ratios to pixels for display
        pixel_detections = self._convert_yolo_detections_to_pixels(detections)
        
        # Update image widget with detections
        self.image_widget.set_yolo_detections(pixel_detections, class_names)
        
        # Enable filter button
        self.yolo_filter_btn.setEnabled(True)
        
        # Show detections automatically
        self.show_yolo_cb.setChecked(True)
        
        # Detect time signature
        self.detect_time_signature()
        
        # Update status
        if num_detections > 0:
            # Count detections by category
            category_counts = {}
            for det in detections:
                class_name = det.get("class_name", "unknown")
                # Determine category
                if "notehead" in class_name.lower():
                    category = "noteheads"
                elif "flag" in class_name.lower():
                    category = "flags"
                elif "rest" in class_name.lower():
                    category = "rests"
                elif "clef" in class_name.lower():
                    category = "clefs"
                elif "key" in class_name.lower() or "accidental" in class_name.lower():
                    category = "keys/accidentals"
                elif "timesig" in class_name.lower():
                    category = "time signatures"
                else:
                    category = "other"
                
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Format status message
            status_parts = [f"{count} {cat}" for cat, count in category_counts.items()]
            status = f"Detected {num_detections} symbols: {', '.join(status_parts)}"
            self.yolo_status_label.setText(status)
        else:
            self.yolo_status_label.setText("No music symbols detected")
    
    def on_yolo_error(self, error_message):
        """Handle YOLO detection error"""
        self.yolo_detect_btn.setEnabled(True)
        self.yolo_status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Detection Error", error_message)
    
    def open_yolo_filter(self):
        """Open YOLO class filter dialog"""
        if not hasattr(self.image_widget, 'yolo_class_names') or not self.image_widget.yolo_class_names:
            QMessageBox.information(self, "No Data", "No music symbol detections available to filter.")
            return
        
        dialog = YOLOClassFilterDialog(
            self.image_widget.yolo_class_names,
            self.image_widget.yolo_visible_classes,
            self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            visible_classes = dialog.get_visible_classes()
            self.image_widget.set_yolo_visible_classes(visible_classes)
            
            # Update status to show filtered count
            if hasattr(self.image_widget, 'yolo_detections'):
                visible_count = sum(1 for det in self.image_widget.yolo_detections 
                                  if det["class_id"] in visible_classes)
                total_count = len(self.image_widget.yolo_detections)
                
                if visible_count != total_count:
                    self.yolo_status_label.setText(
                        f"Showing {visible_count} of {total_count} detected symbols (filtered)"
                    )
                else:
                    self.yolo_status_label.setText(
                        f"Showing all {total_count} detected symbols"
                    )
    
    def update_metronome_bpm(self, bpm):
        """Update metronome BPM and restart timer if playing"""
        if self.is_playing:
            self.stop_metronome()
            self.start_metronome()
    
    def toggle_metronome(self):
        """Toggle metronome play/pause"""
        if self.is_playing:
            self.pause_metronome()
        else:
            self.start_metronome()
    
    def start_metronome(self):
        """Start the high-precision metronome"""
        if not self.metronome.is_initialized:
            return
            
        bpm = self.bpm_spin.value()
        self.expected_beat_interval_ms = int(60000 / bpm)  # Convert BPM to milliseconds per beat
        
        # Reset counters when starting
        self.current_beat = 0
        self.current_measure = 0
        self.current_group_index = 0
        self.current_measure_in_group = 0
        self.total_beats_played = 0
        
        # Enable measure highlighting if measures are detected
        if hasattr(self.image_widget, 'measure_boxes') and self.image_widget.measure_boxes:
            # Highlight the first measure before starting
            self._update_measure_highlight()
        
        # Start high-precision metronome thread
        self.metronome_stop_event.clear()
        self.metronome_thread = threading.Thread(target=self._precision_metronome_thread, daemon=True)
        self.metronome_thread.start()
        
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        self.play_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        
        # Show initial status
        if hasattr(self.image_widget, 'measure_boxes') and self.image_widget.measure_boxes:
            beat_text = f"♪ M1 Beat 1/{self.beats_per_measure}"
        else:
            beat_text = f"Beat 1/{self.beats_per_measure}"
        self.metronome_status.setText(f"Playing at {bpm} BPM - {beat_text}")
    
    def pause_metronome(self):
        """Pause the metronome"""
        self.metronome_stop_event.set()
        if self.metronome_thread and self.metronome_thread.is_alive():
            self.metronome_thread.join(timeout=0.1)  # Brief wait for clean shutdown
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.play_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.metronome_status.setText("Paused")
    
    def stop_metronome(self, show_completion_message=False):
        """Stop the metronome"""
        self.metronome_stop_event.set()
        if self.metronome_thread and self.metronome_thread.is_alive():
            self.metronome_thread.join(timeout=0.1)  # Brief wait for clean shutdown
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.play_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        
        # Clear measure highlighting
        if hasattr(self.image_widget, 'toggle_current_measure_highlight'):
            self.image_widget.toggle_current_measure_highlight(False)
        
        # Reset counters
        self.current_beat = 0
        self.current_measure = 0
        self.current_group_index = 0
        self.current_measure_in_group = 0
        self.total_beats_played = 0
        self.metronome_start_time = None
        
        if show_completion_message:
            self.metronome_status.setText("🎵 Performance complete!")
            QMessageBox.information(self, "Performance Complete", 
                                  "🎵 You've reached the end of the score!\n\nGreat job!")
        else:
            self.metronome_status.setText("Ready")
    
    def metronome_tick(self):
        """Play a metronome tick and track measure progress with timing correction"""
        if not self.is_playing:
            return
        
        # Check if we need to advance to next measure (for beat 1 of new measure)
        if self.current_beat >= self.beats_per_measure:
            self.current_beat = 0
            self.current_measure_in_group += 1
            
            # Check for time signature change in the new measure
            self._check_and_update_time_signature()
            
            # Update visual highlight for the new measure BEFORE playing the beat 1 sound
            self._update_measure_highlight()
        
        # Play metronome tick (now synchronized with highlight)
        self.metronome.play_tick()
        
        # Update beat counter
        self.current_beat += 1
        self.total_beats_played += 1
        
        # No need for timer management - precision thread handles timing
        
        # Update metronome status with current position
        if hasattr(self.image_widget, 'measure_boxes') and self.image_widget.measure_boxes:
            beat_text = f"♪ G{self.current_group_index + 1} M{self.current_measure_in_group + 1} Beat {self.current_beat + 1}/{self.beats_per_measure}"
        else:
            beat_text = f"Beat {self.current_beat + 1}/{self.beats_per_measure}"
        
        bpm = self.bpm_spin.value()
        self.metronome_status.setText(f"Playing at {bpm} BPM - {beat_text}")
    
    def _update_measure_highlight(self):
        """Update visual measure highlight based on current group and measure"""
        if not hasattr(self.image_widget, 'measure_boxes') or not self.image_widget.measure_boxes:
            return
        
        # Get all unique groups and their measure counts
        groups_info = self._get_group_measure_info()
        
        if self.current_group_index < len(groups_info):
            group_info = groups_info[self.current_group_index]
            max_measures_in_group = group_info['measure_count']
            
            if self.current_measure_in_group < max_measures_in_group:
                # Find measure box for current group and measure
                measure_box_index = self._find_measure_box_index(self.current_group_index, self.current_measure_in_group + 1)
                
                if measure_box_index is not None:
                    self.image_widget.set_current_measure(measure_box_index)
                    self.image_widget.toggle_current_measure_highlight(True)
                    print(f"DEBUG: Now playing measure {self.current_measure_in_group + 1} in group {self.current_group_index + 1}")
            else:
                # Current group finished, move to next group
                self.current_group_index += 1
                self.current_measure_in_group = 0
                
                if self.current_group_index < len(groups_info):
                    # Start first measure of next group
                    measure_box_index = self._find_measure_box_index(self.current_group_index, 1)
                    if measure_box_index is not None:
                        self.image_widget.set_current_measure(measure_box_index)
                        self.image_widget.toggle_current_measure_highlight(True)
                        print(f"DEBUG: Moving to group {self.current_group_index + 1}, measure 1")
                else:
                    # All groups finished - try to move to next page
                    print("DEBUG: All groups finished - checking for next page")
                    if self._move_to_next_page():
                        return  # Successfully moved to next page
                    else:
                        # No more pages - stop metronome with completion message
                        print("DEBUG: No more pages - performance complete!")
                        self.stop_metronome(show_completion_message=True)
                        return
        else:
            # No more groups - try to move to next page
            logger.debug("DEBUG: No more groups - checking for next page")
            if self._move_to_next_page():
                return  # Successfully moved to next page
            else:
                # No more pages - stop metronome with completion message
                print("DEBUG: No more pages - performance complete!")
                self.stop_metronome(show_completion_message=True)
                return
    
    def _get_current_page_height(self):
        """Get the height of the current page image"""
        try:
            # Try to get from current image widget
            if hasattr(self.image_widget, 'original_pixmap') and self.image_widget.original_pixmap:
                return self.image_widget.original_pixmap.height()
            
            # Fallback: get from PDF at current DPI
            if self.pdf_document:
                page = self.pdf_document[self.current_page]
                mat = fitz.Matrix(self.dpi_slider.value() / 72.0, self.dpi_slider.value() / 72.0)
                pix = page.get_pixmap(matrix=mat)
                return pix.height
            
            # Default fallback
            return 800
        except Exception as e:
            logger.warning(f"Could not get page height, using default: {e}")
            return 800
    
    def _get_current_page_width(self):
        """Get the width of the current page image"""
        try:
            # Try to get from current image widget
            if hasattr(self.image_widget, 'original_pixmap') and self.image_widget.original_pixmap:
                return self.image_widget.original_pixmap.width()
            
            # Fallback: get from PDF at current DPI
            if self.pdf_document:
                page = self.pdf_document[self.current_page]
                mat = fitz.Matrix(self.dpi_slider.value() / 72.0, self.dpi_slider.value() / 72.0)
                pix = page.get_pixmap(matrix=mat)
                return pix.width
            
            # Default fallback
            return 600
        except Exception as e:
            logger.warning(f"Could not get page width, using default: {e}")
            return 600
    
    def _move_to_next_page(self):
        """Move to next page if available and has stored results. Returns True if successful."""
        if self.current_page + 1 >= self.total_pages:
            return False  # No more pages
        
        # Check if next page has stored results
        if not self.all_pages_results:
            return False  # No stored results available
        
        next_page_key = str(self.current_page + 2)  # +2 because pages are 1-indexed in results
        if next_page_key not in self.all_pages_results['pages']:
            return False  # Next page not processed
        
        # Move to next page
        old_page = self.current_page
        self.current_page += 1
        
        logger.debug(f"DEBUG: Moving from page {old_page + 1} to page {self.current_page + 1}")
        
        # Update UI
        self.page_spin.setValue(self.current_page + 1)
        self.load_current_page()  # This will automatically load stored results
        
        # Reset metronome position for new page
        self.current_group_index = 0
        self.current_measure_in_group = 0
        self.current_beat = 0
        
        # Start highlighting from first measure of new page
        if hasattr(self.image_widget, 'measure_boxes') and self.image_widget.measure_boxes:
            self._update_measure_highlight()
            logger.debug(f"DEBUG: Starting page {self.current_page + 1} with {len(self.image_widget.measure_boxes)} measures")
        
        return True  # Successfully moved to next page
    
    def detect_all_pages(self):
        """Run detection on all pages of the PDF"""
        if self.pdf_path is None:
            QMessageBox.warning(self, "Warning", "Please load a PDF file first.")
            return
        
        reply = QMessageBox.question(
            self, "Detect All Pages", 
            f"This will process all {self.total_pages} pages and may take several minutes.\n"
            "Results will be saved as JSON. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Disable controls during processing
        self.detect_btn.setEnabled(False)
        self.detect_all_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Create and start detection thread
        self.detect_all_thread = DetectAllThread(
            self.pdf_path,
            self.dpi_slider.value(),
            self.yolo_conf_slider.value() / 100.0,
            self.alt_preprocessing_check.isChecked()
        )
        
        self.detect_all_thread.progress.connect(self.update_status)
        self.detect_all_thread.page_completed.connect(self.on_page_completed)
        self.detect_all_thread.finished.connect(self.on_detect_all_finished)
        self.detect_all_thread.error.connect(self.on_detect_all_error)
        self.detect_all_thread.start()
    
    def on_page_completed(self, page_num, results):
        """Handle completion of a single page"""
        measures = results['measures']['measure_count']
        symbols = results['symbols']['detection_count']
        time_sigs = len(results['symbols']['time_signatures'])
        detected_sig = results['symbols']['detected_time_signature']
        
        status_text = f"Page {page_num + 1}: {measures} measures, {symbols} symbols"
        if time_sigs > 0 and detected_sig:
            status_text += f", time sig: {detected_sig['signature']}"
        
        self.status_label.setText(status_text)
    
    def on_detect_all_finished(self, all_results):
        """Handle completion of all pages detection"""
        # Store results in memory
        self.all_pages_results = all_results
        
        total_pages = all_results['metadata']['total_pages']
        total_measures = sum([page['measures']['measure_count'] for page in all_results['pages'].values()])
        total_symbols = sum([page['symbols']['detection_count'] for page in all_results['pages'].values()])
        
        # Count time signatures across all pages
        total_time_sigs = sum([len(page['symbols']['time_signatures']) for page in all_results['pages'].values()])
        
        # Find most common time signature across all pages
        all_time_sigs = []
        for page in all_results['pages'].values():
            if page['symbols']['detected_time_signature']:
                all_time_sigs.append(page['symbols']['detected_time_signature']['signature'])
        
        most_common_time_sig = max(set(all_time_sigs), key=all_time_sigs.count) if all_time_sigs else "Not detected"
        
        # Show completion message
        message = (
            f"Successfully processed {total_pages} pages:\n"
            f"• Total measures detected: {total_measures}\n"
            f"• Total symbols detected: {total_symbols}\n"
            f"• Time signatures detected: {total_time_sigs}\n"
            f"• Most common time signature: {most_common_time_sig}\n\n"
            f"Results saved to JSON file."
        )
        
        QMessageBox.information(self, "Detection Complete", message)
        
        # Save results to JSON file
        self.save_results_to_json()
        
        # Load current page results from stored data
        self.load_stored_page_results()
        
        # Re-enable controls
        self.detect_btn.setEnabled(True)
        self.detect_all_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText("All pages detection complete!")
    
    def on_detect_all_error(self, error_msg):
        """Handle detection error"""
        QMessageBox.critical(self, "Detection Error", f"All pages detection failed:\n{error_msg}")
        self.detect_btn.setEnabled(True)
        self.detect_all_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText("All pages detection failed")
    
    def load_stored_page_results(self):
        """Load detection results for current page from stored data"""
        logger.debug(f"DEBUG: load_stored_page_results called for page {self.current_page + 1}")
        
        if not self.all_pages_results:
            logger.debug("DEBUG: No all_pages_results available")
            return
        
        current_page_key = str(self.current_page + 1)
        if current_page_key not in self.all_pages_results['pages']:
            logger.debug(f"DEBUG: Page {current_page_key} not found in stored results")
            return
        
        page_data = self.all_pages_results['pages'][current_page_key]
        logger.debug(f"DEBUG: Found page data for {current_page_key}")
        
        # Extract measure detection results
        measures = page_data['measures']
        logger.debug(f"DEBUG: Measures data - count: {measures.get('measure_count', 0)}, barlines: {len(measures.get('barlines', []))}")
        # Store original ratio coordinates
        self.detection_results = {
            'staff_lines': measures.get('staff_lines', []),
            'barlines': measures.get('barlines', []),
            'measure_count': measures.get('measure_count', 0),
            'staff_systems': measures.get('staff_systems', []),
            'system_groups': measures.get('system_groups', []),
            'barlines_with_systems': measures.get('barlines_with_systems', []),
            'detected_brackets': measures.get('detected_brackets', []),
            'bracket_candidates': measures.get('bracket_candidates', [])
        }
        
        # Convert ratio coordinates to pixels for display
        pixel_results = self._convert_results_to_pixels(self.detection_results)
        
        # Extract YOLO detection results
        symbols = page_data['symbols']
        yolo_detections = symbols.get('detections', [])
        
        # Convert to format expected by image widget
        yolo_class_names = {}
        if yolo_detections:
            try:
                with open('stage4_class_mapping.json', 'r') as f:
                    import json
                    class_data = json.load(f)
                    yolo_class_names = class_data['class_mapping']
            except FileNotFoundError:
                print("Warning: stage4_class_mapping.json not found")
        
        # Process bracket information for image widget (use pixel coordinates)
        bracket_candidates = pixel_results.get('bracket_candidates', [])
        verified_brackets = pixel_results.get('detected_brackets', [])
        
        # Update image widget with results (using pixel coordinates)
        logger.debug(f"DEBUG: Updating image widget - staff_lines: {len(pixel_results['staff_lines'])}, staff_systems: {len(pixel_results['staff_systems'])}")
        self.image_widget.set_detection_results(
            pixel_results['staff_lines'],
            pixel_results['barlines'], 
            pixel_results['measure_count'],
            [],  # barline_candidates
            [],  # staff_lines_with_ranges
            pixel_results['barlines_with_systems'],
            pixel_results['staff_systems'],
            pixel_results['system_groups'],
            None,  # measure_boxes - will be auto-generated
            bracket_candidates,  # bracket_candidates
            verified_brackets    # verified_brackets
        )
        
        
        # Convert YOLO detections from ratios to pixels for display
        pixel_yolo_detections = self._convert_yolo_detections_to_pixels(yolo_detections)
        
        # Set YOLO detections
        self.image_widget.set_yolo_detections(pixel_yolo_detections, yolo_class_names)
        
        # Auto-generate measure boxes for metronome functionality
        self.generate_measure_boxes()
        measure_box_count = len(self.image_widget.measure_boxes) if hasattr(self.image_widget, 'measure_boxes') else 0
        logger.debug(f"DEBUG: Generated {measure_box_count} measure boxes with {len(verified_brackets)} verified brackets")
        
        # Force update display to show all detection results
        self.image_widget.update()
        
        # Update results display
        total_symbols = len(yolo_detections)
        time_sig_info = symbols.get('detected_time_signature')
        time_sig_text = f" - Time Sig: {time_sig_info['signature']}" if time_sig_info else ""
        
        results_text = (
            f"Page {self.current_page + 1} Results:\n"
            f"- {len(self.detection_results['staff_lines'])} staff lines\n"
            f"- {len(self.detection_results['staff_systems'])} staff systems\n"
            f"- {len(self.detection_results['barlines'])} valid barlines\n"
            f"- {self.detection_results['measure_count']} measures\n"
            f"- {total_symbols} music symbols{time_sig_text}"
        )
        
        self.results_label.setText(results_text)
        self.export_btn.setEnabled(True)
        self.extract_measures_btn.setEnabled(True)
        self.yolo_filter_btn.setEnabled(True)
        
        # Update time signature display using effective_time_signature
        effective_time_sig = symbols.get('effective_time_signature')
        if effective_time_sig:
            self.global_time_signature = effective_time_sig
            self.current_time_signature = (effective_time_sig['signature'], effective_time_sig['beats_per_measure'])
            self.beats_per_measure = effective_time_sig['beats_per_measure']
            self.time_sig_label.setText(f"Time Signature: {effective_time_sig['signature']}")
        elif time_sig_info:
            # Fallback to detected time signature if effective not available
            self.current_time_signature = (time_sig_info['signature'], time_sig_info['beats_per_measure'])
            self.beats_per_measure = time_sig_info['beats_per_measure']
            self.time_sig_label.setText(f"Time Signature: {time_sig_info['signature']}")
        
        logger.debug(f"Loaded stored results for page {self.current_page + 1}: {self.detection_results['measure_count']} measures, {total_symbols} symbols")
    
    def _precision_metronome_thread(self):
        """High-precision metronome thread using sleep with drift correction"""
        beat_count = 0
        start_time = time.time()
        
        while not self.metronome_stop_event.is_set():
            # Calculate when the next beat should occur
            next_beat_time = start_time + (beat_count * self.expected_beat_interval_ms / 1000.0)
            current_time = time.time()
            
            # Sleep until the next beat time
            sleep_time = next_beat_time - current_time
            if sleep_time > 0:
                # Use high-precision sleep
                if sleep_time > 0.001:  # If more than 1ms, use regular sleep
                    time.sleep(sleep_time - 0.001)
                
                # Busy-wait for the last millisecond for maximum precision
                while time.time() < next_beat_time:
                    pass
            
            # Check if we should stop
            if self.metronome_stop_event.is_set():
                break
            
            # Emit beat signal to main thread
            self.metronome_beat_signal.emit()
            beat_count += 1
    
    def _check_and_update_time_signature(self):
        """Check if current measure has a time signature change and update accordingly"""
        if not hasattr(self, 'all_pages_results') or not self.all_pages_results:
            return
            
        current_page_key = str(self.current_page + 1)
        if current_page_key not in self.all_pages_results['pages']:
            return
            
        page_data = self.all_pages_results['pages'][current_page_key]
        measure_time_signatures = page_data['measures'].get('measure_time_signatures', [])
        
        if not measure_time_signatures:
            return
            
        # Calculate absolute measure number (across all groups)
        groups_info = self._get_group_measure_info()
        absolute_measure_number = 0
        
        for i in range(self.current_group_index):
            if i < len(groups_info):
                absolute_measure_number += groups_info[i]['measure_count']
        
        absolute_measure_number += self.current_measure_in_group + 1
        
        # Find time signature for current measure
        for measure_info in measure_time_signatures:
            if measure_info['measure_number'] == absolute_measure_number:
                new_time_sig = measure_info['time_signature']
                old_beats = self.beats_per_measure
                new_beats = new_time_sig['beats_per_measure']
                
                # Check if time signature actually changed
                if new_beats != old_beats:
                    self.beats_per_measure = new_beats
                    self.current_time_signature = (new_time_sig['signature'], new_beats)
                    self.time_sig_label.setText(f"Time Signature: {new_time_sig['signature']}")
                    print(f"DEBUG: Time signature changed to {new_time_sig['signature']} at measure {absolute_measure_number}")
                    
                    # Reset current beat to ensure proper timing with new beat count
                    self.current_beat = 0
                break
    
    def _get_group_measure_info(self):
        """Get information about each system group and their measure counts"""
        if not hasattr(self.image_widget, 'measure_boxes') or not self.image_widget.measure_boxes:
            return []
        
        groups_info = {}
        for box in self.image_widget.measure_boxes:
            group_idx = box.get('system_group_index', 0)
            measure_idx = box.get('measure_index', 1)
            
            if group_idx not in groups_info:
                groups_info[group_idx] = {'measure_count': 0, 'systems': set()}
            
            groups_info[group_idx]['measure_count'] = max(groups_info[group_idx]['measure_count'], measure_idx)
            groups_info[group_idx]['systems'].add(box.get('system_index', 0))
        
        # Convert to sorted list
        sorted_groups = []
        for group_idx in sorted(groups_info.keys()):
            info = groups_info[group_idx]
            sorted_groups.append({
                'group_index': group_idx,
                'measure_count': info['measure_count'],
                'system_count': len(info['systems'])
            })
        
        return sorted_groups
    
    def _find_measure_box_index(self, target_group_index, target_measure_number):
        """Find the index of a measure box with specific group and measure number"""
        if not hasattr(self.image_widget, 'measure_boxes') or not self.image_widget.measure_boxes:
            return None
        
        for i, box in enumerate(self.image_widget.measure_boxes):
            if (box.get('system_group_index') == target_group_index and 
                box.get('measure_index') == target_measure_number):
                return i
        
        return None
    
    def detect_time_signature(self):
        """Detect time signature from YOLO detections"""
        if not hasattr(self.image_widget, 'yolo_detections') or not self.image_widget.yolo_detections:
            self.time_sig_label.setText("Time Signature: Not detected")
            return
        
        # Find time signature detections
        time_sig_detections = []
        for det in self.image_widget.yolo_detections:
            class_name = det.get("class_name", "").lower()
            if "timesig" in class_name:
                time_sig_detections.append(det)
        
        if not time_sig_detections:
            self.time_sig_label.setText("Time Signature: Not detected")
            return
        
        # Find the leftmost (first) time signature
        leftmost_sig = min(time_sig_detections, key=lambda x: x["bbox"][0])
        class_name = leftmost_sig.get("class_name", "")
        confidence = leftmost_sig.get("confidence", 0)
        
        # Interpret time signature
        time_sig_text, beats_per_measure = self.interpret_time_signature(class_name)
        
        if time_sig_text:
            self.current_time_signature = time_sig_text
            self.beats_per_measure = beats_per_measure
            
            self.time_sig_label.setText(f"Time Signature: {time_sig_text} (confidence: {confidence:.2f})")
            self.time_sig_label.setStyleSheet("color: #2E7D32; font-size: 10px; font-weight: bold;")
            
            logger.debug(f"DEBUG: Detected time signature: {time_sig_text}, beats per measure: {beats_per_measure}")
        else:
            self.time_sig_label.setText(f"Time Signature: {class_name} (unknown format)")
            self.time_sig_label.setStyleSheet("color: #FF6D00; font-size: 10px;")
    
    def interpret_time_signature(self, class_name):
        """Interpret time signature class name to readable format and beats per measure"""
        class_name = class_name.lower()
        
        # Common time signatures - return (display_text, beats_per_measure)
        if "timesigcommon" in class_name:
            return "4/4 (Common Time)", 4
        elif "timesig4" in class_name:
            return "4/4", 4
        elif "timesig3" in class_name:
            return "3/4", 3
        elif "timesig2" in class_name:
            return "2/4", 2
        elif "timesig6" in class_name:
            return "6/8", 6
        elif "timesig9" in class_name:
            return "9/8", 9
        elif "timesig12" in class_name:
            return "12/8", 12
        
        # Try to extract numbers from class name
        import re
        numbers = re.findall(r'\d+', class_name)
        if len(numbers) >= 2:
            numerator = int(numbers[0])
            return f"{numbers[0]}/{numbers[1]}", numerator
        elif len(numbers) == 1:
            # Common patterns
            num = int(numbers[0])
            if num == 4:
                return "4/4", 4
            elif num == 3:
                return "3/4", 3
            elif num == 2:
                return "2/4", 2
            elif num == 6:
                return "6/8", 6
            elif num == 9:
                return "9/8", 9
            elif num == 12:
                return "12/8", 12
        
        return None, 4  # Default to 4/4
    
    def generate_measure_boxes(self):
        """Generate measure bounding boxes using SYSTEM-SPECIFIC barlines"""
        if not hasattr(self, 'detection_results') or not self.detection_results:
            return
        
        results = self.detection_results
        staff_systems = results.get('staff_systems', [])
        barlines_with_systems = results.get('barlines_with_systems', [])
        
        logger.debug(f"Debug - Available staff systems: {len(staff_systems)}")
        logger.debug(f"Debug - Barlines with system info: {len(barlines_with_systems)}")
        
        if not staff_systems:
            logger.debug("Warning: No staff systems available from GUI")
            return
            
        if not barlines_with_systems:
            logger.debug("Warning: No system-specific barlines available")
            return
        
        # Group barlines by SYSTEM GROUP - barlines_with_systems uses system GROUP index
        barlines_by_system_group = {}
        for bl_info in barlines_with_systems:
            system_group_idx = bl_info.get('system_idx', 0)  # This is actually system GROUP index
            x = bl_info.get('x', 0)
            
            if system_group_idx not in barlines_by_system_group:
                barlines_by_system_group[system_group_idx] = []
            barlines_by_system_group[system_group_idx].append(x)
        
        logger.debug(f"Debug - Barlines by SYSTEM GROUP: {barlines_by_system_group}")
        
        # Get system groups to map group index to individual systems
        system_groups = results.get('system_groups', [])
        logger.debug(f"Debug - System groups: {system_groups}")
        
        # Generate measure boxes: Each system group's barlines apply to ALL systems in that group
        measure_boxes = []
        
        # Get bracket information for measure start positions
        brackets = []
        if hasattr(self.image_widget, 'verified_brackets') and self.image_widget.verified_brackets:
            brackets = self.image_widget.verified_brackets
            logger.debug(f"Debug - Found {len(brackets)} brackets for measure start positions")
            if brackets:
                logger.debug(f"Debug - First bracket: {brackets[0]}")
        else:
            logger.debug(f"Debug - No verified brackets found. Widget has verified_brackets: {hasattr(self.image_widget, 'verified_brackets')}")
            if hasattr(self.image_widget, 'verified_brackets'):
                logger.debug(f"Debug - verified_brackets length: {len(self.image_widget.verified_brackets) if self.image_widget.verified_brackets else 0}")
        
        # Get page dimensions for coordinate conversion
        page_width = self._get_current_page_width()
        page_height = self._get_current_page_height()
        logger.debug(f"Debug - Page dimensions: {page_width}x{page_height}")
        
        # Process each system group and apply its barlines to all systems in the group
        for group_idx, system_indices in enumerate(system_groups):
            group_barlines = barlines_by_system_group.get(group_idx, [])
            if not group_barlines:
                logger.debug(f"Debug - System Group {group_idx}: No barlines, skipping systems {system_indices}")
                continue
                
            group_barlines_sorted = sorted(group_barlines)
            
            # Find bracket X coordinate for this system group as measure start
            bracket_x_ratio = 0  # Default fallback
            for bracket in brackets:
                bracket_systems = bracket.get('covered_staff_system_indices', [])
                # Check if this bracket covers systems in current group
                if any(sys_idx in bracket_systems for sys_idx in system_indices):
                    bracket_x_pixel = bracket.get('x', 0)
                    # Convert bracket X coordinate to ratio
                    bracket_x_ratio = bracket_x_pixel / page_width
                    logger.debug(f"Debug - System Group {group_idx}: Using bracket at x={bracket_x_pixel} (ratio={bracket_x_ratio:.3f}) as measure start")
                    break
            
            extended_group_barlines = [bracket_x_ratio] + group_barlines_sorted
            
            logger.debug(f"Debug - System Group {group_idx}: barlines {group_barlines_sorted}")
            logger.debug(f"      Applying to systems: {system_indices}")
            
            # Apply these barlines to ALL systems in this group
            for sys_idx in system_indices:
                if sys_idx >= len(staff_systems):
                    continue
                
                system = staff_systems[sys_idx]
                logger.debug(f"Debug - System {sys_idx} (Group {group_idx}): y={system['top']} to {system['bottom']}")
                
                # Create measures for this system using group's barlines
                system_measure_count = 0
                
                for i in range(len(extended_group_barlines) - 1):
                    x1_ratio = extended_group_barlines[i]
                    x2_ratio = extended_group_barlines[i + 1]
                    
                    # Convert ratio coordinates to pixels for width calculation
                    x1_pixel = x1_ratio * page_width
                    x2_pixel = x2_ratio * page_width
                    
                    # Skip if measure is too narrow (in pixels)
                    if x2_pixel - x1_pixel < 20:
                        logger.debug(f"Debug - Skipping narrow measure: {x2_pixel - x1_pixel:.1f} pixels (ratios: {x1_ratio:.3f} to {x2_ratio:.3f})")
                        continue
                    
                    system_measure_count += 1
                    
                    # Calculate optimal Y range considering adjacent systems
                    # Convert system coordinates to pixels for the calculation
                    system_pixel = {
                        'top': system['top'] * page_height,
                        'bottom': system['bottom'] * page_height,
                        'height': (system['bottom'] - system['top']) * page_height
                    }
                    
                    # Convert all systems to pixels for context
                    all_systems_pixel = []
                    for sys in staff_systems:
                        all_systems_pixel.append({
                            'top': sys['top'] * page_height,
                            'bottom': sys['bottom'] * page_height,
                            'height': (sys['bottom'] - sys['top']) * page_height
                        })
                    
                    logger.debug(f"  Debug - System pixel coords: top={system_pixel['top']:.1f}, bottom={system_pixel['bottom']:.1f}, height={system_pixel['height']:.1f}")
                    
                    detector = MeasureDetector(debug=False)  # Create temporary detector for method access
                    y1_pixel, y2_pixel = detector.calculate_optimal_measure_y_range(
                        system_pixel, all_systems_pixel, page_height
                    )
                    
                    logger.debug(f"  Debug - Y range pixels: {y1_pixel:.1f}-{y2_pixel:.1f}")
                    
                    # Convert Y coordinates to ratios
                    y1_ratio = y1_pixel / page_height
                    y2_ratio = y2_pixel / page_height
                    
                    logger.debug(f"  Debug - Y range ratios: {y1_ratio:.3f}-{y2_ratio:.3f}")
                    
                    # Create measure box for this system (all coordinates as ratios)
                    measure_box = {
                        'x': x1_ratio,
                        'y': y1_ratio,
                        'width': x2_ratio - x1_ratio,
                        'height': y2_ratio - y1_ratio,
                        'measure_id': f'P{self.current_page+1}_{sys_idx:02d}_{system_measure_count:03d}',
                        'system_index': sys_idx,
                        'system_group_index': group_idx,
                        'measure_index': system_measure_count
                    }
                    measure_boxes.append(measure_box)
                    logger.debug(f"  Created measure {system_measure_count} for system {sys_idx}: x={x1_ratio:.3f}-{x2_ratio:.3f}, y={y1_ratio:.3f}-{y2_ratio:.3f}")
                
                logger.debug(f"Debug - System {sys_idx} total measures: {system_measure_count}")
        
        self.image_widget.measure_boxes = measure_boxes
        logger.debug(f"DEBUG: Set image_widget.measure_boxes with {len(measure_boxes)} boxes")
        
        # Ensure measure boxes are visible
        if measure_boxes:
            self.image_widget.show_measure_boxes = True
            logger.debug(f"DEBUG: Set show_measure_boxes = True")
            # Update checkbox state to match
            if hasattr(self, 'show_measure_boxes_cb'):
                self.show_measure_boxes_cb.setChecked(True)
        
        # Check current state
        logger.debug(f"DEBUG: image_widget.show_measure_boxes = {self.image_widget.show_measure_boxes}")
        logger.debug(f"DEBUG: len(image_widget.measure_boxes) = {len(self.image_widget.measure_boxes)}")
        
        # Force update display to show measure boxes
        self.image_widget.update()
        logger.debug(f"DEBUG: Called image_widget.update()")
    
    def _merge_nearby_barlines(self, barlines, min_distance=50):
        """Merge barlines that are too close together (likely duplicates)"""
        if not barlines:
            return []
        
        merged = [barlines[0]]  # Start with first barline
        
        for i in range(1, len(barlines)):
            current = barlines[i]
            last_merged = merged[-1]
            
            # If current barline is far enough from last merged, keep it
            if current - last_merged >= min_distance:
                merged.append(current)
            else:
                # Replace last merged with average (or keep the one closer to middle)
                # For simplicity, just keep the first one
                print(f"    Merging nearby barlines: {last_merged} and {current} (distance: {current - last_merged})")
        
        return merged
    
    def extract_measures(self):
        """Extract individual measure images using GUI measure boxes"""
        if not hasattr(self, 'pdf_path') or not self.pdf_path:
            QMessageBox.warning(self, "Warning", "Please load a PDF file first.")
            return
        
        if not hasattr(self, 'detection_results') or not self.detection_results:
            QMessageBox.warning(self, "Warning", "Please run detection first.")
            return
        
        if not hasattr(self.image_widget, 'measure_boxes') or not self.image_widget.measure_boxes:
            QMessageBox.warning(self, "Warning", "Please generate measure boxes first by checking 'Show Measure Boxes'.")
            return
        
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return
        
        try:
            from pathlib import Path
            import fitz
            import json
            from datetime import datetime
            
            # Create output directory structure
            output_path = Path(output_dir)
            page_dir = output_path / f"page_{self.current_page + 1:02d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            
            # Open PDF and get current page
            pdf_document = fitz.open(self.pdf_path)
            page = pdf_document[self.current_page]
            
            # Convert to image
            current_dpi = self.dpi_slider.value()
            mat = fitz.Matrix(current_dpi / 72.0, current_dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            page_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
            
            # Extract measure images using GUI measure boxes
            results = self.detection_results
            measure_boxes = self.image_widget.measure_boxes
            
            # Prepare metadata
            height, width = page_image.shape[:2]
            page_metadata = {
                "page_number": self.current_page + 1,
                "page_dimensions": {"width": width, "height": height},
                "staff_groups": [],
                "system_clusters": results.get('system_groups', []),
                "measures": [],
                "extracted_at": datetime.now().isoformat(),
                "barlines_used": results.get('barlines', []),
                "extraction_method": "GUI_measure_boxes"
            }
            
            # Store staff group information
            staff_systems = results.get('staff_systems', [])
            for i, system in enumerate(staff_systems):
                group_info = {
                    "group_index": int(i),
                    "staff_lines": [{"y": int(y), "index": int(j)} for j, y in enumerate(system.get('lines', []))],
                    "y_range": {
                        "min": int(system.get('top', 0)),
                        "max": int(system.get('bottom', 0))
                    },
                    "center_y": int(system.get('center_y', 0)),
                    "height": int(system.get('height', 0))
                }
                page_metadata["staff_groups"].append(group_info)
            
            # Extract each measure
            extracted_count = 0
            for measure_box in measure_boxes:
                x = measure_box['x']
                y = measure_box['y']
                w = measure_box['width']
                h = measure_box['height']
                
                # Extract measure image
                measure_img = page_image[y:y+h, x:x+w]
                
                # Save measure image
                measure_filename = f"{measure_box['measure_id']}.png"
                measure_path = page_dir / measure_filename
                cv2.imwrite(str(measure_path), measure_img)
                
                # Calculate staff lines relative to measure (if available)
                staff_lines_in_measure = []
                system_idx = measure_box['system_index']
                if system_idx < len(staff_systems):
                    system = staff_systems[system_idx]
                    for j, staff_y in enumerate(system.get('lines', [])):
                        relative_y = int(staff_y - y)
                        if 0 <= relative_y < h:
                            staff_lines_in_measure.append({
                                "y": int(relative_y),
                                "original_y": int(staff_y),
                                "staff_index": int(j),
                                "group_index": int(system_idx)
                            })
                
                # Store measure metadata
                measure_info = {
                    "measure_id": str(measure_box['measure_id']),
                    "filename": str(measure_filename),
                    "measure_number": int(measure_box['measure_index']),
                    "staff_system_index": int(measure_box['system_index']),
                    "system_group_index": int(measure_box.get('system_group_index', 0)),
                    "bounding_box_on_page": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "staff_line_coordinates_in_measure": staff_lines_in_measure
                }
                page_metadata["measures"].append(measure_info)
                extracted_count += 1
            
            # Save page metadata
            metadata_path = page_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(page_metadata, f, indent=2, ensure_ascii=False)
            
            # Save overall metadata
            overall_metadata = {
                "source_file": os.path.basename(self.pdf_path),
                "dpi": current_dpi,
                "total_pages": 1,
                "processed_pages": [self.current_page + 1],
                "pages": {str(self.current_page + 1): page_metadata}
            }
            
            overall_metadata_path = output_path / "metadata.json"
            with open(overall_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(overall_metadata, f, indent=2, ensure_ascii=False)
            
            pdf_document.close()
            
            QMessageBox.information(
                self, "Success", 
                f"Successfully extracted {extracted_count} measures to:\n{output_dir}\n\n"
                f"Page directory: {page_dir}\n"
                f"Metadata files: metadata.json (page and overall)"
            )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to extract measures:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = ScoreEyeGUI()
    window.show()
    
    # Cleanup metronome on exit
    def cleanup():
        window.stop_metronome()
        window.metronome.cleanup()
    
    app.aboutToQuit.connect(cleanup)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()