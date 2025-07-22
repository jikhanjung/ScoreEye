#!/usr/bin/env python3
"""
ScoreEye GUI - Desktop application for measure detection in sheet music
"""

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QSlider, QGroupBox,
    QMessageBox, QProgressBar, QCheckBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QAction, QImage
import fitz  # PyMuPDF
import cv2
import numpy as np
from detect_measure import MeasureDetector


class DetectionThread(QThread):
    """Thread for running detection without freezing the GUI"""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, pdf_path, page_num, dpi, use_alternative_preprocessing=False):
        super().__init__()
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.dpi = dpi
        self.use_alternative_preprocessing = use_alternative_preprocessing
        
    def run(self):
        try:
            self.progress.emit("Initializing detector...")
            detector = MeasureDetector(debug=False)
            
            self.progress.emit(f"Loading page {self.page_num + 1}...")
            results = detector.detect_measures_from_pdf(
                self.pdf_path, self.page_num, self.dpi, self.use_alternative_preprocessing
            )
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


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
        self.show_measure_boxes = False  # Show measure bounding boxes
        self.show_bracket_candidates = False  # Show bracket candidates
        self.show_brackets = False  # Show verified brackets
        self.measure_boxes = []  # List of measure bounding boxes
        self.bracket_candidates = []  # List of bracket candidate lines
        self.verified_brackets = []  # List of verified brackets
        self.measure_count = 0
        self.staff_systems = []
        self.system_groups = []
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
        self.update_display()
        
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
        self.update_display()
        
    def set_scale(self, scale_factor):
        """Set the scale factor for display"""
        self.scale_factor = scale_factor
        self.update_display()
        
    def toggle_staff_lines(self, show):
        """Toggle staff line overlay"""
        self.show_staff_lines = show
        self.update_display()
        
    def toggle_barlines(self, show):
        """Toggle barline overlay"""
        self.show_barlines = show
        self.update_display()
        
    def toggle_candidates(self, show):
        """Toggle barline candidates overlay"""
        self.show_candidates = show
        self.update_display()
        
    def toggle_system_groups(self, show):
        """Toggle system group clustering overlay"""
        self.show_system_groups = show
        self.update_display()
        
    def toggle_bracket_candidates(self, show):
        """Toggle bracket candidate overlay"""
        self.show_bracket_candidates = show
        self.update_display()
        
    def toggle_brackets(self, show):
        """Toggle verified bracket overlay"""
        self.show_brackets = show
        self.update_display()
        
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
            self.update_display()
            return fit_scale
        return 1.0
        
    def update_display(self):
        """Update the displayed image with overlay"""
        if self.original_pixmap is None:
            return
            
        # Scale the pixmap
        scaled_pixmap = self.original_pixmap.scaled(
            int(self.original_pixmap.width() * self.scale_factor),
            int(self.original_pixmap.height() * self.scale_factor),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Create painter for overlay
        painter = QPainter(scaled_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
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
                # Cluster barline과 일반 barline 구분해서 그리기
                cluster_barlines = []
                regular_barlines = []
                
                for bl in self.barlines_with_systems:
                    if bl.get('is_cluster_barline', False):
                        cluster_barlines.append(bl)
                    else:
                        regular_barlines.append(bl)
                
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
        if self.show_measure_boxes and self.measure_boxes:
            pen = QPen(QColor(0, 255, 0, 150), 2)  # Green with transparency
            painter.setPen(pen)
            font = QFont("Arial", 10)
            painter.setFont(font)
            
            for i, box in enumerate(self.measure_boxes):
                # Scale coordinates
                x_scaled = int(box['x'] * self.scale_factor)
                y_scaled = int(box['y'] * self.scale_factor)
                width_scaled = int(box['width'] * self.scale_factor)
                height_scaled = int(box['height'] * self.scale_factor)
                
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
                        print(f"Debug: Skipping invalid candidate: {candidate}")
                except Exception as e:
                    print(f"Debug: Error drawing candidate {candidate}: {e}")
        
        # Draw verified brackets
        if self.show_brackets and self.verified_brackets:
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
            
        painter.end()
        
        self.setPixmap(scaled_pixmap)


class ScoreEyeGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.pdf_path = None
        self.pdf_document = None
        self.current_page = 0
        self.total_pages = 0
        self.detection_results = None
        
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
        
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        detection_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        detection_layout.addWidget(self.status_label)
        
        detection_group.setLayout(detection_layout)
        left_layout.addWidget(detection_group)
        
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
        self.show_measure_boxes_cb.setChecked(False)
        self.show_measure_boxes_cb.setToolTip("Preview measure extraction bounding boxes")
        self.show_measure_boxes_cb.toggled.connect(self.toggle_measure_boxes)
        display_layout.addWidget(self.show_measure_boxes_cb)
        
        self.show_bracket_candidates_cb = QCheckBox("Show Bracket Candidates")
        self.show_bracket_candidates_cb.setChecked(False)
        self.show_bracket_candidates_cb.setToolTip("Show bracket candidate vertical lines")
        self.show_bracket_candidates_cb.toggled.connect(self.toggle_bracket_candidates)
        display_layout.addWidget(self.show_bracket_candidates_cb)
        
        self.show_brackets_cb = QCheckBox("Show Verified Brackets")
        self.show_brackets_cb.setChecked(False)
        self.show_brackets_cb.setToolTip("Show verified bracket detections")
        self.show_brackets_cb.toggled.connect(self.toggle_brackets)
        display_layout.addWidget(self.show_brackets_cb)
        
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
                
                # Load first page
                self.load_current_page()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load PDF: {str(e)}")
                
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
            self.results_label.setText("No results yet")
            self.export_btn.setEnabled(False)
            
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
        
        print(f"Debug GUI: detected_brackets count: {len(detected_brackets)}")
        print(f"Debug GUI: raw_bracket_candidates count: {len(raw_bracket_candidates)}")
        if raw_bracket_candidates:
            print(f"Debug GUI: First raw candidate type: {type(raw_bracket_candidates[0])}")
            print(f"Debug GUI: First raw candidate: {raw_bracket_candidates[0]}")
        
        # Filter and store only valid coordinate lists for candidates
        self.bracket_candidates = []
        for candidate in raw_bracket_candidates:
            if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                self.bracket_candidates.append(candidate)
        
        self.verified_brackets = detected_brackets  # These are verified bracket dicts
        
        print(f"Debug GUI: Filtered bracket_candidates count: {len(self.bracket_candidates)}")
        print(f"Debug GUI: Stored verified_brackets count: {len(self.verified_brackets)}")
        if self.bracket_candidates:
            print(f"Debug GUI: First filtered candidate: {self.bracket_candidates[0]}")
        
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
        self.image_widget.update_display()
    
    def toggle_bracket_candidates(self, checked):
        """Toggle bracket candidate display"""
        self.image_widget.toggle_bracket_candidates(checked)
    
    def toggle_brackets(self, checked):
        """Toggle verified bracket display"""
        self.image_widget.toggle_brackets(checked)
    
    def generate_measure_boxes(self):
        """Generate measure bounding boxes using SYSTEM-SPECIFIC barlines"""
        if not hasattr(self, 'detection_results') or not self.detection_results:
            return
        
        results = self.detection_results
        staff_systems = results.get('staff_systems', [])
        barlines_with_systems = results.get('barlines_with_systems', [])
        
        print(f"Debug - Available staff systems: {len(staff_systems)}")
        print(f"Debug - Barlines with system info: {len(barlines_with_systems)}")
        
        if not staff_systems:
            print("Warning: No staff systems available from GUI")
            return
            
        if not barlines_with_systems:
            print("Warning: No system-specific barlines available")
            return
        
        # Group barlines by SYSTEM GROUP - barlines_with_systems uses system GROUP index
        barlines_by_system_group = {}
        for bl_info in barlines_with_systems:
            system_group_idx = bl_info.get('system_idx', 0)  # This is actually system GROUP index
            x = bl_info.get('x', 0)
            
            if system_group_idx not in barlines_by_system_group:
                barlines_by_system_group[system_group_idx] = []
            barlines_by_system_group[system_group_idx].append(x)
        
        print(f"Debug - Barlines by SYSTEM GROUP: {barlines_by_system_group}")
        
        # Get system groups to map group index to individual systems
        system_groups = results.get('system_groups', [])
        print(f"Debug - System groups: {system_groups}")
        
        # Generate measure boxes: Each system group's barlines apply to ALL systems in that group
        measure_boxes = []
        
        # Get bracket information for measure start positions
        brackets = []
        if hasattr(self.image_widget, 'verified_brackets') and self.image_widget.verified_brackets:
            brackets = self.image_widget.verified_brackets
            print(f"Debug - Found {len(brackets)} brackets for measure start positions")
        
        # Process each system group and apply its barlines to all systems in the group
        for group_idx, system_indices in enumerate(system_groups):
            group_barlines = barlines_by_system_group.get(group_idx, [])
            if not group_barlines:
                print(f"Debug - System Group {group_idx}: No barlines, skipping systems {system_indices}")
                continue
                
            group_barlines_sorted = sorted(group_barlines)
            
            # Find bracket X coordinate for this system group as measure start
            bracket_x = 0  # Default fallback
            for bracket in brackets:
                bracket_systems = bracket.get('covered_staff_system_indices', [])
                # Check if this bracket covers systems in current group
                if any(sys_idx in bracket_systems for sys_idx in system_indices):
                    bracket_x = bracket.get('x', 0)
                    print(f"Debug - System Group {group_idx}: Using bracket at x={bracket_x} as measure start")
                    break
            
            extended_group_barlines = [bracket_x] + group_barlines_sorted
            
            print(f"Debug - System Group {group_idx}: barlines {group_barlines_sorted}")
            print(f"      Applying to systems: {system_indices}")
            
            # Apply these barlines to ALL systems in this group
            for sys_idx in system_indices:
                if sys_idx >= len(staff_systems):
                    continue
                
                system = staff_systems[sys_idx]
                print(f"Debug - System {sys_idx} (Group {group_idx}): y={system['top']} to {system['bottom']}")
                
                # Create measures for this system using group's barlines
                system_measure_count = 0
                for i in range(len(extended_group_barlines) - 1):
                    x1 = extended_group_barlines[i]
                    x2 = extended_group_barlines[i + 1]
                    
                    # Skip if measure is too narrow
                    if x2 - x1 < 20:
                        continue
                    
                    system_measure_count += 1
                    
                    # Calculate optimal Y range considering adjacent systems
                    page_height = self.detection_results['original_image'].shape[0]
                    detector = MeasureDetector(debug=False)  # Create temporary detector for method access
                    y1, y2 = detector.calculate_optimal_measure_y_range(
                        system, staff_systems, page_height
                    )
                    
                    # Create measure box for this system
                    measure_box = {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'measure_id': f'P{self.current_page+1}_{sys_idx:02d}_{system_measure_count:03d}',
                        'system_index': sys_idx,
                        'system_group_index': group_idx,
                        'measure_index': system_measure_count
                    }
                    measure_boxes.append(measure_box)
                    print(f"  Created measure {system_measure_count} for system {sys_idx}: x={x1}-{x2}, y={y1}-{y2}")
                
                print(f"Debug - System {sys_idx} total measures: {system_measure_count}")
        
        self.image_widget.measure_boxes = measure_boxes
    
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
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()