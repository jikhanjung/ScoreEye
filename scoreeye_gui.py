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
        self.measure_count = 0
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
        
    def set_detection_results(self, staff_lines, barlines, measure_count, barline_candidates=None, staff_lines_with_ranges=None):
        """Set the detection results for overlay"""
        self.staff_lines = staff_lines
        self.barlines = barlines
        self.measure_count = measure_count
        self.barline_candidates = barline_candidates or []
        self.staff_lines_with_ranges = staff_lines_with_ranges or []
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
                
        # Draw barlines (only between staff boundaries)
        if self.show_barlines and self.barlines:
            pen = QPen(QColor(255, 0, 0, 180), 3)  # Red with transparency
            painter.setPen(pen)
            font = QFont("Arial", 12)
            painter.setFont(font)
            
            # Calculate staff bounds for limiting barline drawing
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
        
        # Update display
        self.image_widget.set_detection_results(
            results['staff_lines'],
            results['barlines'],
            results['measure_count'],
            results.get('barline_candidates', []),
            results.get('staff_lines_with_ranges', [])
        )
        
        # Update results label
        candidates_count = len(results.get('barline_candidates', []))
        self.results_label.setText(
            f"Detected:\n"
            f"- {len(results['staff_lines'])} staff lines\n"
            f"- {candidates_count} barline candidates\n"
            f"- {len(results['barlines'])} valid barlines\n"
            f"- {results['measure_count']} measures"
        )
        
        # Re-enable controls
        self.detect_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
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


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = ScoreEyeGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()