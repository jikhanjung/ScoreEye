#!/usr/bin/env python3
"""
ScoreEye - Automatic Measure Detection in Sheet Music
This module detects barlines and counts measures in musical score images.
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import argparse
import fitz  # PyMuPDF
import tempfile
from dataclasses import dataclass
import logging
from datetime import datetime

def setup_logger():
    """Setup logger to save debug messages to dated files in logs/ directory"""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Get existing logger or create new one
    logger = logging.getLogger('ScoreEye')
    
    # Only setup if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Create file handler with dated filename
        current_date = datetime.now().strftime("%Y%m%d")
        log_filename = os.path.join(logs_dir, f"scoreeye_{current_date}.log")
        
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logger()


@dataclass
class BarlineDetectionConfig:
    """Configuration parameters for barline detection using relative ratios"""
    
    # Staff line detection ratios
    staff_same_system_spacing_ratio: float = 2.5      # Lines within same staff system
    staff_grouping_kernel_ratio: float = 0.7          # Kernel height for staff grouping
    
    # Barline validation ratios  
    barline_max_extension_ratio: float = 2.5          # Maximum barline length extension
    barline_roi_margin_ratio: float = 0.25            # ROI margin around intersection
    barline_top_margin_ratio: float = 0.7             # Top margin for barline validation
    barline_bottom_margin_ratio: float = 0.7          # Bottom margin for barline validation  
    barline_max_allowed_extension_ratio: float = 1.2  # Maximum allowed extension beyond staff
    
    # HoughLinesP parameter ratios
    hough_min_line_length_ratio: float = 0.5          # Minimum line length
    hough_max_line_gap_ratio: float = 0.3             # Maximum line gap
    hough_max_barline_length_ratio: float = 2.0       # Maximum barline length
    
    # Staff system detection ratios
    staff_system_normal_spacing_ratio: float = 2.0    # Normal spacing within system
    staff_system_grouping_ratio: float = 2.5          # Staff system grouping threshold
    
    # Multi-system consensus validation ratios
    system_group_clustering_ratio: float = 8.0        # Y-coordinate clustering for system groups (increased for quartet clustering)
    barline_consensus_tolerance: float = 0.5          # X-coordinate tolerance for barline matching
    min_consensus_ratio: float = 0.8                  # Minimum ratio of systems that must have barline
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.barline_max_allowed_extension_ratio < 0.5:
            raise ValueError("barline_max_allowed_extension_ratio must be at least 0.5")
        if self.staff_same_system_spacing_ratio < 1.0:
            raise ValueError("staff_same_system_spacing_ratio must be at least 1.0")
    
    @classmethod
    def create_strict_config(cls):
        """Create configuration for strict barline detection (fewer false positives)"""
        return cls(
            barline_top_margin_ratio=0.5,
            barline_bottom_margin_ratio=0.5,
            barline_max_allowed_extension_ratio=1.0,
            barline_roi_margin_ratio=0.2,
            hough_min_line_length_ratio=0.6
        )
    
    @classmethod
    def create_relaxed_config(cls):
        """Create configuration for relaxed barline detection (more detection, may have false positives)"""
        return cls(
            barline_top_margin_ratio=1.0,
            barline_bottom_margin_ratio=1.0,
            barline_max_allowed_extension_ratio=1.5,
            barline_roi_margin_ratio=0.3,
            hough_min_line_length_ratio=0.3
        )
    
    def adjust_margins(self, scale_factor):
        """Adjust all margin ratios by a scale factor"""
        self.barline_top_margin_ratio *= scale_factor
        self.barline_bottom_margin_ratio *= scale_factor
        self.barline_max_allowed_extension_ratio *= scale_factor
        self.barline_roi_margin_ratio *= scale_factor
    
    def get_description(self):
        """Get human-readable description of current configuration"""
        return f"""Barline Detection Configuration:
  Staff Detection:
    - Same system spacing ratio: {self.staff_same_system_spacing_ratio}
    - Grouping kernel ratio: {self.staff_grouping_kernel_ratio}
  
  Barline Validation:
    - Top margin ratio: {self.barline_top_margin_ratio}
    - Bottom margin ratio: {self.barline_bottom_margin_ratio}
    - Max extension ratio: {self.barline_max_allowed_extension_ratio}
    - ROI margin ratio: {self.barline_roi_margin_ratio}
  
  HoughLinesP Parameters:
    - Min line length ratio: {self.hough_min_line_length_ratio}
    - Max line gap ratio: {self.hough_max_line_gap_ratio}
    - Max barline length ratio: {self.hough_max_barline_length_ratio}
  
  Staff System Detection:
    - Normal spacing ratio: {self.staff_system_normal_spacing_ratio}
    - Grouping ratio: {self.staff_system_grouping_ratio}
  
  Multi-System Consensus:
    - System group clustering ratio: {self.system_group_clustering_ratio}
    - Barline consensus tolerance: {self.barline_consensus_tolerance}
    - Minimum consensus ratio: {self.min_consensus_ratio}
"""


class MeasureDetector:
    def __init__(self, debug=False, config=None):
        """Initialize the MeasureDetector.
        
        Args:
            debug (bool): If True, displays intermediate processing steps
            config (BarlineDetectionConfig): Configuration for detection parameters
        """
        self.debug = debug
        self.config = config or BarlineDetectionConfig()
        self.staff_lines = []
        self.barlines = []
        self.binary_img = None
        self.detected_brackets = []  # Store detected bracket information
        
    def load_image(self, image_path):
        """Load and return the image in grayscale.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        return img
    
    def load_pdf_page(self, pdf_path, page_num=0, dpi=300):
        """Load a specific page from PDF and convert to grayscale image.
        
        Args:
            pdf_path (str): Path to the PDF file
            page_num (int): Page number to load (0-indexed)
            dpi (int): Resolution for PDF to image conversion
            
        Returns:
            numpy.ndarray: Grayscale image of the PDF page
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(pdf_path)
        
        if page_num >= pdf_document.page_count:
            pdf_document.close()
            raise ValueError(f"Page {page_num+1} does not exist in PDF (total pages: {pdf_document.page_count})")
        
        # Get page (0-indexed in PyMuPDF)
        page = pdf_document[page_num]
        
        # Convert to image at specified DPI
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to numpy array
        img_data = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        
        # Close PDF
        pdf_document.close()
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
            
        return gray_img
    
    def preprocess_image_alternative(self, img):
        """Alternative preprocessing with fixed threshold for better thin line preservation.
        
        Args:
            img (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: Binary image (white background, black foreground)
        """
        # Skip Gaussian blur for thin lines
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        
        self.binary_img = binary
        
        if self.debug:
            cv2.imshow("Alternative Binary Image", cv2.resize(binary, None, fx=0.5, fy=0.5))
            cv2.waitKey(0)
            
        return binary
    
    def preprocess_image(self, img):
        """Preprocess the image for analysis.
        
        Args:
            img (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: Binary image (white background, black foreground)
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Binary thresholding using Otsu's method
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Store for later use
        self.binary_img = binary
        
        if self.debug:
            cv2.imshow("Binary Image", cv2.resize(binary, None, fx=0.5, fy=0.5))
            cv2.waitKey(0)
            
        return binary
    
    def detect_staff_lines(self, binary_img):
        """Detect horizontal staff lines in the score.
        
        Args:
            binary_img (numpy.ndarray): Binary image
            
        Returns:
            list: Y-coordinates of detected staff lines with their ranges
        """
        # Create horizontal line kernel to enhance staff lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Horizontal projection (sum of pixels in each row)
        horizontal_projection = np.sum(horizontal_lines, axis=1)
        
        if np.max(horizontal_projection) == 0:
            return []
            
        # Normalize projection
        horizontal_projection = horizontal_projection / np.max(horizontal_projection)
        
        # Dynamic threshold based on mean projection value
        threshold = np.mean(horizontal_projection[horizontal_projection > 0]) * 0.5
        threshold = max(threshold, 0.1)  # Minimum threshold
        
        # Find peaks (staff lines have high projection values)
        peaks, properties = find_peaks(horizontal_projection, 
                                     height=threshold,
                                     distance=2)
        
        # Group peaks into staff systems and find their ranges
        staff_lines_with_ranges = []
        current_staff = []
        
        # Calculate average line spacing first
        if len(peaks) > 1:
            spacings = np.diff(peaks)
            avg_spacing = np.median(spacings[spacings < 30])  # Typical staff line spacing
        else:
            avg_spacing = 10
            
        for i, peak in enumerate(peaks):
            if len(current_staff) == 0:
                current_staff.append(peak)
            elif peak - current_staff[-1] < avg_spacing * self.config.staff_same_system_spacing_ratio:  # Lines within same staff
                current_staff.append(peak)
            else:  # New staff system
                if len(current_staff) >= 4:  # Valid staff has at least 4 lines
                    # Find the actual range of each staff line
                    for line_y in current_staff:
                        x_start, x_end = self._find_staff_line_range(horizontal_lines, line_y)
                        staff_lines_with_ranges.append({
                            'y': line_y,
                            'x_start': x_start,
                            'x_end': x_end
                        })
                current_staff = [peak]
        
        # Add last staff
        if len(current_staff) >= 4:
            for line_y in current_staff:
                x_start, x_end = self._find_staff_line_range(horizontal_lines, line_y)
                staff_lines_with_ranges.append({
                    'y': line_y,
                    'x_start': x_start,
                    'x_end': x_end
                })
            
        self.staff_lines = [line['y'] for line in staff_lines_with_ranges]
        self.staff_lines_with_ranges = staff_lines_with_ranges
        
        if self.debug:
            plt.figure(figsize=(10, 6))
            plt.plot(horizontal_projection)
            plt.plot(peaks, horizontal_projection[peaks], "x", color='red')
            plt.title("Horizontal Projection - Staff Line Detection")
            plt.xlabel("Y coordinate")
            plt.ylabel("Normalized projection")
            plt.show()
            
        return staff_lines_with_ranges
    
    def _find_staff_line_range(self, horizontal_lines, y):
        """Find the actual start and end x-coordinates of a staff line.
        
        Args:
            horizontal_lines (numpy.ndarray): Processed image with horizontal lines
            y (int): Y-coordinate of the staff line
            
        Returns:
            tuple: (x_start, x_end) coordinates
        """
        # Get the row at y position
        if y < 0 or y >= horizontal_lines.shape[0]:
            return 0, horizontal_lines.shape[1]
            
        row = horizontal_lines[y, :]
        
        # Find first and last non-zero pixels
        nonzero_indices = np.where(row > 0)[0]
        
        if len(nonzero_indices) == 0:
            # No line found, return reasonable default
            return 0, horizontal_lines.shape[1]
            
        x_start = nonzero_indices[0]
        x_end = nonzero_indices[-1]
        
        return int(x_start), int(x_end)
    
    def get_adaptive_kernel_size(self):
        """Calculate adaptive kernel size based on staff line spacing.
        
        Returns:
            int: Kernel height for vertical morphological operations
        """
        if len(self.staff_lines) < 2:
            return 15  # Default value
        
        # Calculate staff line spacings
        spacings = []
        for i in range(len(self.staff_lines) - 1):
            spacing = self.staff_lines[i+1] - self.staff_lines[i]
            if spacing < 30:  # Only consider intra-staff spacings
                spacings.append(spacing)
        
        if not spacings:
            return 15  # Default if no valid spacings found
            
        avg_spacing = np.median(spacings)
        
        # Kernel height = 70% of staff line spacing
        # This detects staff intersections while ignoring gaps between staff lines
        kernel_height = max(8, int(avg_spacing * self.config.staff_grouping_kernel_ratio))
        return kernel_height

    def preprocess_for_hough(self, img):
        """HoughLinesP에 최적화된 전처리
        
        Args:
            img (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: Processed binary image optimized for line detection
        """
        # 1. 노이즈 제거 (약한 블러링)
        denoised = cv2.medianBlur(img, 3)  # Median 필터로 점 노이즈 제거
        
        # 2. 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. 적응적 이진화 (지역별 최적화)
        binary = cv2.adaptiveThreshold(enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 
                                      blockSize=15, C=10)
        
        # 4. 형태학적 정리 (작은 노이즈 제거)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        if self.debug:
            cv2.imshow("Hough Preprocessed", cv2.resize(cleaned, None, fx=0.5, fy=0.5))
            cv2.waitKey(0)
        
        return cleaned
    
    def detect_all_vertical_lines(self, binary_img):
        """모든 수직에 가까운 선분 검출
        
        Args:
            binary_img (numpy.ndarray): Binary image
            
        Returns:
            list: Detected line segments
        """
        # 매우 관대한 파라미터로 시작
        lines = cv2.HoughLinesP(
            binary_img,
            rho=1,                    # 거리 해상도 (픽셀 단위)
            theta=np.pi/180,          # 각도 해상도 (1도)
            threshold=8,              # 매우 낮은 임계값 (8개 점만 있어도 선분 인정)
            minLineLength=5,          # 최소 5픽셀 길이
            maxLineGap=3              # 최대 3픽셀 갭 허용
        )
        
        return lines if lines is not None else []
    
    def filter_vertical_lines(self, lines, angle_tolerance=15):
        """수직에 가까운 선분만 필터링
        
        Args:
            lines (list): Raw line segments from HoughLinesP
            angle_tolerance (int): Tolerance for vertical angle (degrees)
            
        Returns:
            list: Filtered vertical line segments with metadata
        """
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 각도 계산 (수직선은 90도 또는 -90도)
            if x2 == x1:  # 완전 수직선
                angle = 90
            else:
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                angle = abs(angle)
            
            # 수직에 가까운 선분만 선택 (90도 ± tolerance)
            if angle >= (90 - angle_tolerance):
                vertical_lines.append({
                    'line': line[0],
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2,
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2),
                    'angle': 90 if x2 == x1 else np.arctan((y2-y1)/(x2-x1)) * 180/np.pi
                })
        
        return vertical_lines
    
    def group_lines_by_x_coordinate(self, vertical_lines, x_tolerance=8):
        """X좌표가 비슷한 선분들을 그룹화
        
        Args:
            vertical_lines (list): Vertical line segments with metadata
            x_tolerance (int): Tolerance for grouping by x-coordinate
            
        Returns:
            list: Groups of lines with similar x-coordinates
        """
        if not vertical_lines:
            return []
        
        # X좌표로 정렬
        sorted_lines = sorted(vertical_lines, key=lambda l: l['center_x'])
        
        groups = []
        current_group = [sorted_lines[0]]
        
        for line in sorted_lines[1:]:
            # 현재 그룹의 평균 X좌표와 비교
            group_avg_x = np.mean([l['center_x'] for l in current_group])
            
            if abs(line['center_x'] - group_avg_x) <= x_tolerance:
                current_group.append(line)
            else:
                # 새 그룹 시작
                groups.append(current_group)
                current_group = [line]
        
        # 마지막 그룹 추가
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def analyze_line_group(self, group):
        """선분 그룹을 분석하여 바라인 후보인지 판단
        
        Args:
            group (list): Group of line segments with similar x-coordinates
            
        Returns:
            dict: Analysis results including barline score
        """
        analysis = {
            'center_x': np.mean([l['center_x'] for l in group]),
            'x_std': np.std([l['center_x'] for l in group]),
            'total_length': sum([l['length'] for l in group]),
            'line_count': len(group),
            'y_coverage': max([l['center_y'] for l in group]) - min([l['center_y'] for l in group]),
            'avg_angle': np.mean([l['angle'] for l in group]),
            'angle_consistency': np.std([l['angle'] for l in group])
        }
        
        # 바라인 가능성 점수 계산
        score = self.calculate_barline_score(analysis)
        analysis['barline_score'] = score
        
        return analysis
    
    def calculate_barline_score(self, analysis):
        """바라인 가능성 점수 계산 (0-100)
        
        Args:
            analysis (dict): Line group analysis results
            
        Returns:
            float: Barline possibility score (0-100)
        """
        score = 0
        
        # 1. 수직 정렬 점수 (X좌표 표준편차가 작을수록 높음)
        if analysis['x_std'] < 2:
            score += 30
        elif analysis['x_std'] < 5:
            score += 20
        elif analysis['x_std'] < 10:
            score += 10
        
        # 2. 선분 개수 점수 (많을수록 높음, 단 과도하면 감점)
        line_count = analysis['line_count']
        if 3 <= line_count <= 8:
            score += 25
        elif line_count >= 2:
            score += 15
        
        # 3. Y축 커버리지 점수 (스태프 영역을 잘 커버할수록 높음)
        if analysis['y_coverage'] > 40:
            score += 25
        elif analysis['y_coverage'] > 20:
            score += 15
        
        # 4. 각도 일관성 점수 (모든 선분이 비슷한 각도일수록 높음)
        if analysis['angle_consistency'] < 5:
            score += 20
        elif analysis['angle_consistency'] < 10:
            score += 10
        
        return min(score, 100)
    
    def validate_barline_with_staff(self, barline_analysis, staff_lines):
        """스태프 라인과의 교차를 확인하여 바라인 검증 (완전한 범위 포함)
        
        Args:
            barline_analysis (dict): Barline candidate analysis
            staff_lines (list): List of staff line y-coordinates
            
        Returns:
            dict: Validation results
        """
        center_x = int(barline_analysis['center_x'])
        intersections = []
        
        for staff_y in staff_lines:
            # 바라인 X좌표에서 스태프 라인 주변 확인
            intersection_found = self.check_intersection_at_staff(center_x, staff_y)
            if intersection_found:
                intersections.append(staff_y)
        
        # 기본 교차점 검증
        basic_valid = len(intersections) >= 3
        
        # 완전한 범위 검증 추가
        full_span_valid = False
        if basic_valid and len(staff_lines) >= 5:
            # 5개 staff line 시스템의 경우 최상단-최하단 검증
            staff_groups = self._get_5_line_groups(staff_lines)
            
            for group in staff_groups:
                if self._barline_covers_full_staff_group(center_x, group):
                    full_span_valid = True
                    break
        else:
            # 5개 미만인 경우는 기본 검증만 적용
            full_span_valid = basic_valid
        
        # 교차점 분석
        validation_result = {
            'intersection_count': len(intersections),
            'staff_coverage_ratio': len(intersections) / len(staff_lines) if staff_lines else 0,
            'intersections': intersections,
            'is_valid_barline': basic_valid and full_span_valid,
            'full_span_valid': full_span_valid
        }
        
        return validation_result
    
    def _get_5_line_groups(self, staff_lines):
        """Staff line을 5개씩 그룹화"""
        groups = []
        for i in range(0, len(staff_lines) - 4, 5):
            groups.append(staff_lines[i:i+5])
        return groups
    
    def _barline_covers_full_staff_group(self, x, staff_group):
        """Barline이 5개 staff line 그룹을 완전히 커버하는지 확인 (적절한 길이 포함)"""
        if len(staff_group) != 5:
            return False
        
        top_staff = min(staff_group)
        bottom_staff = max(staff_group)
        
        # 1. 최상단과 최하단에서 교차점 확인
        top_intersect = self.check_intersection_at_staff(x, top_staff)
        bottom_intersect = self.check_intersection_at_staff(x, bottom_staff)
        
        if not (top_intersect and bottom_intersect):
            return False
        
        # 2. 길이 검증: barline이 너무 길지 않아야 함
        if not hasattr(self, 'binary_img') or self.binary_img is None:
            return True  # 이미지 정보가 없으면 기본적으로 허용
        
        # 전체 이미지에서 이 x 좌표의 수직 선분 범위 확인
        column = self.binary_img[:, x]
        vertical_segments = self._find_continuous_segments(column)
        
        # Staff group 범위를 커버하는 가장 긴 세그먼트 찾기
        covering_segment = None
        max_coverage = 0
        
        for start_y, end_y in vertical_segments:
            # 이 세그먼트가 staff group을 얼마나 커버하는지 계산
            coverage = self._calculate_staff_coverage(start_y, end_y, top_staff, bottom_staff)
            if coverage > max_coverage:
                max_coverage = coverage
                covering_segment = (start_y, end_y)
        
        if covering_segment is None:
            return False
        
        # 3. 세그먼트 길이 제한 확인: staff 간격에 상대적 기준
        start_y, end_y = covering_segment
        staff_height = bottom_staff - top_staff
        
        # Staff 간격 계산
        if len(staff_group) >= 2:
            spacings = [staff_group[i+1] - staff_group[i] for i in range(len(staff_group)-1)]
            avg_spacing = np.median(spacings)
        else:
            avg_spacing = 12  # 기본값
            
        max_allowed_length = staff_height + int(avg_spacing * self.config.barline_max_extension_ratio)  # Staff 높이 + 설정된 연장 비율
        actual_length = end_y - start_y + 1
        
        length_valid = actual_length <= max_allowed_length
        
        if self.debug:
            if not length_valid:
                logger.debug(f"      Barline too long: {actual_length}px > {max_allowed_length}px allowed")
                logger.debug(f"      (Staff spacing: {avg_spacing:.1f}px, ratio: {actual_length/avg_spacing:.1f}x)")
            else:
                logger.debug(f"      Barline length OK: {actual_length}px <= {max_allowed_length}px")
                logger.debug(f"      (Ratio: {actual_length/avg_spacing:.1f}x staff spacing)")
        
        return length_valid
    
    def _calculate_staff_coverage(self, segment_start, segment_end, top_staff, bottom_staff):
        """세그먼트가 staff group을 얼마나 커버하는지 계산"""
        # 세그먼트와 staff 영역의 겹치는 부분 계산
        overlap_start = max(segment_start, top_staff - 10)  # 10px 여유
        overlap_end = min(segment_end, bottom_staff + 10)   # 10px 여유
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_length = overlap_end - overlap_start + 1
        staff_height = bottom_staff - top_staff + 20  # 위아래 10px씩 여유
        
        return overlap_length / staff_height  # 0.0 ~ 1.0+ 범위
    
    def check_intersection_at_staff(self, x, staff_y):
        """특정 X좌표에서 스태프 라인과의 교차점 확인 (상대적 기준)
        
        Args:
            x (int): X-coordinate to check
            staff_y (int): Staff line y-coordinate
            
        Returns:
            bool: True if intersection exists
        """
        if not hasattr(self, 'binary_img') or self.binary_img is None:
            return False
        
        # 평균 staff line 간격 계산
        if len(self.staff_lines) >= 2:
            spacings = [self.staff_lines[i+1] - self.staff_lines[i] 
                       for i in range(len(self.staff_lines)-1)]
            avg_spacing = np.median([s for s in spacings if s < 30])  # 30px 이하만 intra-staff 간격
            if len([s for s in spacings if s < 30]) == 0:
                avg_spacing = 12  # 기본값
        else:
            avg_spacing = 12  # 기본값
        
        # 상대적 margin: 간격의 25%
        margin = max(2, int(avg_spacing * self.config.barline_roi_margin_ratio))
        
        # 스태프 라인 주변 영역에서 수직 픽셀 존재 확인
        roi_start = max(0, staff_y - margin)
        roi_end = min(self.binary_img.shape[0], staff_y + margin + 1)
        
        if x < self.binary_img.shape[1]:
            roi_column = self.binary_img[roi_start:roi_end, x]
            return np.any(roi_column > 0)
        
        return False
    
    def select_final_barlines(self, analyzed_groups, staff_lines, min_score=40):
        """최종 바라인 선별
        
        Args:
            analyzed_groups (list): Analyzed line groups
            staff_lines (list): Staff line y-coordinates
            min_score (int): Minimum score threshold
            
        Returns:
            list: Final selected barlines
        """
        final_barlines = []
        
        for group_analysis in analyzed_groups:
            # 1. 점수 기준 1차 필터링
            if group_analysis['barline_score'] < min_score:
                continue
            
            # 2. 스태프 교차 검증
            validation = self.validate_barline_with_staff(group_analysis, staff_lines)
            if not validation['is_valid_barline']:
                continue
            
            # 3. 최종 바라인으로 선택
            barline = {
                'x': int(group_analysis['center_x']),
                'score': group_analysis['barline_score'],
                'staff_intersections': validation['intersection_count'],
                'coverage_ratio': validation['staff_coverage_ratio']
            }
            
            final_barlines.append(barline)
        
        # X좌표 기준 정렬
        final_barlines.sort(key=lambda b: b['x'])
        
        return final_barlines
    
    def auto_tune_hough_parameters(self, binary_img):
        """이미지 특성에 따른 파라미터 자동 조정
        
        Args:
            binary_img (numpy.ndarray): Binary image for analysis
            
        Returns:
            dict: Tuned parameters
        """
        # 이미지 크기 분석
        height, width = binary_img.shape
        pixel_density = np.sum(binary_img > 0) / (height * width)
        
        # 스태프 간격 분석
        if len(self.staff_lines) >= 2:
            avg_staff_spacing = np.median([self.staff_lines[i+1] - self.staff_lines[i] 
                                          for i in range(len(self.staff_lines)-1)])
        else:
            avg_staff_spacing = 12  # 기본값
        
        # 동적 파라미터 계산
        params = {
            'threshold': max(5, int(10 * pixel_density)),
            'minLineLength': max(3, int(avg_staff_spacing * 0.3)),
            'maxLineGap': max(2, int(avg_staff_spacing * 0.2)),
            'x_tolerance': max(5, int(width * 0.005)),  # 이미지 너비의 0.5%
            'angle_tolerance': 25 if pixel_density < 0.1 else 15  # 노이즈 많으면 관대하게
        }
        
        if self.debug:
            logger.debug(f"Auto-tuned parameters: {params}")
        
        return params
    
    def detect_barlines_per_system(self, binary_img):
        """각 staff system별로 독립적으로 barline 검출
        
        Args:
            binary_img (numpy.ndarray): Binary image
            
        Returns:
            list: X-coordinates of detected barlines (전체 통합)
        """
        if len(self.staff_lines) < 3:
            return []
        
        # Store binary image for intersection checking
        self.binary_img = binary_img
        
        # 1. Staff system들을 먼저 그룹화
        staff_systems = self.group_staff_lines_into_systems()
        if not staff_systems:
            if self.debug:
                logger.debug("No complete staff systems found, falling back to global detection")
            return self.detect_barlines_hough_global(binary_img)
        
        all_barlines_by_system = []  # 각 system별 barline들
        
        # 2. 각 staff system별로 독립적으로 검출
        for system_idx, system in enumerate(staff_systems):
            if self.debug:
                logger.debug(f"\n=== Processing Staff System {system_idx + 1} ===")
                logger.debug(f"Y range: {system['top']} - {system['bottom']} (height: {system['height']})")
            
            # System ROI 추출
            roi_margin = int(system['height'] * 0.3)  # 30% 여유
            roi_top = max(0, system['top'] - roi_margin)
            roi_bottom = min(binary_img.shape[0], system['bottom'] + roi_margin)
            
            system_roi = binary_img[roi_top:roi_bottom, :]
            
            if self.debug:
                logger.debug(f"ROI: {roi_top} - {roi_bottom} (expanded with {roi_margin}px margin)")
            
            # 이 ROI에서 barline 검출
            system_barlines = self.detect_barlines_in_roi(system_roi, system, roi_top)
            
            # System에 barline 정보 저장 (상세 정보 포함)
            system_barlines_detailed = []
            for x in system_barlines:
                barline_info = {
                    'x': x,
                    'system_idx': system_idx,
                    'y_start': system['top'] - 5,
                    'y_end': system['bottom'] + 5,
                    'system_height': system['height'],
                    'staff_lines': system['lines']
                }
                system_barlines_detailed.append(barline_info)
            
            all_barlines_by_system.append(system_barlines_detailed)
            system['barlines'] = system_barlines
            system['barline_count'] = len(system_barlines)
            
            if self.debug:
                logger.info(f"Detected {len(system_barlines)} barlines in system {system_idx + 1}: {system_barlines}")
        
        # 3. Multi-system consensus validation 적용
        validated_barlines = self.validate_barlines_with_consensus(all_barlines_by_system)
        
        # 4. 검증된 barline들로 최종 결과 구성
        all_barlines = []
        barlines_with_systems = []
        
        for barline in validated_barlines:
            all_barlines.append(barline['x'])
            barlines_with_systems.append(barline)
        
        # 결과를 self에 저장
        self.staff_systems = staff_systems
        self.barlines_with_systems = barlines_with_systems
        
        if self.debug:
            logger.debug(f"\n=== Final Results (After Consensus Validation) ===")
            logger.debug(f"Total validated barlines: {len(all_barlines)}")
            logger.debug(f"Barline-system assignments: {len(barlines_with_systems)}")
            for system_idx, system in enumerate(staff_systems):
                original_count = system['barline_count']
                validated_count = len([bl for bl in validated_barlines if system_idx in bl.get('systems_with_barline', [])])
                logger.debug(f"System {system_idx + 1}: {original_count} detected → {validated_count} validated")
        
        return sorted(all_barlines)
    
    def detect_barlines_in_roi(self, roi_img, system_info, roi_offset_y):
        """특정 staff system ROI에서 barline 검출
        
        Args:
            roi_img (numpy.ndarray): ROI binary image
            system_info (dict): Staff system metadata
            roi_offset_y (int): ROI의 원본 이미지에서의 y-offset
            
        Returns:
            list: X-coordinates of barlines detected in this ROI
        """
        # ROI에 맞게 staff lines 좌표 조정
        adjusted_staff_lines = [y - roi_offset_y for y in system_info['lines']]
        
        # 파라미터 조정: staff line 간격에 상대적 기준
        avg_spacing = system_info['avg_spacing']
        params = {
            'threshold': max(5, int(system_info['height'] * 0.2)),  # System 높이 기반
            'minLineLength': max(3, int(avg_spacing * self.config.hough_min_line_length_ratio)),
            'maxLineGap': max(2, int(avg_spacing * self.config.hough_max_line_gap_ratio)),
            'angle_tolerance': 20,
            'max_barline_length': system_info['height'] + int(avg_spacing * self.config.hough_max_barline_length_ratio)  # Staff 높이 + 설정된 비율
        }
        
        if self.debug:
            logger.debug(f"  ROI parameters: {params}")
        
        # HoughLinesP 검출
        all_lines = cv2.HoughLinesP(
            roi_img,
            rho=1,
            theta=np.pi/180,
            threshold=params['threshold'],
            minLineLength=params['minLineLength'],
            maxLineGap=params['maxLineGap']
        )
        
        if all_lines is None:
            return []
        
        if self.debug:
            logger.debug(f"  Raw lines detected: {len(all_lines)}")
        
        # 수직성 필터링
        vertical_lines = self.filter_vertical_lines(all_lines, params['angle_tolerance'])
        if self.debug:
            logger.debug(f"  Vertical lines: {len(vertical_lines)}")
        
        if not vertical_lines:
            return []
        
        # X좌표 기반 그룹핑
        x_tolerance = max(5, int(roi_img.shape[1] * 0.01))  # ROI 너비의 1%
        line_groups = self.group_lines_by_x_coordinate(vertical_lines, x_tolerance)
        
        if self.debug:
            logger.debug(f"  Line groups: {len(line_groups)}")
        
        if not line_groups:
            return []
        
        # 각 그룹 분석 및 검증
        final_barlines = []
        for group in line_groups:
            analysis = self.analyze_line_group(group)
            
            # ROI 내에서의 검증 (완화된 기준)
            if analysis['barline_score'] >= 20:  # 더 관대한 점수
                center_x = int(analysis['center_x'])
                
                # 1. 기본 교차점 확인
                intersections = 0
                for staff_y in adjusted_staff_lines:
                    if (0 <= staff_y < roi_img.shape[0] and 
                        0 <= center_x < roi_img.shape[1]):
                        # 교차점 확인 (ROI 좌표계)
                        roi_start = max(0, staff_y - 2)
                        roi_end = min(roi_img.shape[0], staff_y + 3)
                        if np.any(roi_img[roi_start:roi_end, center_x] > 0):
                            intersections += 1
                
                # 2. 완전한 범위 검증: 최상단부터 최하단까지 걸쳐야 함 (적절한 길이 포함)
                if intersections >= 3:  # 최소 3개 교차
                    full_span_valid = self.validate_barline_full_span(
                        group, center_x, adjusted_staff_lines, roi_img, params['max_barline_length'])
                    
                    if full_span_valid:
                        final_barlines.append(center_x)
                        if self.debug:
                            print(f"    Valid barline: x={center_x}, score={analysis['barline_score']:.1f}, "
                                  f"intersections={intersections}, full_span=True")
                    else:
                        if self.debug:
                            print(f"    Rejected barline: x={center_x}, score={analysis['barline_score']:.1f}, "
                                  f"intersections={intersections}, full_span=False")
                elif self.debug:
                    print(f"    Rejected barline: x={center_x}, score={analysis['barline_score']:.1f}, "
                          f"intersections={intersections} (too few)")
        
        return sorted(final_barlines)
    
    def validate_barline_full_span(self, line_group, center_x, adjusted_staff_lines, roi_img, max_allowed_length=None):
        """Barline이 최상단부터 최하단 staff line까지 완전히 걸치는지 검증 (길이 제한 포함)
        
        Args:
            line_group (list): 같은 x 좌표의 선분들
            center_x (int): 그룹의 중심 x 좌표
            adjusted_staff_lines (list): ROI 좌표계의 staff line y 좌표들
            roi_img (numpy.ndarray): ROI binary image
            max_allowed_length (int): 허용되는 최대 barline 길이
            
        Returns:
            bool: True if barline spans from top to bottom staff line with proper length
        """
        if len(adjusted_staff_lines) < 3:
            return False
        
        # 1. 시스템의 최상단과 최하단 staff line 좌표
        top_staff = min(adjusted_staff_lines)
        bottom_staff = max(adjusted_staff_lines) 
        
        # 2. 선분 그룹의 실제 y 범위 계산
        group_y_min = float('inf')
        group_y_max = float('-inf')
        
        for line_info in line_group:
            # HoughLinesP 결과에서 실제 선분 좌표 추출
            x1, y1, x2, y2 = line_info['line']
            group_y_min = min(group_y_min, y1, y2)
            group_y_max = max(group_y_max, y1, y2)
        
        if self.debug:
            logger.debug(f"      Group Y range: {group_y_min:.1f} - {group_y_max:.1f}")
            logger.debug(f"      Staff range: {top_staff} - {bottom_staff}")
        
        # 3. Y 범위 검증: staff line 간격에 상대적인 기준
        # Staff line 간격을 계산 (ROI 좌표계 기준)
        if len(adjusted_staff_lines) >= 2:
            spacings = [adjusted_staff_lines[i+1] - adjusted_staff_lines[i] 
                       for i in range(len(adjusted_staff_lines)-1)]
            avg_spacing = np.median(spacings)
        else:
            avg_spacing = 12  # 기본값
        
        # 상대적 기준으로 여유값 계산
        top_margin = avg_spacing * self.config.barline_top_margin_ratio      # 설정된 top margin 비율
        bottom_margin = avg_spacing * self.config.barline_bottom_margin_ratio   # 설정된 bottom margin 비율  
        max_extension = avg_spacing * self.config.barline_max_allowed_extension_ratio   # 설정된 최대 연장 비율
        
        # 3.1 최소 범위 검증: 최상단과 최하단에 도달해야 함
        reaches_top = group_y_min <= (top_staff + top_margin)
        reaches_bottom = group_y_max >= (bottom_staff - bottom_margin)
        
        # 3.2 최대 범위 검증: 너무 길면 안됨
        extends_too_much_above = group_y_min < (top_staff - max_extension)
        extends_too_much_below = group_y_max > (bottom_staff + max_extension)
        
        if not (reaches_top and reaches_bottom):
            if self.debug:
                logger.debug(f"      Range check failed: reaches_top={reaches_top}, reaches_bottom={reaches_bottom}")
            return False
            
        if extends_too_much_above or extends_too_much_below:
            if self.debug:
                logger.debug(f"      Length check failed: extends_above={extends_too_much_above}, extends_below={extends_too_much_below}")
                logger.debug(f"      Allowed range: {top_staff - max_extension} to {bottom_staff + max_extension}")
            return False
        
        # 4. 실제 픽셀 교차점 검증: 최상단과 최하단에서 교차 확인
        top_intersect = self.check_staff_intersection_in_roi(
            roi_img, center_x, top_staff, avg_spacing)
        bottom_intersect = self.check_staff_intersection_in_roi(
            roi_img, center_x, bottom_staff, avg_spacing)
        
        if self.debug:
            logger.debug(f"      Pixel intersect: top={top_intersect}, bottom={bottom_intersect}")
            logger.debug(f"      Staff spacing: {avg_spacing:.1f}px, margins: top={top_margin:.1f}, bottom={bottom_margin:.1f}, ext={max_extension:.1f}")
        
        # 5. 길이 제한 검증: staff 간격에 상대적 기준
        if max_allowed_length is not None:
            actual_length = group_y_max - group_y_min + 1
            length_valid = actual_length <= max_allowed_length
            
            if not length_valid:
                if self.debug:
                    print(f"      Length check failed: {actual_length:.1f}px > {max_allowed_length}px allowed")
                    print(f"      (Staff spacing: {avg_spacing:.1f}px)")
                return False
            elif self.debug:
                logger.debug(f"      Length check passed: {actual_length:.1f}px <= {max_allowed_length}px")
                logger.debug(f"      (Ratio: {actual_length/avg_spacing:.1f}x staff spacing)")
        
        return top_intersect and bottom_intersect
    
    def check_staff_intersection_in_roi(self, roi_img, x, staff_y, avg_spacing=12):
        """ROI 내에서 특정 위치의 픽셀 교차점 확인 (상대적 기준)
        
        Args:
            roi_img (numpy.ndarray): ROI binary image
            x (int): X coordinate to check
            staff_y (int): Staff line Y coordinate (in ROI coordinates)
            avg_spacing (float): Average staff line spacing for relative margin
            
        Returns:
            bool: True if intersection exists
        """
        if (not (0 <= x < roi_img.shape[1]) or 
            not (0 <= staff_y < roi_img.shape[0])):
            return False
        
        # Staff line 주변 margin: 간격의 25%
        margin = max(2, int(avg_spacing * self.config.barline_roi_margin_ratio))
        
        # Staff line 주변 margin 범위에서 픽셀 확인
        y_start = max(0, staff_y - margin)
        y_end = min(roi_img.shape[0], staff_y + margin + 1)
        
        # 해당 x 좌표의 column에서 교차점 확인
        column_segment = roi_img[y_start:y_end, x]
        return np.any(column_segment > 0)

    def detect_barlines_hough_global(self, binary_img):
        """HoughLinesP 기반 전역 바라인 검출 (백업 방식)
        
        Args:
            binary_img (numpy.ndarray): Binary image
            
        Returns:
            list: X-coordinates of detected barlines
        """
        if len(self.staff_lines) < 3:
            return []
        
        # Store binary image for intersection checking
        self.binary_img = binary_img
        
        # 1. 전처리 최적화
        # HoughLinesP는 원본 바이너리 이미지에서 더 잘 작동하므로 기본 전처리 사용
        processed_img = binary_img.copy()
        
        # 2. 파라미터 자동 조정
        params = self.auto_tune_hough_parameters(processed_img)
        
        # 3. 모든 수직 선분 검출
        all_lines = cv2.HoughLinesP(
            processed_img,
            rho=1,
            theta=np.pi/180,
            threshold=params['threshold'],
            minLineLength=params['minLineLength'],
            maxLineGap=params['maxLineGap']
        )
        
        if all_lines is None:
            all_lines = []
        
        if self.debug:
            logger.debug(f"Raw HoughLinesP detected: {len(all_lines)} lines")
        
        # 4. 수직성 필터링
        vertical_lines = self.filter_vertical_lines(all_lines, params['angle_tolerance'])
        if self.debug:
            logger.debug(f"Vertical lines filtered: {len(vertical_lines)}")
        
        if not vertical_lines:
            return []
        
        # 5. X좌표 기반 그룹핑
        line_groups = self.group_lines_by_x_coordinate(vertical_lines, params['x_tolerance'])
        if self.debug:
            logger.debug(f"Line groups formed: {len(line_groups)}")
        
        if not line_groups:
            return []
        
        # 6. 각 그룹 분석
        analyzed_groups = [self.analyze_line_group(group) for group in line_groups]
        
        # 7. 스태프 기반 검증 및 최종 선별
        min_score = 30  # 관대한 점수 기준
        final_barlines = self.select_final_barlines(analyzed_groups, self.staff_lines, min_score)
        
        if self.debug:
            logger.debug(f"Final barlines selected: {len(final_barlines)}")
            for i, barline in enumerate(final_barlines):
                logger.debug(f"  Barline {i+1}: x={barline['x']}, score={barline['score']:.1f}, "
                      f"intersections={barline['staff_intersections']}")
        
        # Staff system 기반으로 결과 재구성
        staff_systems = self.group_staff_lines_into_systems()
        if not staff_systems:
            if self.debug:
                logger.debug("No complete staff systems found (need 5-line groups)")
            return [b['x'] for b in final_barlines]
        
        # Barline들을 staff system에 할당
        barlines_with_systems = self.assign_barlines_to_staff_systems(
            [b['x'] for b in final_barlines], staff_systems)
        
        # 결과를 self에 저장 (시각화에서 사용)
        self.staff_systems = staff_systems
        self.barlines_with_systems = barlines_with_systems
        
        return [b['x'] for b in final_barlines]
    
    def detect_barlines(self, binary_img):
        """바라인 검출 - Staff System별 독립적 검출
        
        Args:
            binary_img (numpy.ndarray): Binary image
            
        Returns:
            list: X-coordinates of detected barlines
        """
        return self.detect_barlines_per_system(binary_img)

    def detect_barlines_segment_based(self, binary_img):
        """Detect barline candidates using segment-based approach.
        
        Args:
            binary_img (numpy.ndarray): Binary image
            
        Returns:
            list: X-coordinates of detected barline candidates
        """
        if len(self.staff_lines) < 3:
            return []
            
        # Segment-based detection: collect vertical segments from each staff line
        all_candidate_x = []
        
        # Process each staff line individually
        for staff_y in self.staff_lines:
            # Define ROI around this staff line (±5 pixels, 11픽셀 총 크기)
            roi_start = max(0, staff_y - 5)
            roi_end = min(binary_img.shape[0], staff_y + 6)
            
            # Extract ROI
            roi = binary_img[roi_start:roi_end, :]
            
            # Vertical projection within this ROI
            vertical_projection = np.sum(roi, axis=0)
            
            # Find peaks in projection (potential barline intersections)
            if np.max(vertical_projection) > 0:
                # Normalize and find significant projections
                normalized = vertical_projection / np.max(vertical_projection)
                threshold = max(0.05, np.mean(normalized[normalized > 0]) * 0.2)  # 대폭 완화
                
                # Collect x-coordinates with significant projection
                candidates = np.where(normalized > threshold)[0]
                all_candidate_x.extend(candidates.tolist())
                
                # 디버그 출력 추가
                if self.debug:
                    print(f"Staff {staff_y}: max_proj={np.max(vertical_projection):.3f}, "
                          f"threshold={threshold:.3f}, candidates={len(candidates)}")
        
        # Cluster candidate x-coordinates to find barline positions
        if not all_candidate_x:
            return []
            
        clustered_barlines = self._cluster_barline_candidates(all_candidate_x)
        
        if self.debug:
            debug_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
            for x in clustered_barlines:
                cv2.line(debug_img, (x, 0), (x, debug_img.shape[0]), (0, 0, 255), 2)
            cv2.imshow("Barline Candidates", cv2.resize(debug_img, None, fx=0.5, fy=0.5))
            cv2.waitKey(0)
            
        return clustered_barlines
    
    def _cluster_barline_candidates(self, candidate_x_list):
        """Cluster x-coordinates to find barline positions.
        
        Args:
            candidate_x_list (list): List of x-coordinates from all staff lines
            
        Returns:
            list: Clustered barline x-coordinates
        """
        if not candidate_x_list:
            return []
            
        # Sort candidates
        candidates = sorted(candidate_x_list)
        
        # Cluster nearby candidates
        clusters = []
        current_cluster = [candidates[0]]
        
        for x in candidates[1:]:
            if x - current_cluster[-1] <= 8:  # Within 8 pixels - same barline
                current_cluster.append(x)
            else:
                # Finalize current cluster
                if len(current_cluster) >= 1:  # 1개 스태프에서도 허용 (완화된 조건)
                    center = int(np.mean(current_cluster))
                    clusters.append(center)
                current_cluster = [x]
        
        # Handle last cluster
        if len(current_cluster) >= 1:  # 1개 스태프에서도 허용 (완화된 조건)
            center = int(np.mean(current_cluster))
            clusters.append(center)
            
        return clusters
    
    def _is_vertical_line_column(self, binary_img, x):
        """Check if a column represents a vertical line rather than scattered pixels.
        
        Args:
            binary_img (numpy.ndarray): Binary image
            x (int): X-coordinate of column to check
            
        Returns:
            bool: True if this column looks like a vertical line
        """
        column = binary_img[:, x]
        
        # Find continuous segments in this column
        segments = self._find_continuous_segments(column)
        
        if not segments:
            return False
            
        # Look for at least one reasonably long segment
        longest_segment_length = max(end - start + 1 for start, end in segments)
        
        # Must have a segment of reasonable length
        min_segment_length = 8  # Minimum continuous segment
        if len(self.staff_lines) >= 2:
            # Base on staff line spacing
            staff_spacing = np.median([self.staff_lines[i+1] - self.staff_lines[i] 
                                     for i in range(len(self.staff_lines)-1)])
            min_segment_length = max(8, int(staff_spacing * 0.8))
            
        return longest_segment_length >= min_segment_length
    
    def _validate_barline_candidate(self, binary_img, x):
        """Validate if a vertical line candidate could be a barline.
        
        Args:
            binary_img (numpy.ndarray): Binary image
            x (int): X-coordinate to check
            
        Returns:
            bool: True if this could be a valid barline candidate
        """
        if x < 0 or x >= binary_img.shape[1]:
            return False
            
        column = binary_img[:, x]
        
        # Find the longest continuous segment in this column
        segments = self._find_continuous_segments(column)
        
        if not segments:
            return False
            
        # Get the longest segment
        longest_segment = max(segments, key=lambda s: s[1] - s[0])
        longest_length = longest_segment[1] - longest_segment[0] + 1
        
        # Barline should be reasonably long
        min_length = 15
        if len(self.staff_lines) >= 5:
            # Should be at least half the staff height
            staff_height = max(self.staff_lines) - min(self.staff_lines)
            min_length = max(15, int(staff_height * 0.4))
            
        return longest_length >= min_length
    
    def _find_continuous_segments(self, column):
        """Find all continuous segments of black pixels in a column.
        
        Args:
            column (numpy.ndarray): 1D array representing a column of the image
            
        Returns:
            list: List of (start_y, end_y) tuples for each segment
        """
        segments = []
        in_segment = False
        start_y = 0
        
        for y, pixel_value in enumerate(column):
            if pixel_value > 0:  # Black pixel (foreground)
                if not in_segment:
                    in_segment = True
                    start_y = y
            else:  # White pixel (background)
                if in_segment:
                    segments.append((start_y, y - 1))
                    in_segment = False
        
        # Handle segment that goes to the end
        if in_segment:
            segments.append((start_y, len(column) - 1))
            
        return segments
    
    def _segment_intersects_staff_area(self, start_y, end_y):
        """Check if a vertical segment intersects with the staff area.
        
        Args:
            start_y, end_y (int): Start and end y-coordinates of the segment
            
        Returns:
            bool: True if segment intersects staff area
        """
        if not self.staff_lines:
            return True  # If no staff detected, accept all segments
            
        staff_top = min(self.staff_lines)
        staff_bottom = max(self.staff_lines)
        
        # Allow some margin beyond staff boundaries
        margin = (staff_bottom - staff_top) * 0.3
        extended_top = staff_top - margin
        extended_bottom = staff_bottom + margin
        
        # Check if segment overlaps with extended staff area
        return not (end_y < extended_top or start_y > extended_bottom)
    
    def _is_continuous_barline(self, column, start_y, end_y, max_gap=3):
        """Check if a segment is truly continuous (allowing small gaps).
        
        Args:
            column (numpy.ndarray): The column data
            start_y, end_y (int): Segment boundaries
            max_gap (int): Maximum allowed gap in pixels
            
        Returns:
            bool: True if segment is sufficiently continuous
        """
        segment = column[start_y:end_y+1]
        
        # Count gaps in the segment
        gap_count = 0
        current_gap = 0
        
        for pixel in segment:
            if pixel == 0:  # White pixel (gap)
                current_gap += 1
            else:  # Black pixel
                if current_gap > max_gap:
                    gap_count += 1
                current_gap = 0
        
        # Handle gap at the end
        if current_gap > max_gap:
            gap_count += 1
            
        # Allow at most 1 significant gap in a barline
        return gap_count <= 1
    
    def _group_nearby_candidates(self, candidates, max_distance=5):
        """Group nearby barline candidates that likely represent the same line.
        
        Args:
            candidates (list): List of x-coordinates
            max_distance (int): Maximum distance to consider candidates as the same line
            
        Returns:
            list: Grouped candidates (represented by their center)
        """
        if not candidates:
            return []
            
        candidates.sort()
        groups = []
        current_group = [candidates[0]]
        
        for x in candidates[1:]:
            if x - current_group[-1] <= max_distance:
                current_group.append(x)
            else:
                # Finalize current group
                center = sum(current_group) // len(current_group)
                groups.append(center)
                current_group = [x]
        
        # Add last group
        if current_group:
            center = sum(current_group) // len(current_group)
            groups.append(center)
            
        return groups
    
    def filter_barlines(self, barline_candidates):
        """Filter barlines to keep only those that cross exactly 5 staff lines.
        
        Args:
            barline_candidates (list): X-coordinates of potential barlines
            
        Returns:
            list: Filtered barline x-coordinates that cross exactly 5 staff lines
        """
        if len(self.staff_lines) < 5:
            return []
            
        filtered_barlines = []
        
        # Simple grouping: assume staff lines are in groups of 5
        staff_groups = []
        for i in range(0, len(self.staff_lines), 5):
            if i + 4 < len(self.staff_lines):
                group = self.staff_lines[i:i+5]
                staff_groups.append(group)
        
        # If no complete groups, try to find the best 5-line group
        if not staff_groups and len(self.staff_lines) >= 5:
            # Take first 5 lines
            staff_groups = [self.staff_lines[:5]]
        
        # Check each barline candidate
        for x in barline_candidates:
            if x < 0 or x >= self.binary_img.shape[1]:
                continue
                
            # Check against each staff group
            for staff_group in staff_groups:
                if self._is_valid_barline_for_staff(x, staff_group):
                    filtered_barlines.append(x)
                    break  # Found a valid staff, move to next candidate
        
        # Remove duplicates, sort, and merge nearby barlines
        filtered_barlines = sorted(list(set(filtered_barlines)))
        return self._merge_nearby_barlines(filtered_barlines)
    
    def _is_within_staff_bounds_simple(self, x, staff_group):
        """Simple check if barline is within staff horizontal bounds.
        
        Args:
            x (int): X-coordinate of barline
            staff_group (list): List of staff line y-coordinates
            
        Returns:
            bool: True if within reasonable bounds
        """
        if not hasattr(self, 'staff_lines_with_ranges') or not self.staff_lines_with_ranges:
            return True  # No range info, allow all
            
        # Find horizontal ranges for this staff group
        staff_starts = []
        staff_ends = []
        
        for staff_y in staff_group:
            for line_info in self.staff_lines_with_ranges:
                if line_info['y'] == staff_y:
                    staff_starts.append(line_info['x_start'])
                    staff_ends.append(line_info['x_end'])
                    break
        
        if not staff_starts:
            return True  # No range info found
            
        # Check if x is within the staff's horizontal span (with margin)
        min_start = min(staff_starts)
        max_end = max(staff_ends)
        margin = 50  # Allow some margin
        
        return min_start - margin <= x <= max_end + margin
    
    def detect_system_groups(self):
        """Y좌표 기반으로 staff system들을 clustering하여 system group들을 찾는다.
        4중주, 오케스트라 등에서 함께 움직이는 system들을 그룹화한다.
        
        Returns:
            list: 각 system group의 리스트 [[system1_indices], [system2_indices], ...]
        """
        logger.debug(f"detect_system_groups: Starting with {len(getattr(self, 'staff_systems', []))} systems")
        # staff_systems가 초기화되지 않은 경우, group_staff_lines_into_systems()를 먼저 호출
        if not hasattr(self, 'staff_systems') or not self.staff_systems:
            staff_systems = self.group_staff_lines_into_systems()
            if not staff_systems:
                return []
            self.staff_systems = staff_systems
        
        if len(self.staff_systems) < 2:
            # System이 2개 미만이면 clustering 불필요
            return [list(range(len(self.staff_systems)))] if self.staff_systems else []
        
        # 각 staff system의 중심 Y좌표 계산
        system_centers = []
        for i, system_info in enumerate(self.staff_systems):
            # 'lines' key 사용 (group_staff_lines_into_systems에서 반환하는 구조에 맞춤)
            staff_lines = system_info['lines']
            center_y = np.mean(staff_lines)
            system_centers.append({'idx': i, 'center_y': center_y})
        
        # Y좌표로 정렬
        system_centers.sort(key=lambda x: x['center_y'])
        
        # Adaptive clustering threshold 계산
        if len(self.staff_lines) >= 5:
            avg_spacing = np.median([self.staff_lines[i+1] - self.staff_lines[i] 
                                   for i in range(len(self.staff_lines)-1)])
            
            # System 간의 실제 간격들을 분석해서 threshold 결정
            system_gaps = []
            for i in range(1, len(system_centers)):
                gap = system_centers[i]['center_y'] - system_centers[i-1]['center_y']
                system_gaps.append(gap)
            
            if system_gaps:
                # K-means style clustering을 사용해서 quartet 패턴 감지
                sorted_gaps = sorted(system_gaps)
                
                if len(sorted_gaps) >= 3:
                    # 간격들을 히스토그램으로 분석해서 두 개의 클러스터로 분리
                    # 4중주 패턴에서는 보통 작은 간격(quartet 내부)과 큰 간격(quartet 간) 두 그룹이 있음
                    
                    if len(sorted_gaps) >= 6:  # 12개 system의 경우 11개 간격
                        # Jump detection: 간격의 급격한 변화를 찾아서 quartet 경계를 결정
                        print(f"DEBUG: Using Jump detection path (sorted_gaps >= 6)")
                        # 연속된 간격들 사이의 차이를 계산
                        gap_jumps = []
                        for i in range(1, len(sorted_gaps)):
                            jump = sorted_gaps[i] - sorted_gaps[i-1]
                            gap_jumps.append(jump)
                        
                        if gap_jumps:
                            # 큰 jump이 있다면 그것을 기준으로 threshold 설정
                            max_jump = max(gap_jumps)
                            max_jump_idx = gap_jumps.index(max_jump)
                            
                            if max_jump > 50:  # 충분히 큰 jump가 있는 경우
                                # jump 직전과 직후의 중간값을 threshold로 사용
                                small_gap_max = sorted_gaps[max_jump_idx]
                                large_gap_min = sorted_gaps[max_jump_idx + 1]
                                auto_threshold = (small_gap_max + large_gap_min) / 2
                                
                                # Use user-configured threshold if it's significantly different from default
                                if abs(self.config.system_group_clustering_ratio - 8.0) > 0.5:
                                    # User has adjusted threshold, use it instead
                                    cluster_threshold = avg_spacing * self.config.system_group_clustering_ratio
                                    print(f"DEBUG: Override jump detection with user threshold: {cluster_threshold:.1f} (ratio={self.config.system_group_clustering_ratio})")
                                else:
                                    cluster_threshold = auto_threshold
                                
                                if self.debug:
                                    print(f"Jump detection analysis:")
                                    print(f"  All gaps: {[int(g) for g in sorted_gaps]}")
                                    print(f"  Gap jumps: {[int(j) for j in gap_jumps]}")
                                    print(f"  Max jump: {max_jump:.1f} at index {max_jump_idx}")
                                    print(f"  Small gaps max: {small_gap_max:.1f}, Large gaps min: {large_gap_min:.1f}")
                                    print(f"  Quartet threshold: {cluster_threshold:.1f}")
                            else:
                                # jump이 작다면 percentile 방법 사용
                                q60 = np.percentile(sorted_gaps, 60)
                                cluster_threshold = q60
                                
                                if self.debug:
                                    print(f"Percentile-based threshold: {cluster_threshold:.1f} (60th percentile)")
                        else:
                            cluster_threshold = avg_spacing * self.config.system_group_clustering_ratio
                            print(f"DEBUG: Using system_group_clustering_ratio = {self.config.system_group_clustering_ratio}, avg_spacing = {avg_spacing:.1f}, threshold = {cluster_threshold:.1f}")
                    else:
                        # 간격 수가 적을 때는 단순한 방법 사용
                        median_gap = np.median(sorted_gaps)
                        # 중간값보다 1.5배 큰 간격을 quartet 간 간격으로 간주
                        cluster_threshold = median_gap * 1.3
                        
                        if self.debug:
                            print(f"Simple quartet threshold: {cluster_threshold:.1f} (median: {median_gap:.1f} * 1.3)")
                else:
                    cluster_threshold = avg_spacing * self.config.system_group_clustering_ratio
                    print(f"DEBUG: Using system_group_clustering_ratio = {self.config.system_group_clustering_ratio}, avg_spacing = {avg_spacing:.1f}, threshold = {cluster_threshold:.1f}")
            else:
                cluster_threshold = avg_spacing * self.config.system_group_clustering_ratio
                logger.debug(f"DEBUG: Using system_group_clustering_ratio = {self.config.system_group_clustering_ratio}, avg_spacing = {avg_spacing:.1f}, threshold = {cluster_threshold:.1f}")
        else:
            cluster_threshold = 100  # 기본값
        
        # Clustering 수행
        system_groups = []
        current_group = [system_centers[0]['idx']]
        
        for i in range(1, len(system_centers)):
            prev_center = system_centers[i-1]['center_y']
            curr_center = system_centers[i]['center_y']
            
            if curr_center - prev_center <= cluster_threshold:
                # 같은 그룹에 추가
                current_group.append(system_centers[i]['idx'])
            else:
                # 새 그룹 시작
                system_groups.append(current_group)
                current_group = [system_centers[i]['idx']]
        
        # 마지막 그룹 추가
        if current_group:
            system_groups.append(current_group)
        
        if self.debug:
            logger.debug(f"\n=== System Group Clustering ===")
            logger.debug(f"Clustering threshold: {cluster_threshold:.1f} pixels")
            logger.info(f"Detected {len(system_groups)} system group(s):")
            for i, group in enumerate(system_groups):
                group_centers = [system_centers[j]['center_y'] for j in range(len(system_centers)) 
                               if system_centers[j]['idx'] in group]
                logger.debug(f"  Group {i+1}: Systems {group} (Y centers: {group_centers})")
        
        return system_groups
    
    def validate_barlines_with_consensus(self, all_barlines_by_system):
        """Multi-system consensus를 통해 유효한 barline들만 선별한다.
        System group 내의 모든(또는 대부분) system에서 검출되는 barline만 유효한 것으로 간주한다.
        
        Args:
            all_barlines_by_system (list): 각 system별 barline 리스트
            
        Returns:
            list: Consensus validation을 통과한 barline들
        """
        if not all_barlines_by_system:
            return []
        
        system_groups = self.detect_system_groups()
        validated_barlines = []
        
        # Staff line spacing 기반으로 tolerance 계산
        if len(self.staff_lines) >= 5:
            avg_spacing = np.median([self.staff_lines[i+1] - self.staff_lines[i] 
                                   for i in range(len(self.staff_lines)-1)])
            x_tolerance = int(avg_spacing * self.config.barline_consensus_tolerance)
        else:
            x_tolerance = 5  # 기본값
        
        if self.debug:
            logger.debug(f"\n=== Multi-System Barline Consensus Validation ===")
            logger.debug(f"X-coordinate tolerance: {x_tolerance} pixels")
            logger.debug(f"Minimum consensus ratio: {self.config.min_consensus_ratio}")
        
        for group_idx, system_indices in enumerate(system_groups):
            if self.debug:
                logger.debug(f"\nProcessing System Group {group_idx + 1}: {system_indices}")
            
            # 이 그룹의 system들에서 검출된 모든 barline들 수집
            group_barlines = []
            for sys_idx in system_indices:
                if sys_idx < len(all_barlines_by_system):
                    for barline in all_barlines_by_system[sys_idx]:
                        group_barlines.append({
                            'x': barline['x'],
                            'system_idx': sys_idx,
                            'barline_data': barline
                        })
            
            if not group_barlines:
                continue
            
            # X좌표 기준으로 정렬
            group_barlines.sort(key=lambda b: b['x'])
            
            # X좌표가 비슷한 barline들을 clustering
            barline_clusters = []
            current_cluster = [group_barlines[0]]
            
            for i in range(1, len(group_barlines)):
                if group_barlines[i]['x'] - current_cluster[-1]['x'] <= x_tolerance:
                    current_cluster.append(group_barlines[i])
                else:
                    barline_clusters.append(current_cluster)
                    current_cluster = [group_barlines[i]]
            
            if current_cluster:
                barline_clusters.append(current_cluster)
            
            # 각 cluster에서 consensus 검증
            min_required_systems = max(1, int(len(system_indices) * self.config.min_consensus_ratio))
            
            for cluster in barline_clusters:
                # 이 cluster에서 barline을 검출한 system들 확인
                systems_with_barline = set(b['system_idx'] for b in cluster)
                consensus_count = len(systems_with_barline)
                
                if consensus_count >= min_required_systems:
                    # Consensus 통과 - cluster 전체를 관통하는 긴 barline 생성
                    avg_x = int(np.mean([b['x'] for b in cluster]))
                    
                    # Cluster의 전체 Y 범위 계산 (cluster 내 모든 system의 top~bottom)
                    cluster_top = float('inf')
                    cluster_bottom = float('-inf')
                    
                    for sys_idx in system_indices:
                        if sys_idx < len(self.staff_systems):
                            system = self.staff_systems[sys_idx]
                            cluster_top = min(cluster_top, system['top'])
                            cluster_bottom = max(cluster_bottom, system['bottom'])
                    
                    if cluster_top == float('inf'):
                        continue
                    
                    # Cluster 전체를 관통하는 barline 생성
                    cluster_barline = {
                        'x': avg_x,
                        'system_idx': group_idx,  # Group index as identifier
                        'y_start': cluster_top - 10,  # 약간의 여유
                        'y_end': cluster_bottom + 10,  # 약간의 여유
                        'consensus_count': consensus_count,
                        'group_size': len(system_indices),
                        'systems_with_barline': list(systems_with_barline),
                        'is_cluster_barline': True,  # 클러스터 바라인 표시
                        'cluster_height': cluster_bottom - cluster_top
                    }
                    
                    validated_barlines.append(cluster_barline)
                    
                    if self.debug:
                        print(f"  ✓ Barline at x={avg_x} - Consensus: {consensus_count}/{len(system_indices)} systems")
                        print(f"    Systems: {sorted(systems_with_barline)}")
                else:
                    if self.debug:
                        avg_x = int(np.mean([b['x'] for b in cluster]))
                        print(f"  ✗ Barline at x={avg_x} - Insufficient consensus: {consensus_count}/{len(system_indices)} systems")
        
        if self.debug:
            logger.debug(f"\nConsensus validation result: {len(validated_barlines)} validated barlines")
        
        return validated_barlines
    
    def _is_valid_barline_for_staff(self, x, staff_group):
        """Comprehensive check if x-coordinate represents a valid barline for a staff group.
        
        Args:
            x (int): X-coordinate of potential barline
            staff_group (list): List of 5 staff line y-coordinates
            
        Returns:
            bool: True if this is a valid barline
        """
        column = self.binary_img[:, x]
        
        # 1. Check horizontal bounds
        if not self._is_within_staff_bounds_simple(x, staff_group):
            return False
            
        # 2. Find continuous vertical segments in this column
        segments = self._find_continuous_segments(column)
        if not segments:
            return False
            
        # 3. Look for a segment that spans most/all of the staff
        staff_top = min(staff_group)
        staff_bottom = max(staff_group)
        staff_height = staff_bottom - staff_top
        
        valid_segment = None
        for start_y, end_y in segments:
            segment_length = end_y - start_y + 1
            
            # Segment should be reasonably long and cover the staff area
            if (segment_length >= staff_height * 0.5 and  # At least half staff height
                start_y <= staff_top + 10 and              # Starts near or above top
                end_y >= staff_bottom - 10):               # Ends near or below bottom
                valid_segment = (start_y, end_y)
                break
                
        if valid_segment is None:
            return False
            
        # 4. Count intersections with staff lines within this segment
        intersections = 0
        segment_start, segment_end = valid_segment
        
        for staff_y in staff_group:
            # Staff line must be within or very close to the segment
            if segment_start - 3 <= staff_y <= segment_end + 3:
                # Check if there's actually a pixel intersection
                window_start = max(0, staff_y - 1)
                window_end = min(len(column), staff_y + 2)
                
                if np.any(column[window_start:window_end] > 0):
                    intersections += 1
                    
        # Must intersect at least 4 staff lines (relaxed from 5)
        if intersections < 4:
            return False
            
        # 5. Check that it's relatively thin (not a thick blob)
        # Look at adjacent columns to estimate thickness
        thickness = self._estimate_barline_thickness(x, segment_start, segment_end)
        if thickness > 6:  # Barlines should be thin
            return False
            
        return True
    
    def _estimate_barline_thickness(self, x, start_y, end_y):
        """Estimate the thickness of a potential barline.
        
        Args:
            x (int): Center x-coordinate
            start_y, end_y (int): Y-range of the line
            
        Returns:
            int: Estimated thickness in pixels
        """
        thickness = 1  # Count center column
        
        # Check columns to the left and right
        for offset in [1, -1, 2, -2, 3, -3]:
            check_x = x + offset
            if 0 <= check_x < self.binary_img.shape[1]:
                # Count how much of the vertical range has pixels
                segment_pixels = np.sum(self.binary_img[start_y:end_y+1, check_x] > 0)
                segment_length = end_y - start_y + 1
                
                # If this column has significant coverage of the segment, count it
                if segment_pixels >= segment_length * 0.3:
                    thickness += 1
                else:
                    break  # Stop at first gap
                    
        return thickness
    
    def group_staff_lines_into_systems(self):
        """Staff line들을 5개씩 그룹화하여 독립적인 staff system으로 구성
        
        Returns:
            list: List of staff systems, each containing 5 staff lines with metadata
        """
        if len(self.staff_lines) < 5:
            return []
            
        staff_systems = []
        
        # 간격 기반으로 staff line들을 그룹화
        spacings = []
        for i in range(len(self.staff_lines) - 1):
            spacing = self.staff_lines[i+1] - self.staff_lines[i]
            spacings.append(spacing)
        
        if not spacings:
            return []
            
        # 정상적인 staff line 간격 vs staff system 간 간격 구분
        avg_spacing = np.median(spacings)
        normal_spacing_threshold = avg_spacing * self.config.staff_system_normal_spacing_ratio  # 같은 system 내 간격
        
        # 그룹화
        current_system = [self.staff_lines[0]]
        
        for i in range(1, len(self.staff_lines)):
            spacing = self.staff_lines[i] - self.staff_lines[i-1]
            
            if spacing <= normal_spacing_threshold and len(current_system) < 5:
                # 같은 staff system 내
                current_system.append(self.staff_lines[i])
            else:
                # 새로운 staff system 시작 또는 현재 시스템 완성
                if len(current_system) == 5:
                    # 완성된 5선 시스템 저장
                    staff_systems.append({
                        'lines': current_system.copy(),
                        'top': current_system[0],
                        'bottom': current_system[-1],
                        'center_y': (current_system[0] + current_system[-1]) // 2,
                        'height': current_system[-1] - current_system[0],
                        'avg_spacing': np.mean([current_system[j+1] - current_system[j] 
                                               for j in range(len(current_system)-1)])
                    })
                
                current_system = [self.staff_lines[i]]
        
        # 마지막 그룹 처리
        if len(current_system) == 5:
            staff_systems.append({
                'lines': current_system.copy(),
                'top': current_system[0],
                'bottom': current_system[-1], 
                'center_y': (current_system[0] + current_system[-1]) // 2,
                'height': current_system[-1] - current_system[0],
                'avg_spacing': np.mean([current_system[j+1] - current_system[j] 
                                       for j in range(len(current_system)-1)])
            })
            
        if self.debug:
            logger.info(f"Detected {len(staff_systems)} complete staff systems (5 lines each)")
            for i, system in enumerate(staff_systems):
                logger.debug(f"  System {i+1}: y={system['top']}-{system['bottom']}, "
                      f"height={system['height']}, spacing={system['avg_spacing']:.1f}")
        
        return staff_systems
    
    def assign_barlines_to_staff_systems(self, barlines, staff_systems):
        """Barline들을 해당하는 staff system에 할당
        
        Args:
            barlines (list): List of barline x-coordinates  
            staff_systems (list): List of staff system metadata
            
        Returns:
            list: List of barlines with assigned staff system info
        """
        barlines_with_systems = []
        
        for x in barlines:
            # 이 barline이 어떤 staff system들과 교차하는지 확인
            intersecting_systems = []
            
            for system_idx, system in enumerate(staff_systems):
                # Barline이 이 staff system과 교차하는지 확인
                intersects = self.barline_intersects_staff_system(x, system)
                if intersects:
                    intersecting_systems.append(system_idx)
            
            # 교차하는 모든 staff system에 대해 barline 정보 생성
            for system_idx in intersecting_systems:
                system = staff_systems[system_idx]
                barlines_with_systems.append({
                    'x': x,
                    'system_idx': system_idx,
                    'y_start': system['top'] - 5,  # 약간의 여유
                    'y_end': system['bottom'] + 5,
                    'system_height': system['height'],
                    'staff_lines': system['lines']
                })
        
        if self.debug:
            logger.debug(f"Assigned {len(barlines_with_systems)} barline-system pairs")
            for bl in barlines_with_systems:
                logger.debug(f"  Barline x={bl['x']} -> System {bl['system_idx']} "
                      f"(y={bl['y_start']}-{bl['y_end']})")
        
        return barlines_with_systems
    
    def barline_intersects_staff_system(self, x, staff_system):
        """특정 barline이 staff system과 교차하는지 확인
        
        Args:
            x (int): Barline x-coordinate
            staff_system (dict): Staff system metadata
            
        Returns:
            bool: True if barline intersects this staff system
        """
        if not hasattr(self, 'binary_img') or self.binary_img is None:
            return False
            
        # Staff system의 각 staff line과 교차하는지 확인
        intersection_count = 0
        
        for staff_y in staff_system['lines']:
            if self.check_intersection_at_staff(x, staff_y):
                intersection_count += 1
        
        # 5개 staff line 중 최소 3개와 교차해야 유효한 barline
        return intersection_count >= 3

    def _group_staff_lines_exact(self):
        """Group staff lines into systems of exactly 5 lines each.
        
        Returns:
            list: List of staff groups, each containing exactly 5 staff line y-coordinates
        """
        if len(self.staff_lines) < 5:
            return []
            
        staff_groups = []
        
        # Calculate spacing statistics
        spacings = []
        for i in range(len(self.staff_lines) - 1):
            spacing = self.staff_lines[i+1] - self.staff_lines[i]
            spacings.append(spacing)
        
        if not spacings:
            return []
            
        # Use more robust grouping based on spacing patterns
        avg_spacing = np.median(spacings)
        
        # Normal staff line spacing vs. gap between staffs
        normal_spacing = avg_spacing
        max_normal_spacing = normal_spacing * 1.8  # Within same staff
        min_staff_gap = normal_spacing * 3.0       # Between different staffs
        
        # Group lines
        current_group = [self.staff_lines[0]]
        
        for i in range(1, len(self.staff_lines)):
            current_spacing = self.staff_lines[i] - self.staff_lines[i-1]
            
            if current_spacing <= max_normal_spacing:
                # Part of same staff
                current_group.append(self.staff_lines[i])
            else:
                # Gap too large - new staff
                if len(current_group) == 5:
                    staff_groups.append(current_group)
                elif len(current_group) > 5:
                    # If we have more than 5, take the first 5 (most likely to be staff)
                    staff_groups.append(current_group[:5])
                
                current_group = [self.staff_lines[i]]
        
        # Handle last group
        if len(current_group) == 5:
            staff_groups.append(current_group)
        elif len(current_group) > 5:
            staff_groups.append(current_group[:5])
            
        return staff_groups
    
    def _is_within_staff_horizontal_bounds(self, x, staff_group):
        """Check if x-coordinate is within the horizontal bounds of the staff.
        
        Args:
            x (int): X-coordinate to check
            staff_group (list): List of 5 staff line y-coordinates
            
        Returns:
            bool: True if within bounds
        """
        if not hasattr(self, 'staff_lines_with_ranges'):
            return True  # No range info, assume it's valid
            
        # Get horizontal bounds for this staff group
        x_starts = []
        x_ends = []
        
        for staff_y in staff_group:
            for line_info in self.staff_lines_with_ranges:
                if line_info['y'] == staff_y:
                    x_starts.append(line_info['x_start'])
                    x_ends.append(line_info['x_end'])
                    break
        
        if not x_starts:
            return True  # No range info found
            
        min_start = min(x_starts)
        max_end = max(x_ends)
        
        # Allow small margin beyond staff bounds
        margin = 30
        return min_start - margin <= x <= max_end + margin
    
    def _count_staff_intersections(self, x, staff_group):
        """Count how many staff lines this vertical line intersects with high confidence.
        
        Args:
            x (int): X-coordinate of the vertical line
            staff_group (list): List of 5 staff line y-coordinates
            
        Returns:
            int: Number of staff lines intersected (0-5)
        """
        column = self.binary_img[:, x]
        intersections = 0
        
        # Get the main vertical segment that spans the staff area
        staff_top = min(staff_group)
        staff_bottom = max(staff_group)
        
        # Find segments that overlap with staff area
        segments = self._find_continuous_segments(column)
        main_segments = []
        
        for start_y, end_y in segments:
            # Check if segment overlaps significantly with staff area
            if (start_y <= staff_bottom + 10 and end_y >= staff_top - 10 and 
                end_y - start_y + 1 >= 10):  # Minimum segment length
                main_segments.append((start_y, end_y))
        
        # If no significant segments found, this is not a barline
        if not main_segments:
            return 0
            
        # Check intersections with staff lines
        for y in staff_group:
            # Check if any main segment covers this staff line
            for start_y, end_y in main_segments:
                if start_y - 2 <= y <= end_y + 2:
                    intersections += 1
                    break  # Found intersection, move to next staff line
                    
        return intersections
    
    def _is_straight_vertical_line(self, x, max_deviation=3):
        """Check if the vertical content at x forms a reasonably straight line.
        
        Args:
            x (int): X-coordinate to check
            max_deviation (int): Maximum allowed horizontal deviation
            
        Returns:
            bool: True if it's a straight vertical line
        """
        # Check columns around x for vertical content
        start_x = max(0, x - max_deviation)
        end_x = min(self.binary_img.shape[1], x + max_deviation + 1)
        
        # Count vertical pixels in the region
        total_pixels = 0
        center_pixels = 0
        
        for check_x in range(start_x, end_x):
            column_pixels = np.sum(self.binary_img[:, check_x] > 0)
            total_pixels += column_pixels
            
            if check_x == x:
                center_pixels = column_pixels
        
        # The center column should have a significant portion of the vertical content
        if total_pixels == 0:
            return False
            
        center_ratio = center_pixels / total_pixels
        return center_ratio >= 0.3  # At least 30% of pixels should be in center column
    
    def _validates_as_barline(self, x, staff_group):
        """Final validation that this is truly a barline.
        
        Args:
            x (int): X-coordinate of candidate
            staff_group (list): List of staff line y-coordinates
            
        Returns:
            bool: True if this validates as a barline
        """
        column = self.binary_img[:, x]
        
        staff_top = min(staff_group)
        staff_bottom = max(staff_group)
        
        # Find the main vertical segment
        segments = self._find_continuous_segments(column)
        
        # Look for a segment that reasonably spans the staff
        for start_y, end_y in segments:
            segment_length = end_y - start_y + 1
            
            # Must be long enough and span most of the staff
            if (segment_length >= 20 and 
                start_y <= staff_top + 15 and 
                end_y >= staff_bottom - 15):
                
                # Check that it's not too thick (barlines are thin)
                # Look at adjacent columns
                thickness = 1
                for offset in [1, -1, 2, -2]:
                    check_x = x + offset
                    if (0 <= check_x < self.binary_img.shape[1] and 
                        np.sum(self.binary_img[start_y:end_y+1, check_x]) > segment_length * 0.3):
                        thickness += 1
                
                # Barlines should be relatively thin (1-4 pixels wide)
                if thickness <= 4:
                    return True
                    
        return False
    
    def _group_staff_lines(self):
        """Group staff lines into systems."""
        if len(self.staff_lines) < 3:
            return []
            
        staff_groups = []
        
        if len(self.staff_lines) >= 5:
            # Calculate average staff line spacing
            spacings = []
            for i in range(len(self.staff_lines) - 1):
                spacing = self.staff_lines[i+1] - self.staff_lines[i]
                spacings.append(spacing)
            
            avg_spacing = np.median(spacings)
            
            # Group staff lines
            current_group = [self.staff_lines[0]]
            for i in range(1, len(self.staff_lines)):
                if self.staff_lines[i] - current_group[-1] <= avg_spacing * self.config.staff_system_grouping_ratio:
                    current_group.append(self.staff_lines[i])
                else:
                    if len(current_group) >= 4:  # Need at least 4 lines for valid staff
                        staff_groups.append(current_group)
                    current_group = [self.staff_lines[i]]
            
            # Add last group
            if len(current_group) >= 4:
                staff_groups.append(current_group)
        else:
            # If not enough lines, use all detected lines as one group
            if len(self.staff_lines) >= 3:
                staff_groups = [self.staff_lines]
                
        return staff_groups
    
    def _merge_nearby_barlines(self, barlines):
        """Merge barlines that are very close to each other."""
        if not barlines:
            return []
            
        barlines.sort()
        merged = [barlines[0]]
        
        for x in barlines[1:]:
            # If barlines are far enough apart, keep them separate
            if x - merged[-1] > 15:  # Minimum distance between barlines
                merged.append(x)
                
        return merged
        
        self.barlines = filtered_barlines
        return filtered_barlines
    
    def count_measures(self):
        """Count the number of measures based on detected barlines.
        
        Returns:
            int: Number of measures
        """
        # For multi-system consensus validation, use barlines from staff systems
        if hasattr(self, 'staff_systems') and self.staff_systems:
            # Find the system with the most barlines (most reliable)
            max_barlines = 0
            for system in self.staff_systems:
                barline_count = system.get('barline_count', 0)
                if barline_count > max_barlines:
                    max_barlines = barline_count
            
            # Number of measures = number of barlines - 1
            if max_barlines >= 2:
                return max_barlines - 1
            else:
                return 0
        else:
            # Fallback to legacy method
            if len(self.barlines) >= 2:
                return len(self.barlines) - 1
            else:
                return 0
    
    def _coords_to_ratio(self, coords_dict):
        """Convert pixel coordinates to page ratios (0-1)
        
        Args:
            coords_dict: Dictionary containing coordinate values
            
        Returns:
            dict: Coordinates converted to ratios
        """
        if not hasattr(self, 'image') or self.image is None:
            return coords_dict
            
        height, width = self.image.shape[:2]
        converted = coords_dict.copy()
        
        # Convert x coordinates to ratios
        if 'x' in converted:
            converted['x'] = converted['x'] / width
        if 'x1' in converted:
            converted['x1'] = converted['x1'] / width
        if 'x2' in converted:
            converted['x2'] = converted['x2'] / width
        if 'center_x' in converted:
            converted['center_x'] = converted['center_x'] / width
        if 'left' in converted:
            converted['left'] = converted['left'] / width
        if 'right' in converted:
            converted['right'] = converted['right'] / width
            
        # Convert y coordinates to ratios
        if 'y' in converted:
            converted['y'] = converted['y'] / height
        if 'y1' in converted:
            converted['y1'] = converted['y1'] / height
        if 'y2' in converted:
            converted['y2'] = converted['y2'] / height
        if 'center_y' in converted:
            converted['center_y'] = converted['center_y'] / height
        if 'top' in converted:
            converted['top'] = converted['top'] / height
        if 'bottom' in converted:
            converted['bottom'] = converted['bottom'] / height
        if 'y_start' in converted:
            converted['y_start'] = converted['y_start'] / height
        if 'y_end' in converted:
            converted['y_end'] = converted['y_end'] / height
            
        # Convert lists of coordinates
        if 'lines' in converted and isinstance(converted['lines'], list):
            converted['lines'] = [line / height for line in converted['lines']]
            
        return converted
    
    def _list_coords_to_ratio(self, coords_list):
        """Convert list of coordinate dictionaries to ratios
        
        Args:
            coords_list: List of dictionaries containing coordinates
            
        Returns:
            list: List with coordinates converted to ratios
        """
        return [self._coords_to_ratio(item) for item in coords_list]
    
    def visualize_results(self, img, output_path=None):
        """Visualize detected barlines and measure count.
        
        Args:
            img (numpy.ndarray): Original image
            output_path (str, optional): Path to save the visualization
            
        Returns:
            numpy.ndarray: Image with visualization overlay
        """
        # Convert to color for visualization
        if len(img.shape) == 2:
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = img.copy()
            
        # Draw staff lines in blue
        for y in self.staff_lines:
            cv2.line(vis_img, (0, y), (vis_img.shape[1], y), (255, 0, 0), 1)
            
        # Draw barlines in red (within staff system boundaries)
        if hasattr(self, 'barlines_with_systems') and self.barlines_with_systems:
            # 새로운 방식: Staff system 범위 내에서만 그리기
            barline_count = 1
            for bl in self.barlines_with_systems:
                x = bl['x']
                y_start = max(0, bl['y_start'])
                y_end = min(vis_img.shape[0], bl['y_end'])
                
                cv2.line(vis_img, (x, y_start), (x, y_end), (0, 0, 255), 2)
                
                # Add barline number (첫 번째 시스템에만)
                if bl['system_idx'] == 0 or (bl['system_idx'] > 0 and 
                    not any(b['x'] == x and b['system_idx'] < bl['system_idx'] 
                           for b in self.barlines_with_systems)):
                    cv2.putText(vis_img, str(barline_count), (x-10, y_start + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    barline_count += 1
                    
        else:
            # 기존 방식 (호환성)
            for i, x in enumerate(self.barlines):
                cv2.line(vis_img, (x, 0), (x, vis_img.shape[0]), (0, 0, 255), 2)
                
                # Add barline number
                cv2.putText(vis_img, str(i+1), (x-10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                       
        # Add measure count
        measure_count = self.count_measures()
        cv2.putText(vis_img, f"Measures: {measure_count}", (20, vis_img.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
        if output_path:
            cv2.imwrite(output_path, vis_img)
            
        return vis_img
    
    def detect_measures(self, image_path, use_alternative_preprocessing=False):
        """Main method to detect measures in a score image.
        
        Args:
            image_path (str): Path to the input image
            use_alternative_preprocessing (bool): Use alternative preprocessing method
            
        Returns:
            dict: Detection results including measure count and barline positions
        """
        # Load image
        img = self.load_image(image_path)
        
        # Preprocess - 대안 방법 선택 가능
        if use_alternative_preprocessing:
            binary = self.preprocess_image_alternative(img)
        else:
            binary = self.preprocess_image(img)
        
        # Detect staff lines
        staff_lines = self.detect_staff_lines(binary)
        logger.info(f"Detected {len(staff_lines)} staff lines")
        
        # Detect barlines
        barline_candidates = self.detect_barlines(binary)
        logger.info(f"Found {len(barline_candidates)} barline candidates")
        
        # Filter barlines
        barlines = self.filter_barlines(barline_candidates)
        logger.info(f"Filtered to {len(barlines)} valid barlines")
        
        # Count measures
        measure_count = self.count_measures()
        
        # Detect brackets (after staff systems are available)
        brackets = []
        if hasattr(self, 'staff_systems') and self.staff_systems:
            brackets = self.detect_brackets(binary)
            logger.info(f"Detected {len(brackets)} brackets")
        
        results = {
            'measure_count': measure_count,
            'barlines': barlines,
            'barline_candidates': barline_candidates,
            'staff_lines': [line['y'] if isinstance(line, dict) else line for line in staff_lines],
            'staff_lines_with_ranges': staff_lines if staff_lines and isinstance(staff_lines[0], dict) else [],
            'barlines_with_systems': getattr(self, 'barlines_with_systems', []),
            'staff_systems': getattr(self, 'staff_systems', []),
            'system_groups': self.detect_system_groups() if hasattr(self, 'staff_systems') else [],
            'detected_brackets': brackets,
            'bracket_candidates': getattr(self, 'bracket_candidates', [])
        }
        
        # Convert coordinates to ratios (0-1) relative to page size
        logger.debug(f"Converting coordinates to ratios - image available: {hasattr(self, 'image') and self.image is not None}")
        if hasattr(self, 'image') and self.image is not None:
            height, width = self.image.shape[:2]
            logger.debug(f"Page dimensions: {width}x{height}")
            
            # Convert barlines (simple x coordinates)
            if results['barlines']:
                results['barlines'] = [x / width for x in results['barlines']]
            
            # Convert barline candidates (x coordinates)  
            if results['barline_candidates']:
                results['barline_candidates'] = [x / width for x in results['barline_candidates']]
                
            # Convert staff lines (y coordinates)
            if results['staff_lines']:
                results['staff_lines'] = [y / height for y in results['staff_lines']]
                
            # Convert staff lines with ranges
            if results['staff_lines_with_ranges']:
                results['staff_lines_with_ranges'] = self._list_coords_to_ratio(results['staff_lines_with_ranges'])
                
            # Convert barlines with systems
            if results['barlines_with_systems']:
                results['barlines_with_systems'] = self._list_coords_to_ratio(results['barlines_with_systems'])
        
        return results
    
    def detect_measures_from_pdf(self, pdf_path, page_num=0, dpi=300, use_alternative_preprocessing=False):
        """Detect measures from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            page_num (int): Page number to process (0-indexed)
            dpi (int): Resolution for PDF conversion
            use_alternative_preprocessing (bool): Use alternative preprocessing method
            
        Returns:
            dict: Detection results including measure count and barline positions
        """
        # Load PDF page as image
        img = self.load_pdf_page(pdf_path, page_num, dpi)
        self.image = img  # Store for coordinate conversion
        
        # Preprocess - 대안 방법 선택 가능
        if use_alternative_preprocessing:
            binary = self.preprocess_image_alternative(img)
        else:
            binary = self.preprocess_image(img)
        
        # Detect staff lines
        staff_lines = self.detect_staff_lines(binary)
        logger.info(f"Detected {len(staff_lines)} staff lines")
        
        # Detect barlines
        barline_candidates = self.detect_barlines(binary)
        logger.info(f"Found {len(barline_candidates)} barline candidates")
        
        # Filter barlines
        barlines = self.filter_barlines(barline_candidates)
        logger.info(f"Filtered to {len(barlines)} valid barlines")
        
        # Count measures
        measure_count = self.count_measures()
        
        # Detect brackets (after staff systems are available)
        brackets = []
        if hasattr(self, 'staff_systems') and self.staff_systems:
            brackets = self.detect_brackets(binary)
            logger.info(f"Detected {len(brackets)} brackets")
        
        results = {
            'measure_count': measure_count,
            'barlines': barlines,
            'barline_candidates': barline_candidates,
            'staff_lines': [line['y'] if isinstance(line, dict) else line for line in staff_lines],
            'staff_lines_with_ranges': staff_lines if staff_lines and isinstance(staff_lines[0], dict) else [],
            'barlines_with_systems': getattr(self, 'barlines_with_systems', []),
            'staff_systems': getattr(self, 'staff_systems', []),
            'system_groups': self.detect_system_groups() if hasattr(self, 'staff_systems') else [],
            'detected_brackets': brackets,
            'bracket_candidates': getattr(self, 'bracket_candidates', []),
            'original_image': img
        }
        
        # Convert coordinates to ratios (0-1) relative to page size
        logger.debug(f"Converting coordinates to ratios - image available: {hasattr(self, 'image') and self.image is not None}")
        if hasattr(self, 'image') and self.image is not None:
            height, width = self.image.shape[:2]
            logger.debug(f"Page dimensions: {width}x{height}")
            
            # Convert barlines (simple x coordinates)
            if results['barlines']:
                results['barlines'] = [x / width for x in results['barlines']]
            
            # Convert barline candidates (x coordinates)  
            if results['barline_candidates']:
                results['barline_candidates'] = [x / width for x in results['barline_candidates']]
                
            # Convert staff lines (y coordinates)
            if results['staff_lines']:
                results['staff_lines'] = [y / height for y in results['staff_lines']]
                
            # Convert staff lines with ranges
            if results['staff_lines_with_ranges']:
                results['staff_lines_with_ranges'] = self._list_coords_to_ratio(results['staff_lines_with_ranges'])
                
            # Convert barlines with systems
            if results['barlines_with_systems']:
                results['barlines_with_systems'] = self._list_coords_to_ratio(results['barlines_with_systems'])
                
            # Convert staff systems
            if results['staff_systems']:
                results['staff_systems'] = self._list_coords_to_ratio(results['staff_systems'])
                
            # Convert system groups  
            if results['system_groups']:
                results['system_groups'] = self._list_coords_to_ratio(results['system_groups'])
                
            # Convert detected brackets
            if results['detected_brackets']:
                logger.debug(f"Converting {len(results['detected_brackets'])} brackets to ratios")
                results['detected_brackets'] = self._list_coords_to_ratio(results['detected_brackets'])
                
            # Convert bracket candidates
            if results['bracket_candidates']:
                results['bracket_candidates'] = self._list_coords_to_ratio(results['bracket_candidates'])
        
        return results

    # ============================================================================
    # BRACKET DETECTION METHODS (Phase 1-3 implementation)
    # ============================================================================
    
    def detect_brackets(self, binary_img):
        """
        Detect square brackets that group staff systems.
        Implements the 3-phase hybrid approach from devlog/20250722_05_bracket_detection_plan.md
        
        Args:
            binary_img: Binary image (preprocessed)
            
        Returns:
            list: List of detected bracket information dictionaries
        """
        if not hasattr(self, 'staff_systems') or not self.staff_systems:
            if self.debug:
                logger.debug("Warning: No staff systems available for bracket detection")
            return []
        
        # Phase 1: Find vertical bracket candidates using HoughLinesP
        vertical_candidates = self._find_vertical_bracket_candidates(binary_img)
        
        if self.debug:
            logger.info(f"Found {len(vertical_candidates)} vertical line candidates")
            if vertical_candidates:
                logger.debug(f"Phase 1 - First candidate type: {type(vertical_candidates[0])}, value: {vertical_candidates[0]}")
        
        # Phase 2: Verify bracket corners using template matching (simplified for now)
        verified_brackets = self._verify_bracket_candidates(binary_img, vertical_candidates)
        
        if self.debug:
            logger.debug(f"Verified {len(verified_brackets)} brackets after corner verification")
            if verified_brackets:
                logger.debug(f"Phase 2 - First verified type: {type(verified_brackets[0])}, value: {verified_brackets[0]}")
        
        # Phase 2.5: Cluster nearby brackets (merge thick brackets detected as multiple lines)
        clustered_brackets = self._cluster_brackets_by_proximity(verified_brackets)
        
        if self.debug:
            logger.debug(f"Clustered to {len(clustered_brackets)} unique brackets after proximity grouping")
        
        # Phase 3: Extract bracket information and map to staff systems
        bracket_info = self._extract_bracket_information(clustered_brackets)
        
        # Store results
        self.detected_brackets = bracket_info
        # Ensure bracket_candidates only contains raw coordinates [x1, y1, x2, y2]
        self.bracket_candidates = []
        for candidate in vertical_candidates:
            if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                self.bracket_candidates.append(candidate)
        
        if self.debug:
            logger.debug(f"Stored bracket_candidates (raw coords) count: {len(self.bracket_candidates)}")
            if self.bracket_candidates:
                logger.debug(f"First bracket_candidate: {self.bracket_candidates[0]}")
            logger.debug(f"Final bracket detection results:")
            for i, bracket in enumerate(bracket_info):
                logger.debug(f"  Bracket {i}: x={bracket['x']}, y_range=({bracket['y_start']}-{bracket['y_end']}), systems={bracket['covered_staff_system_indices']}")
        
        return bracket_info
    
    def calculate_optimal_measure_y_range(self, system, all_systems, page_height):
        """
        Calculate optimal Y range for measures in a system, considering adjacent systems
        
        Args:
            system: Current system dict with 'top', 'bottom', 'height'
            all_systems: List of all systems on the page (sorted by Y position)
            page_height: Total page height for boundary systems
            
        Returns:
            tuple: (y_start, y_end) for optimal measure range
        """
        current_top = system['top']
        current_bottom = system['bottom'] 
        current_height = system['height']
        
        # Find system index in all_systems
        system_idx = -1
        for i, sys in enumerate(all_systems):
            if sys['top'] == current_top and sys['bottom'] == current_bottom:
                system_idx = i
                break
        
        if system_idx == -1:
            # Fallback: use basic margin
            margin = int(current_height * 0.3)
            return max(0, current_top - margin), min(page_height, current_bottom + margin)
        
        # Calculate Y range based on position
        y_start = current_top
        y_end = current_bottom
        
        # Check above (previous system)
        if system_idx > 0:
            prev_system = all_systems[system_idx - 1]
            gap_above = current_top - prev_system['bottom']
            # Use half of the gap above
            y_start = current_top - int(gap_above * 0.5)
            if hasattr(self, 'debug') and self.debug:
                logger.debug(f"    System {system_idx}: Gap above = {gap_above}px, using {int(gap_above * 0.5)}px")
        else:
            # Top system: extend upward by 100% of system height
            extension = int(current_height * 1.0)  # 100% of system height
            y_start = max(0, current_top - extension)
            if hasattr(self, 'debug') and self.debug:
                logger.debug(f"    System {system_idx} (TOP): Extending upward by {extension}px")
        
        # Check below (next system) 
        if system_idx < len(all_systems) - 1:
            next_system = all_systems[system_idx + 1]
            gap_below = next_system['top'] - current_bottom
            # Use half of the gap below
            y_end = current_bottom + int(gap_below * 0.5)
            if hasattr(self, 'debug') and self.debug:
                logger.debug(f"    System {system_idx}: Gap below = {gap_below}px, using {int(gap_below * 0.5)}px")
        else:
            # Bottom system: extend downward by 100% of system height
            extension = int(current_height * 1.0)  # 100% of system height
            y_end = min(page_height, current_bottom + extension)
            if hasattr(self, 'debug') and self.debug:
                logger.debug(f"    System {system_idx} (BOTTOM): Extending downward by {extension}px")
        
        if hasattr(self, 'debug') and self.debug:
            old_margin = int(current_height * 0.3)
            old_y1, old_y2 = max(0, current_top - old_margin), min(page_height, current_bottom + old_margin)
            logger.debug(f"    Y range optimization: {old_y1}-{old_y2} → {y_start}-{y_end} (height: {old_y2-old_y1} → {y_end-y_start})")
        
        return int(y_start), int(y_end)
    
    def _find_vertical_bracket_candidates(self, binary_img):
        """
        Phase 1: Find vertical line candidates in the left ROI using HoughLinesP
        
        Args:
            binary_img: Binary image
            
        Returns:
            list: List of vertical line candidates as [x1, y1, x2, y2]
        """
        height, width = binary_img.shape
        
        # 1.1. Set ROI - left 15% of image, full height range of staff systems
        all_staff_y = []
        for system in self.staff_systems:
            all_staff_y.extend(system.get('lines', []))
        
        if not all_staff_y:
            if self.debug:
                logger.debug("Warning: No staff lines found for ROI calculation")
            return []
        
        roi_x_end = int(width * 0.15)  # Left 15% of image
        roi_y_start = max(0, int(min(all_staff_y)) - 50)  # 50px margin above top staff
        roi_y_end = min(height, int(max(all_staff_y)) + 50)  # 50px margin below bottom staff
        
        # Extract ROI
        roi = binary_img[roi_y_start:roi_y_end, 0:roi_x_end]
        
        if self.debug:
            logger.debug(f"ROI for bracket detection: x=0-{roi_x_end}, y={roi_y_start}-{roi_y_end}")
            logger.debug(f"ROI size: {roi.shape}")
        
        # 1.2. Calculate dynamic parameters
        if not self.staff_systems:
            return []
        
        # Calculate minimum height: 1.5 times the smallest staff system height
        min_heights = []
        avg_spacings = []
        for system in self.staff_systems:
            if 'height' in system:
                min_heights.append(system['height'])
            if 'avg_spacing' in system:
                avg_spacings.append(system['avg_spacing'])
        
        if min_heights:
            min_line_length = int(min(min_heights) * 1.5)
        else:
            min_line_length = int((roi_y_end - roi_y_start) * 0.3)  # Fallback: 30% of ROI height
        
        if avg_spacings:
            max_line_gap = int(np.mean(avg_spacings) * 0.5)
        else:
            max_line_gap = 10  # Fallback value
        
        if self.debug:
            logger.debug(f"HoughLinesP parameters: minLineLength={min_line_length}, maxLineGap={max_line_gap}")
        
        # 1.3. Apply HoughLinesP
        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        # 1.4. Filter for vertical lines (88-92 degrees)
        vertical_candidates = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Convert back to full image coordinates
                x1_full = x1
                y1_full = y1 + roi_y_start
                x2_full = x2  
                y2_full = y2 + roi_y_start
                
                # Calculate angle
                if x2_full != x1_full:
                    angle = abs(np.arctan2(y2_full - y1_full, x2_full - x1_full) * 180 / np.pi)
                else:
                    angle = 90  # Perfect vertical line
                
                # Filter for near-vertical lines
                if 88 <= angle <= 92:
                    vertical_candidates.append([x1_full, y1_full, x2_full, y2_full])
                    
                    if self.debug:
                        print(f"  Vertical candidate: ({x1_full}, {y1_full})-({x2_full}, {y2_full}), angle={angle:.1f}°")
        
        return vertical_candidates
    
    def _verify_bracket_candidates(self, binary_img, vertical_candidates):
        """
        Phase 2: Verify bracket candidates by checking for horizontal elements
        (Simplified version - full template matching would require actual templates)
        
        Args:
            binary_img: Binary image
            vertical_candidates: List of vertical line candidates
            
        Returns:
            list: List of verified bracket candidates
        """
        verified_brackets = []
        
        for candidate in vertical_candidates:
            x1, y1, x2, y2 = candidate
            
            # Ensure y1 is top, y2 is bottom
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            
            # For simplified verification, check for horizontal elements near the endpoints
            has_top_horizontal = self._check_horizontal_element(binary_img, x1, y1, direction='top')
            has_bottom_horizontal = self._check_horizontal_element(binary_img, x2, y2, direction='bottom')
            
            # A bracket should have both top and bottom horizontal elements
            if has_top_horizontal and has_bottom_horizontal:
                verified_brackets.append(candidate)
                
                if self.debug:
                    print(f"  Verified bracket: ({x1}, {y1})-({x2}, {y2})")
            elif self.debug:
                logger.debug(f"  Rejected candidate: ({x1}, {y1})-({x2}, {y2}) - missing horizontal elements")
        
        return verified_brackets
    
    def _check_horizontal_element(self, binary_img, x, y, direction='top'):
        """
        Check for horizontal elements near a point (simplified bracket corner detection)
        
        Args:
            binary_img: Binary image
            x, y: Point coordinates
            direction: 'top' or 'bottom'
            
        Returns:
            bool: True if horizontal element found
        """
        height, width = binary_img.shape
        
        # Define search area around the point
        search_size = 15
        x_start = max(0, x - 5)
        x_end = min(width, x + search_size)
        y_start = max(0, y - search_size//2)
        y_end = min(height, y + search_size//2)
        
        # Extract search region
        search_region = binary_img[y_start:y_end, x_start:x_end]
        
        if search_region.size == 0:
            return False
        
        # Look for horizontal lines using simple morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))
        horizontal_elements = cv2.morphologyEx(search_region, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Check if we found any horizontal elements
        horizontal_pixels = np.sum(horizontal_elements > 0)
        threshold = 5  # Minimum number of horizontal pixels
        
        return horizontal_pixels >= threshold
    
    def _cluster_brackets_by_proximity(self, verified_brackets):
        """
        Cluster nearby bracket candidates to merge thick brackets detected as multiple lines
        Only merges brackets that are close in X AND have continuous Y ranges (no gaps)
        
        Args:
            verified_brackets: List of verified bracket candidates [x1, y1, x2, y2]
            
        Returns:
            list: List of clustered (merged) bracket candidates
        """
        if not verified_brackets:
            return []
        
        if self.debug:
            logger.debug(f"  Clustering {len(verified_brackets)} bracket candidates")
            logger.debug(f"  Logic: Similar X coordinates (50px) + continuous Y ranges (100px gap) = same bracket")
        
        # Use simple X proximity clustering (ignoring Y gaps)
        clustered = self._cluster_brackets_by_x_proximity(verified_brackets, x_tolerance=50)
        
        if self.debug:
            logger.debug(f"  Final bracket clustering: {len(clustered)} unique brackets")
            for i, cluster in enumerate(clustered):
                logger.debug(f"    Bracket {i}: x_avg={int((cluster[0] + cluster[2])/2)}, y_range=({cluster[1]}-{cluster[3]})")
        
        return clustered
    
    def _cluster_brackets_by_x_proximity(self, brackets, x_tolerance=20):
        """
        Helper method to cluster brackets by X coordinate proximity
        Brackets with similar X coordinates AND continuous Y ranges are merged
        If Y ranges have gaps, they remain separate brackets
        """
        if not brackets:
            return []
        
        # Sort brackets by X coordinate first, then by Y coordinate
        sorted_brackets = sorted(brackets, key=lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        
        # First group by X coordinate
        x_groups = []
        current_x_group = [sorted_brackets[0]]
        x_tolerance = 50  # Similar X position tolerance
        
        for bracket in sorted_brackets[1:]:
            current_x = (bracket[0] + bracket[2]) / 2
            group_x = (current_x_group[0][0] + current_x_group[0][2]) / 2
            
            if abs(current_x - group_x) <= x_tolerance:
                # Similar X - add to current group
                current_x_group.append(bracket)
            else:
                # Different X - start new group
                x_groups.append(current_x_group)
                current_x_group = [bracket]
        
        x_groups.append(current_x_group)  # Don't forget last group
        
        if self.debug:
            logger.debug(f"    Found {len(x_groups)} X-coordinate groups")
        
        # Now within each X group, cluster by Y continuity
        clustered = []
        for group_idx, x_group in enumerate(x_groups):
            if self.debug:
                logger.debug(f"    Processing X-group {group_idx} with {len(x_group)} brackets")
            
            # Sort by Y coordinate within the group
            y_sorted = sorted(x_group, key=lambda b: (b[1] + b[3]) / 2)
            
            # Cluster by Y continuity
            y_clusters = []
            current_y_cluster = [y_sorted[0]]
            
            for bracket in y_sorted[1:]:
                # Check Y continuity with current cluster
                if self._brackets_y_continuous(current_y_cluster, bracket, y_gap_tolerance=100):
                    current_y_cluster.append(bracket)
                    if self.debug:
                        bracket_y = (bracket[1] + bracket[3]) / 2
                        print(f"      Y-continuous: adding bracket at y={int(bracket_y)}")
                else:
                    # Y gap detected - start new cluster
                    y_clusters.append(current_y_cluster)
                    current_y_cluster = [bracket]
                    if self.debug:
                        bracket_y = (bracket[1] + bracket[3]) / 2
                        print(f"      Y-gap detected: starting new cluster at y={int(bracket_y)}")
            
            y_clusters.append(current_y_cluster)  # Don't forget last cluster
            
            # Merge each Y cluster
            for y_cluster in y_clusters:
                merged_bracket = self._merge_bracket_cluster(y_cluster)
                clustered.append(merged_bracket)
        
        return clustered
    
    def _brackets_y_continuous(self, current_cluster, new_bracket, y_gap_tolerance=100):
        """
        Check if a new bracket has continuous Y range with current cluster
        
        Args:
            current_cluster: List of brackets already in cluster
            new_bracket: New bracket to check [x1, y1, x2, y2]
            y_gap_tolerance: Maximum Y gap allowed between brackets
            
        Returns:
            bool: True if Y ranges are continuous (overlapping or close)
        """
        # Get Y range of new bracket
        new_y_start = min(new_bracket[1], new_bracket[3])
        new_y_end = max(new_bracket[1], new_bracket[3])
        
        # Get Y range of current cluster
        cluster_y_coords = []
        for bracket in current_cluster:
            cluster_y_coords.extend([bracket[1], bracket[3]])
        
        cluster_y_start = min(cluster_y_coords)
        cluster_y_end = max(cluster_y_coords)
        
        # Check for overlap or proximity
        # Brackets are continuous if they overlap or gap is small
        y_gap = max(0, max(new_y_start - cluster_y_end, cluster_y_start - new_y_end))
        
        is_continuous = y_gap <= y_gap_tolerance
        
        if self.debug:
            logger.debug(f"        Y continuity check: cluster({cluster_y_start}-{cluster_y_end}) vs new({new_y_start}-{new_y_end}), gap={y_gap}, continuous={is_continuous}")
        
        return is_continuous
    
    def _brackets_similar(self, bracket1, bracket2, tolerance=30):
        """Check if two brackets are similar (for detecting used brackets)"""
        x1_avg = (bracket1[0] + bracket1[2]) / 2
        x2_avg = (bracket2[0] + bracket2[2]) / 2
        y1_avg = (bracket1[1] + bracket1[3]) / 2
        y2_avg = (bracket2[1] + bracket2[3]) / 2
        
        return (abs(x1_avg - x2_avg) <= tolerance and abs(y1_avg - y2_avg) <= tolerance)
    
    def _merge_bracket_cluster(self, bracket_cluster):
        """
        Merge multiple bracket candidates in a cluster into a single representative bracket
        
        Args:
            bracket_cluster: List of bracket candidates [x1, y1, x2, y2] to merge
            
        Returns:
            list: Single merged bracket [x1, y1, x2, y2]
        """
        if len(bracket_cluster) == 1:
            return bracket_cluster[0]
        
        # Calculate average/representative coordinates
        x_coords = []
        y_coords = []
        
        for bracket in bracket_cluster:
            x1, y1, x2, y2 = bracket
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # Use median for more robust merging
        avg_x = int(np.median(x_coords))
        min_y = int(min(y_coords))
        max_y = int(max(y_coords))
        
        # Return merged bracket as [x, min_y, x, max_y] (vertical line format)
        return [avg_x, min_y, avg_x, max_y]
    
    def _extract_bracket_information(self, verified_brackets):
        """
        Phase 3: Extract bracket information and map to staff systems
        
        Args:
            verified_brackets: List of verified bracket candidates
            
        Returns:
            list: List of bracket information dictionaries
        """
        bracket_info = []
        
        for bracket in verified_brackets:
            x1, y1, x2, y2 = bracket
            
            # Calculate bracket properties
            bracket_x = int((x1 + x2) / 2)
            bracket_y_start = min(y1, y2)
            bracket_y_end = max(y1, y2)
            
            # Find staff systems covered by this bracket
            covered_systems = []
            for i, system in enumerate(self.staff_systems):
                system_top = system.get('top', 0)
                system_bottom = system.get('bottom', 0)
                
                # Check if system is within bracket's Y range (with some tolerance)
                tolerance = 20  # pixels
                if (bracket_y_start - tolerance <= system_top <= bracket_y_end + tolerance and
                    bracket_y_start - tolerance <= system_bottom <= bracket_y_end + tolerance):
                    covered_systems.append(i)
            
            # Create bracket information dictionary
            bracket_data = {
                'type': 'bracket',
                'x': bracket_x,
                'y_start': int(bracket_y_start),
                'y_end': int(bracket_y_end),
                'bounding_box': {
                    'x': bracket_x,
                    'y_start': int(bracket_y_start), 
                    'y_end': int(bracket_y_end)
                },
                'covered_staff_system_indices': covered_systems,
                'raw_coordinates': [int(x1), int(y1), int(x2), int(y2)]
            }
            
            bracket_info.append(bracket_data)
        
        return bracket_info


def main():
    parser = argparse.ArgumentParser(description='Detect measures in sheet music images or PDFs')
    parser.add_argument('input', help='Path to the input image or PDF file')
    parser.add_argument('-o', '--output', help='Path to save visualization')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug visualizations')
    parser.add_argument('-p', '--page', type=int, default=1, help='Page number for PDF (1-indexed, default: 1)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF conversion (default: 300)')
    parser.add_argument('--config-preset', choices=['default', 'strict', 'relaxed'], default='default',
                       help='Configuration preset: default (balanced), strict (fewer false positives), relaxed (more detection)')
    parser.add_argument('--top-margin-ratio', type=float, help='Top margin ratio for barline validation (default: 0.7)')
    parser.add_argument('--bottom-margin-ratio', type=float, help='Bottom margin ratio for barline validation (default: 0.7)')
    parser.add_argument('--max-extension-ratio', type=float, help='Maximum extension ratio beyond staff (default: 1.2)')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config_preset == 'strict':
        config = BarlineDetectionConfig.create_strict_config()
    elif args.config_preset == 'relaxed':
        config = BarlineDetectionConfig.create_relaxed_config()
    else:
        config = BarlineDetectionConfig()
    
    # Override with specific parameters if provided
    if args.top_margin_ratio is not None:
        config.barline_top_margin_ratio = args.top_margin_ratio
    if args.bottom_margin_ratio is not None:
        config.barline_bottom_margin_ratio = args.bottom_margin_ratio
    if args.max_extension_ratio is not None:
        config.barline_max_allowed_extension_ratio = args.max_extension_ratio
    
    # Create detector
    detector = MeasureDetector(debug=args.debug, config=config)
    
    if args.debug:
        logger.debug(config.get_description())
    
    try:
        # Check if input is PDF or image
        file_ext = os.path.splitext(args.input)[1].lower()
        
        if file_ext == '.pdf':
            # Process PDF
            logger.debug(f"Processing PDF: {args.input}, Page: {args.page}")
            results = detector.detect_measures_from_pdf(args.input, args.page - 1, args.dpi)
            
            # Use the image from PDF for visualization
            img = results['original_image']
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            # Process image
            results = detector.detect_measures(args.input)
            
            # Load original image for visualization
            img = cv2.imread(args.input)
        
        logger.debug(f"\nDetection Results:")
        logger.debug(f"Number of measures: {results['measure_count']}")
        logger.debug(f"Barline positions: {results['barlines']}")
        
        # Create visualization
        if not args.output:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            if file_ext == '.pdf':
                output_path = f'output/{base_name}_page{args.page}_barlines.png'
            else:
                output_path = f'output/{base_name}_barlines.png'
        else:
            output_path = args.output
            
        vis_img = detector.visualize_results(img, output_path)
        logger.debug(f"\nVisualization saved to: {output_path}")
        
        # Display result
        cv2.imshow("Measure Detection Result", cv2.resize(vis_img, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.debug(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())