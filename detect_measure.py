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
from pdf2image import convert_from_path
import tempfile


class MeasureDetector:
    def __init__(self, debug=False):
        """Initialize the MeasureDetector.
        
        Args:
            debug (bool): If True, displays intermediate processing steps
        """
        self.debug = debug
        self.staff_lines = []
        self.barlines = []
        self.binary_img = None
        
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
            
        # Convert PDF page to image
        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num+1, last_page=page_num+1)
        
        if not images:
            raise ValueError(f"Failed to convert PDF page {page_num+1}")
            
        # Convert PIL image to numpy array and then to grayscale
        img_array = np.array(images[0])
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = img_array
            
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
            elif peak - current_staff[-1] < avg_spacing * 2.5:  # Lines within same staff
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
        kernel_height = max(8, int(avg_spacing * 0.7))
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
        """스태프 라인과의 교차를 확인하여 바라인 검증
        
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
        
        # 교차점 분석
        validation_result = {
            'intersection_count': len(intersections),
            'staff_coverage_ratio': len(intersections) / len(staff_lines) if staff_lines else 0,
            'intersections': intersections,
            'is_valid_barline': len(intersections) >= 3  # 최소 3개 스태프와 교차
        }
        
        return validation_result
    
    def check_intersection_at_staff(self, x, staff_y):
        """특정 X좌표에서 스태프 라인과의 교차점 확인
        
        Args:
            x (int): X-coordinate to check
            staff_y (int): Staff line y-coordinate
            
        Returns:
            bool: True if intersection exists
        """
        if not hasattr(self, 'binary_img') or self.binary_img is None:
            return False
            
        # 스태프 라인 주변 ±3픽셀 영역에서 수직 픽셀 존재 확인
        roi_start = max(0, staff_y - 3)
        roi_end = min(self.binary_img.shape[0], staff_y + 4)
        
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
            print(f"Auto-tuned parameters: {params}")
        
        return params
    
    def detect_barlines_hough(self, binary_img):
        """HoughLinesP 기반 바라인 검출 메인 함수
        
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
            print(f"Raw HoughLinesP detected: {len(all_lines)} lines")
        
        # 4. 수직성 필터링
        vertical_lines = self.filter_vertical_lines(all_lines, params['angle_tolerance'])
        if self.debug:
            print(f"Vertical lines filtered: {len(vertical_lines)}")
        
        if not vertical_lines:
            return []
        
        # 5. X좌표 기반 그룹핑
        line_groups = self.group_lines_by_x_coordinate(vertical_lines, params['x_tolerance'])
        if self.debug:
            print(f"Line groups formed: {len(line_groups)}")
        
        if not line_groups:
            return []
        
        # 6. 각 그룹 분석
        analyzed_groups = [self.analyze_line_group(group) for group in line_groups]
        
        # 7. 스태프 기반 검증 및 최종 선별
        min_score = 30  # 관대한 점수 기준
        final_barlines = self.select_final_barlines(analyzed_groups, self.staff_lines, min_score)
        
        if self.debug:
            print(f"Final barlines selected: {len(final_barlines)}")
            for i, barline in enumerate(final_barlines):
                print(f"  Barline {i+1}: x={barline['x']}, score={barline['score']:.1f}, "
                      f"intersections={barline['staff_intersections']}")
        
        return [b['x'] for b in final_barlines]
    
    def detect_barlines(self, binary_img):
        """바라인 검출 - HoughLinesP 기반 구현
        
        Args:
            binary_img (numpy.ndarray): Binary image
            
        Returns:
            list: X-coordinates of detected barlines
        """
        return self.detect_barlines_hough(binary_img)

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
                if self.staff_lines[i] - current_group[-1] <= avg_spacing * 2.5:
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
        # Number of measures = number of barlines - 1
        # (assuming first and last barlines mark beginning and end)
        if len(self.barlines) >= 2:
            return len(self.barlines) - 1
        else:
            return 0
    
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
            
        # Draw barlines in red
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
        print(f"Detected {len(staff_lines)} staff lines")
        
        # Detect barlines
        barline_candidates = self.detect_barlines(binary)
        print(f"Found {len(barline_candidates)} barline candidates")
        
        # Filter barlines
        barlines = self.filter_barlines(barline_candidates)
        print(f"Filtered to {len(barlines)} valid barlines")
        
        # Count measures
        measure_count = self.count_measures()
        
        results = {
            'measure_count': measure_count,
            'barlines': barlines,
            'barline_candidates': barline_candidates,
            'staff_lines': [line['y'] if isinstance(line, dict) else line for line in staff_lines],
            'staff_lines_with_ranges': staff_lines if staff_lines and isinstance(staff_lines[0], dict) else []
        }
        
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
        
        # Preprocess - 대안 방법 선택 가능
        if use_alternative_preprocessing:
            binary = self.preprocess_image_alternative(img)
        else:
            binary = self.preprocess_image(img)
        
        # Detect staff lines
        staff_lines = self.detect_staff_lines(binary)
        print(f"Detected {len(staff_lines)} staff lines")
        
        # Detect barlines
        barline_candidates = self.detect_barlines(binary)
        print(f"Found {len(barline_candidates)} barline candidates")
        
        # Filter barlines
        barlines = self.filter_barlines(barline_candidates)
        print(f"Filtered to {len(barlines)} valid barlines")
        
        # Count measures
        measure_count = self.count_measures()
        
        results = {
            'measure_count': measure_count,
            'barlines': barlines,
            'barline_candidates': barline_candidates,
            'staff_lines': [line['y'] if isinstance(line, dict) else line for line in staff_lines],
            'staff_lines_with_ranges': staff_lines if staff_lines and isinstance(staff_lines[0], dict) else [],
            'original_image': img
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Detect measures in sheet music images or PDFs')
    parser.add_argument('input', help='Path to the input image or PDF file')
    parser.add_argument('-o', '--output', help='Path to save visualization')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug visualizations')
    parser.add_argument('-p', '--page', type=int, default=1, help='Page number for PDF (1-indexed, default: 1)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF conversion (default: 300)')
    
    args = parser.parse_args()
    
    # Create detector
    detector = MeasureDetector(debug=args.debug)
    
    try:
        # Check if input is PDF or image
        file_ext = os.path.splitext(args.input)[1].lower()
        
        if file_ext == '.pdf':
            # Process PDF
            print(f"Processing PDF: {args.input}, Page: {args.page}")
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
        
        print(f"\nDetection Results:")
        print(f"Number of measures: {results['measure_count']}")
        print(f"Barline positions: {results['barlines']}")
        
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
        print(f"\nVisualization saved to: {output_path}")
        
        # Display result
        cv2.imshow("Measure Detection Result", cv2.resize(vis_img, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())