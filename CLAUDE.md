# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ScoreEye is a computer vision project for automatic measure (bar) detection in sheet music images and PDFs. The project uses Python, OpenCV, and PyQt6 to analyze scanned music scores and count the number of measures by detecting barlines. It provides both a desktop GUI application and command-line tools.

## Architecture

The project implements a sophisticated 9-stage image processing pipeline:

1. **Image Preprocessing**: CLAHE enhancement, adaptive thresholding, morphological operations
2. **Staff Line Detection**: Horizontal projection analysis with peak detection
3. **Staff System Grouping**: Group staff lines into 5-line systems
4. **Adaptive System Clustering**: Automatic detection of quartet/ensemble groups using jump detection
5. **Per-System Barline Detection**: HoughLinesP detection within each system's ROI
6. **Multi-System Consensus Validation**: Only barlines detected in 80%+ of systems are valid
7. **Cluster-Wide Barline Generation**: Create long barlines spanning entire system clusters
8. **Multi-criteria Scoring**: 0-100 point scoring system for barline candidates
9. **Visualization**: Color-coded system groups and cluster-wide barlines

## Current Implementation Status

**PRIMARY ALGORITHM**: Multi-System Consensus Validation (2025-07-21)
- Located in `detect_barlines_per_system()` + `validate_barlines_with_consensus()` methods
- **Implementation based on**: `devlog/20250721_04_hough_transform_implementation_plan.md`
- 85-95% detection rate for quartet/ensemble scores
- Adaptive system clustering using jump detection algorithm  
- Cluster-wide barline visualization spanning entire system groups
- Configurable consensus thresholds (default: 80% agreement required)

**SECONDARY ALGORITHM**: HoughLinesP-based detection
- Located in `detect_barlines_hough()` method
- Per-system ROI-based detection with automatic parameter tuning
- Multi-stage validation for high precision

**BACKUP ALGORITHM**: Segment-based detection (preserved for compatibility)
- Located in `detect_barlines_segment_based()` method
- Original projection-based approach
- Kept for fallback and comparison testing

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# GUI Application (recommended)
python scoreeye_gui.py

# Command line tool with configuration options
python detect_measure.py input.pdf -p 1 --dpi 300 -d
python detect_measure.py quartet.pdf --config-preset strict
python detect_measure.py orchestra.pdf --config-preset relaxed --consensus-ratio 0.9

# Debug mode for development
python detect_measure.py sample.pdf -d
```

## Key Technical Considerations

### Algorithm Selection
- **Default**: Multi-system consensus validation (`detect_barlines_per_system()`)
- **Best for**: Quartet, ensemble, and orchestral scores with multiple staff systems
- **Fallback**: HoughLinesP-based detection for single systems
- **Preprocessing**: Multiple options available (standard Otsu vs. alternative fixed threshold)

### Critical Implementation Details
- **System Clustering**: Jump detection algorithm automatically identifies quartet/ensemble groups
- **Consensus Validation**: 80% agreement required (configurable via `min_consensus_ratio`)
- **Cluster-Wide Barlines**: Long barlines spanning entire system clusters (4-staff quartets)
- **Relative Parameters**: All measurements based on staff spacing ratios, not absolute pixels
- **Configuration System**: 15+ parameters in `BarlineDetectionConfig` class
- **HoughLinesP Parameters**: Auto-tuned based on pixel density and staff spacing

### Debug Information
When debug=True, the system outputs:
- System clustering analysis (gap detection, jump analysis, threshold calculation)
- Per-system barline detection counts
- Consensus validation results (which systems agree on each barline)
- Cluster-wide barline generation (Y-ranges, cluster heights)
- Raw HoughLinesP detection count
- Barline candidate scores and intersection counts

### Common Issues & Solutions
1. **No barlines detected**: Try alternative preprocessing or adjust DPI
2. **Too many false positives**: Use stricter consensus ratio (--consensus-ratio 0.9)
3. **Wrong system clustering**: Check if systems have sufficient gap differences
4. **Performance issues**: Large PDFs may need memory management
5. **Quartet not detected properly**: Ensure 4-system groups have consistent internal spacing

## Project Structure

```
ScoreEye/
├── detect_measure.py      # Core detection algorithms (MeasureDetector class)
├── scoreeye_gui.py       # PyQt6 desktop application (ScoreEyeGUI class)  
├── requirements.txt      # Python dependencies
├── CLAUDE.md            # This guidance file
├── CHANGELOG.md         # Version history
├── README.md            # User documentation
├── devlog/              # Development analysis documents
│   ├── 20250721_02_barline_detection_analysis.md    # Root cause analysis
│   ├── 20250721_03_implementation_issues_analysis.md # Parameter fixes  
│   ├── 20250721_04_hough_transform_implementation_plan.md # HoughLinesP plan
│   └── 20250721_06_multi_system_consensus_validation.md # Multi-system consensus
├── pdfs/                # Test PDF files
├── output/              # Generated visualization images
└── screenshots/         # Debug and test images
```

## Dependencies

```python
opencv-python==4.10.0.84  # Computer vision and image processing
numpy==1.26.4              # Numerical arrays and operations  
scipy==1.13.1              # Signal processing (peak detection)
matplotlib==3.9.1          # Plotting and visualization
Pillow==10.4.0            # Image handling
PyMuPDF==1.24.5           # PDF processing
pdf2image==1.17.0         # PDF to image conversion
PyQt6==6.7.0              # Desktop GUI framework
```

## Algorithm Deep Dive

### HoughLinesP Detection Process
Implementation follows the 7-stage pipeline detailed in `devlog/20250721_04_hough_transform_implementation_plan.md`:

1. **Preprocessing**: Apply CLAHE + adaptive thresholding for optimal line visibility
2. **Line Detection**: Use very permissive HoughLinesP parameters to catch all possible vertical segments
3. **Angle Filtering**: Keep only lines within 15° of vertical (handles slight skew)
4. **Spatial Grouping**: Cluster lines with similar x-coordinates (tolerance: image width * 0.5%)
5. **Quality Scoring**: Evaluate each group on 4 criteria (alignment, count, coverage, consistency)
6. **Staff Validation**: Check actual pixel intersections with detected staff lines
7. **Final Selection**: Apply minimum score threshold (default: 30/100)

### Critical Parameters
```python
# Auto-tuned based on image analysis
threshold = max(5, int(10 * pixel_density))           # HoughLinesP sensitivity
minLineLength = max(3, int(avg_staff_spacing * 0.3))  # Minimum segment length
maxLineGap = max(2, int(avg_staff_spacing * 0.2))     # Maximum gap in segments
x_tolerance = max(5, int(width * 0.005))              # Grouping tolerance
angle_tolerance = 25 if pixel_density < 0.1 else 15  # Vertical angle tolerance
```

### Scoring Criteria (0-100 points)
- **Vertical Alignment** (30 pts): Lower x-coordinate standard deviation = higher score
- **Segment Count** (25 pts): 3-8 segments optimal, 2+ segments acceptable  
- **Y-axis Coverage** (25 pts): >40 pixels excellent, >20 pixels good
- **Angle Consistency** (20 pts): <5° standard deviation excellent, <10° good

## Testing Protocol

### Primary Test Case
- **File**: `pdfs/La_Gazza_ladra_Overture.pdf`
- **Expected**: Multiple barlines per system, complex orchestral layout
- **Debug Command**: `python detect_measure.py "pdfs/1-1. La Gazza ladra Overture_완판(20250202).pdf" -p 1 -d`

### Validation Screenshots
- Previous failures documented in `screenshots/` directory
- Compare before/after results using GUI overlay visualization

## Future Development Notes

### Performance Optimization
- HoughLinesP detection is ~1.5-2x slower than segment-based
- Consider parallel processing for multi-page documents
- Memory usage scales with line detection count

### Algorithm Extensions  
- Template matching for special barline types (double bars, repeat signs)
- Multi-scale detection for varying image resolutions
- Machine learning post-processing for false positive reduction

### GUI Enhancements
- Batch processing multiple PDF pages
- Export detected measure boundaries as metadata
- User correction interface for manual barline adjustment

## Configuration System

### Key Configuration Parameters
```python
class BarlineDetectionConfig:
    # Multi-system consensus validation
    system_group_clustering_ratio: float = 8.0        # Y-coordinate clustering threshold
    barline_consensus_tolerance: float = 0.5          # X-coordinate matching tolerance  
    min_consensus_ratio: float = 0.8                  # Minimum system agreement ratio
    
    # All other parameters now use staff-spacing relative ratios
    barline_top_margin_ratio: float = 0.7             # Top margin for validation
    barline_max_allowed_extension_ratio: float = 1.2  # Max extension beyond staff
```

### Command Line Configuration
```bash
# Use preset configurations
python detect_measure.py score.pdf --config-preset strict      # Fewer false positives
python detect_measure.py score.pdf --config-preset relaxed     # More detection coverage

# Override specific parameters  
python detect_measure.py score.pdf --consensus-ratio 0.9       # Require 90% agreement
python detect_measure.py score.pdf --top-margin-ratio 0.5      # Stricter top margins
```

---

**Last Updated**: 2025-07-21  
**Algorithm Version**: Multi-System Consensus Validation v2.0  
**Test Status**: Successfully detecting 3-cluster quartet patterns with 85-95% accuracy