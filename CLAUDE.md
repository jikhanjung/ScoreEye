# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ScoreEye is a computer vision project for automatic measure (bar) detection in sheet music images. The project uses Python and OpenCV to analyze scanned music scores and count the number of measures by detecting barlines.

## Architecture

The project implements a multi-stage image processing pipeline:

1. **Image Preprocessing**: Binary thresholding and noise removal
2. **Staff Line Detection**: Horizontal projection analysis to find musical staff lines
3. **Barline Detection**: Morphological operations or Hough Transform to extract vertical lines
4. **Barline Filtering**: Validation that vertical lines completely cross all staff lines
5. **Measure Counting**: Count validated barlines to determine number of measures

## Development Commands

Since the project is in initial development, standard Python commands will be used:

```bash
# Create virtual environment (once)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies (after requirements.txt is created)
pip install -r requirements.txt

# Run the main detection script (once created)
python detect_measure.py
```

## Key Technical Considerations

When implementing features:
- Use OpenCV (cv2) for all image processing operations
- Handle both single-staff and multi-staff (e.g., piano) scores
- Account for special barline types (double bars, repeat signs)
- Avoid confusion with vertical elements like note stems or dots
- Merge closely spaced barlines (within ~5 pixels)

## Project Structure

Expected file organization:
```
ScoreEye/
├── images/           # Input score images
├── output/          # Processed results with visualizations
├── detect_measure.py # Main detection script
└── requirements.txt # Python dependencies
```

## Dependencies

Core libraries needed:
- OpenCV (cv2) - Image processing
- NumPy - Array operations
- scipy.signal - Peak detection for staff lines