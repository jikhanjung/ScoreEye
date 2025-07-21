# ScoreEye Configuration Guide

This document explains how to configure ScoreEye's barline detection parameters for optimal results across different sheet music types.

## Overview

ScoreEye uses relative measurements based on staff line spacing rather than absolute pixel values. This makes the detection adaptive to different scan resolutions and staff sizes. All parameters are expressed as ratios relative to the average staff line spacing.

## Configuration Class

The `BarlineDetectionConfig` class contains all configurable parameters:

```python
from detect_measure import BarlineDetectionConfig, MeasureDetector

# Use default configuration
config = BarlineDetectionConfig()
detector = MeasureDetector(config=config)

# Create custom configuration
custom_config = BarlineDetectionConfig(
    barline_top_margin_ratio=0.9,
    barline_max_allowed_extension_ratio=1.5
)
detector = MeasureDetector(config=custom_config)
```

## Configuration Parameters

### Staff Line Detection
- **`staff_same_system_spacing_ratio`** (default: 2.5)
  - Determines which lines belong to the same staff system
  - Smaller values = stricter grouping, larger values = more lenient
  
- **`staff_grouping_kernel_ratio`** (default: 0.7)
  - Controls morphological operations for staff line grouping
  - Affects how staff lines are cleaned and connected

### Barline Validation
- **`barline_top_margin_ratio`** (default: 0.7)
  - Top margin allowance for barline validation
  - How far above the top staff line a barline can extend
  
- **`barline_bottom_margin_ratio`** (default: 0.7)  
  - Bottom margin allowance for barline validation
  - How far below the bottom staff line a barline can extend
  
- **`barline_max_allowed_extension_ratio`** (default: 1.2)
  - Maximum total extension beyond staff boundaries
  - Controls overall barline length limits
  
- **`barline_roi_margin_ratio`** (default: 0.25)
  - Region of interest margin around staff intersections
  - Affects precision of intersection detection

### HoughLinesP Algorithm Parameters
- **`hough_min_line_length_ratio`** (default: 0.5)
  - Minimum length for detected line segments
  - Smaller values detect shorter segments, larger values require longer lines
  
- **`hough_max_line_gap_ratio`** (default: 0.3)
  - Maximum gap allowed within a single line
  - How much interruption is tolerated in a barline
  
- **`hough_max_barline_length_ratio`** (default: 2.0)
  - Maximum allowed barline length
  - Prevents detection of very long vertical elements as barlines

### Staff System Detection
- **`staff_system_normal_spacing_ratio`** (default: 2.0)
  - Normal spacing threshold within a system
  - Used to group staff lines into systems
  
- **`staff_system_grouping_ratio`** (default: 2.5)
  - Threshold for grouping staff lines into systems
  - Larger values create bigger staff system groups

## Preset Configurations

### Default Configuration
Balanced settings suitable for most sheet music:
```python
config = BarlineDetectionConfig()  # Uses default values
```

### Strict Configuration
Reduces false positives, may miss some valid barlines:
```python
config = BarlineDetectionConfig.create_strict_config()
```
- Tighter margins (0.5 instead of 0.7)
- Stricter extension limits (1.0 instead of 1.2)
- Longer minimum line requirements

### Relaxed Configuration  
Detects more barlines, may include false positives:
```python
config = BarlineDetectionConfig.create_relaxed_config()
```
- Larger margins (1.0 instead of 0.7)
- More generous extension limits (1.5 instead of 1.2)
- Shorter minimum line requirements

## Command Line Usage

You can configure detection parameters directly from the command line:

### Using Presets
```bash
# Strict detection (fewer false positives)
python detect_measure.py score.pdf --config-preset strict

# Relaxed detection (more detection, possible false positives)  
python detect_measure.py score.pdf --config-preset relaxed
```

### Custom Parameters
```bash
# Custom margin ratios
python detect_measure.py score.pdf \
    --top-margin-ratio 0.9 \
    --bottom-margin-ratio 0.8 \
    --max-extension-ratio 1.4
```

### Debug Mode with Configuration
```bash
# See configuration details in debug output
python detect_measure.py score.pdf --debug --config-preset strict
```

## Tuning Guidelines

### If you're getting too many false positives:
1. Use strict preset: `--config-preset strict`
2. Reduce margin ratios: `--top-margin-ratio 0.5 --bottom-margin-ratio 0.5`
3. Reduce extension ratio: `--max-extension-ratio 1.0`

### If you're missing valid barlines:
1. Use relaxed preset: `--config-preset relaxed`
2. Increase margin ratios: `--top-margin-ratio 1.0 --bottom-margin-ratio 1.0`
3. Increase extension ratio: `--max-extension-ratio 1.5`

### For handwritten or low-quality scores:
1. Use relaxed configuration
2. Increase `hough_max_line_gap_ratio` to 0.4 or 0.5
3. Decrease `hough_min_line_length_ratio` to 0.3

### For high-quality printed scores:
1. Use strict configuration
2. Increase `hough_min_line_length_ratio` to 0.6 or 0.7
3. Decrease margin ratios to 0.5

## Programmatic Configuration

```python
from detect_measure import BarlineDetectionConfig, MeasureDetector

# Create custom configuration
config = BarlineDetectionConfig(
    # Adjust for handwritten music
    barline_top_margin_ratio=1.0,
    barline_bottom_margin_ratio=1.0,
    hough_max_line_gap_ratio=0.4,
    hough_min_line_length_ratio=0.3
)

# Apply configuration  
detector = MeasureDetector(debug=True, config=config)

# Process with custom settings
results = detector.detect_measures_from_pdf("handwritten_score.pdf")
```

## Advanced Usage

### Dynamic Configuration Adjustment
```python
config = BarlineDetectionConfig()

# Scale all margins by factor
config.adjust_margins(1.5)  # Make detection 50% more lenient

# Get configuration summary
print(config.get_description())
```

### Configuration Validation
The configuration class automatically validates parameters:
```python
# This will raise ValueError
try:
    bad_config = BarlineDetectionConfig(barline_max_allowed_extension_ratio=0.1)
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

1. **Start with presets**: Try `default`, `strict`, and `relaxed` first
2. **Use debug mode**: Always test with `--debug` to see detection stages
3. **Iterative tuning**: Adjust one parameter at a time
4. **Document your settings**: Save configurations that work well for specific score types
5. **Test systematically**: Use the same test images when tuning parameters

## Configuration Storage

For repeated use, save successful configurations:
```python
# Save configuration for classical scores
classical_config = BarlineDetectionConfig(
    barline_top_margin_ratio=0.8,
    barline_bottom_margin_ratio=0.8,
    hough_min_line_length_ratio=0.6
)

# Save configuration for modern scores
modern_config = BarlineDetectionConfig.create_relaxed_config()
modern_config.hough_max_line_gap_ratio = 0.4
```

---

This configuration system allows ScoreEye to adapt to various sheet music styles, scanning qualities, and detection requirements while maintaining the relative measurement approach that makes the system robust across different resolutions.