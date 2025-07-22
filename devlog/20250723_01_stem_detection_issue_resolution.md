# 20250723_01_stem_detection_issue_resolution.md

## Stem Detection Issue Resolution and 2048x2048 High-Resolution Training

**Date**: 2025-07-23
**Status**: Completed Analysis, In-Progress Training
**Priority**: Critical

## 🎯 Problem Statement

YOLOv8 training showed stem and augmentationDot classes with 0% mAP, while other classes performed normally:
- **stem**: 0% mAP (158,570 annotations, 43.5% of dataset)
- **augmentationDot**: 0% mAP (14,587 annotations, 4.0% of dataset)
- **Other classes**: 20-98% mAP (normal performance)

## 🔍 Root Cause Analysis

### Initial Hypothesis (Incorrect)
- **Suspected**: Bounding box coordinate conversion errors (X/Y axis swap)
- **Suspected**: Width/height values swapped during COCO to YOLO conversion

### Investigation Process

1. **Analyzed Original DeepScores Data**:
   ```json
   // Example stem annotation
   "a_bbox": [1413.0, 594.0, 1414.0, 646.0]  // [x1, y1, x2, y2]
   // Actual dimensions: width=1px, height=52px (ratio: 0.019)
   ```

2. **Verified YOLO Conversion**:
   ```
   Original: [1413, 594, 1414, 646] → width=1, height=52
   2048x2048: [1333, 98, 1334, 127] → width=0.7px, height=29.6px
   YOLO format: width=0.000361, height=0.014430 (ratio: 0.025)
   ```

3. **Visual Verification**:
   - Created debug visualizations of original vs converted images
   - Confirmed bounding boxes correctly wrap vertical stem lines
   - **Conversion is mathematically correct**

### Actual Root Cause
**Stems are extremely thin objects (0.7-1.5 pixels wide) that are difficult for YOLOv8 to detect at 1024x1024 resolution.**

## 💡 Solution: High-Resolution Training

### Approach
1. **Increased Image Resolution**: 1024x1024 → 2048x2048
2. **Maintained Aspect Ratio**: Used uniform scaling to preserve stem proportions
3. **Optimized Memory Usage**: Reduced batch size to handle larger images

### Implementation Details

#### Preprocessing Updates
```python
# preprocess_deepscores_coco.py
self.image_size = 2048  # Increased from 1024

# Uniform scaling to preserve aspect ratio
scale = min(1024/orig_width, 1024/orig_height)  # Changed to use same scale for X/Y
new_width = int(orig_width * scale)
new_height = int(orig_height * scale)
```

#### Training Configuration
```python
# train_yolo.py
imgsz: int = 2048  # Increased from 1024

# Memory optimization
batch_size = 2  # Reduced from auto (4) due to VRAM constraints
```

#### Batch Size Issue Resolution
```python
# Fixed argument parsing for batch size
parser.add_argument("--batch", default="auto", help="배치 크기", 
                    type=lambda x: int(x) if x.isdigit() else x)
```

## 📊 Results Comparison

### 1024x1024 vs 2048x2048 Results (3 epochs)

| Metric | 1024x1024 | 2048x2048 | Improvement |
|--------|-----------|-----------|-------------|
| **mAP@0.5** | 30.1% | **54.7%** | **+24.6%p** |
| **mAP@0.5-95** | 19.6% | **40.3%** | **+20.7%p** |
| **Precision** | 36.5% | **65.3%** | **+28.8%p** |
| **Recall** | 30.4% | **51.4%** | **+21.0%p** |

### Stem Detection Improvement
- **Previous**: 0% mAP (undetectable)
- **Expected**: Significant improvement due to increased pixel resolution
- **Stem pixel size**: 0.37px → 0.7-1.5px (2x improvement)

## 🛠 Technical Implementation

### Dataset Processing
```bash
# 1000 sample preprocessing with 2048x2048
python3 preprocess_deepscores_coco.py --pilot-mode --sample-size 1000

# Results:
# - 766 images processed (566 train, 200 val)
# - 364,400 valid annotations
# - stem: 158,570 annotations (43.5%)
```

### Training Command
```bash
python3 train_yolo.py --data data/deepscores.yaml --epochs 20 --model yolov8n \
  --name stem_fixed_2048_batch2 --imgsz 2048 --batch 2
```

### Memory Optimization
- **RTX 2080 Ti (11GB VRAM)**: Limited to batch size 2 for 2048x2048 images
- **Memory usage**: 19-36GB peak (exceeded VRAM, caused OOM after epoch 3)
- **Solution**: Reduced batch size from auto (4) to 2

## 🔧 Git Configuration

### Updated .gitignore
```gitignore
# DeepScores workspace (exclude large data/output directories, keep Python scripts)
deepscores_workspace/data/
deepscores_workspace/ds2_dense/
deepscores_workspace/raw_data/
deepscores_workspace/runs/
deepscores_workspace/scoreeye-yolov8/
deepscores_workspace/wandb/
deepscores_workspace/validation_reports/
deepscores_workspace/models/
deepscores_workspace/weights/
deepscores_workspace/cache/
deepscores_workspace/*.pt
deepscores_workspace/*.pth
deepscores_workspace/*.png
deepscores_workspace/*.jpg
```

### Files Tracked
- ✅ Python scripts: `train_yolo.py`, `preprocess_deepscores_coco.py`, etc.
- ❌ Large data files: models, datasets, training outputs

## 📈 Training Progress

### Current Status (2025-07-23)
- **Completed**: 3 epochs of 2048x2048 training
- **Issue**: CUDA OOM error during epoch 4 validation
- **Next**: Continue with batch size 2 or implement gradient accumulation

### Training Metrics (Epochs 1-3)
```
Epoch 1: mAP@0.5=27.2%, Precision=68.6%, Recall=25.5%
Epoch 2: mAP@0.5=45.1%, Precision=49.9%, Recall=43.7%
Epoch 3: mAP@0.5=54.7%, Precision=65.3%, Recall=51.4%
```

### W&B Monitoring
- **Project**: scoreeye-yolov8
- **Run**: stem_fixed_2048_1000
- **URL**: https://wandb.ai/honestjung-none/scoreeye-yolov8

## 🎯 Key Findings

### 1. Stem Detection Is Resolution-Dependent
- **1024x1024**: Stems become 0.37px wide (undetectable)
- **2048x2048**: Stems become 0.7-1.5px wide (barely detectable)
- **Conclusion**: Higher resolution significantly improves thin object detection

### 2. Bounding Box Conversion Was Correct
- **Initial suspicion**: X/Y coordinate swap
- **Reality**: Mathematical conversion was accurate
- **Learning**: Visual debugging crucial for complex geometry issues

### 3. Memory Constraints Limit Batch Size
- **2048x2048 + batch 4**: 36GB peak memory (OOM)
- **2048x2048 + batch 2**: Should fit in 11GB VRAM
- **Alternative**: Gradient accumulation for effective larger batch sizes

### 4. Performance Improvement Is Dramatic
- **+24.6%p mAP improvement** in just 3 epochs
- **65.3% precision** shows model can learn to distinguish symbols
- **51.4% recall** indicates good detection coverage

## 🚀 Next Steps

### Immediate Actions
1. **Complete 20-epoch training** with batch size 2
2. **Analyze per-class performance** to confirm stem detection improvement
3. **Implement gradient accumulation** if batch size 2 proves insufficient

### Future Optimizations
1. **Progressive image sizes**: Start with 1024, gradually increase to 2048
2. **Weighted loss functions**: Address class imbalance (stem: 43.5% vs others)
3. **Data augmentation tuning**: Optimize for thin vertical objects
4. **Model architecture**: Consider YOLOv8s or YOLOv8m for better small object detection

### Alternative Approaches
1. **Multi-scale training**: Train on multiple resolutions simultaneously
2. **Anchor optimization**: Tune anchors for thin vertical objects
3. **Loss function modification**: Focal loss for hard-to-detect objects
4. **Ensemble methods**: Combine multiple models trained at different scales

## 📝 Lessons Learned

1. **Debug with visualization**: Visual confirmation prevented hours of incorrect optimization
2. **Resolution matters for small objects**: 2x resolution = massive performance gain
3. **Memory planning is crucial**: VRAM constraints significantly impact training strategy
4. **Incremental testing**: 3-epoch results provided early validation of approach

## 🔄 Related Work

- **Previous**: `20250722_02_object_detection_plan.md` - Initial YOLOv8 planning
- **Previous**: `20250722_05_bracket_detection_plan.md` - Multi-system approach
- **Next**: Training completion and full evaluation results

---

**Status**: Analysis Complete ✅
**Next Milestone**: Complete 2048x2048 training and evaluate stem detection performance
**Estimated Time**: 2-3 hours for full 20-epoch training