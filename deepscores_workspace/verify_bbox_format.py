#!/usr/bin/env python3
"""
Verify the correct interpretation of bbox format in DeepScores
"""

# Example stem bboxes from our analysis
stem_bboxes = [
    [1673.0, 159.0, 1675.0, 217.0],
    [1723.0, 175.0, 1724.0, 218.0], 
    [1636.0, 167.0, 1638.0, 218.0],
    [1655.0, 516.0, 1656.0, 590.0],
    [1610.0, 513.0, 1611.0, 606.0]
]

print("=== BBOX FORMAT VERIFICATION ===")
print()

for i, bbox in enumerate(stem_bboxes, 1):
    print(f"Stem Example {i}: {bbox}")
    
    # WRONG interpretation: [x, y, width, height]
    x, y, w_wrong, h_wrong = bbox
    ratio_wrong = w_wrong / h_wrong
    print(f"  WRONG [x,y,w,h]: x={x}, y={y}, width={w_wrong}, height={h_wrong}")
    print(f"  WRONG ratio: {ratio_wrong:.3f} ({'HORIZONTAL' if ratio_wrong > 1 else 'VERTICAL'})")
    
    # CORRECT interpretation: [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    ratio_correct = width / height if height > 0 else float('inf')
    print(f"  CORRECT [x1,y1,x2,y2]: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"  CORRECT dimensions: width={width}, height={height}")
    print(f"  CORRECT ratio: {ratio_correct:.3f} ({'HORIZONTAL' if ratio_correct > 1 else 'VERTICAL'})")
    print()

print("=== CONCLUSION ===")
print("The bbox format is clearly [x1, y1, x2, y2] because:")
print("1. Stems should be thin vertical lines (width << height)")
print("2. Correct interpretation gives width=1-3 pixels, height=40-90 pixels")
print("3. This matches expected musical stem dimensions")
print("4. Wrong interpretation gives impossible dimensions (width=1000+ pixels)")