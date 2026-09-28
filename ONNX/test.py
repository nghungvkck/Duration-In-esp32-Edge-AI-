import torch

# Check file float32
ckpt_float = torch.load("../notebook/model_94_no_smooth/pruned_student.pth", map_location='cpu')
print("=== pruned_student.pth ===")
print(type(ckpt_float))
if isinstance(ckpt_float, dict):
    keys = list(ckpt_float.keys())
    print(f"Total keys: {len(keys)}")
    print(f"First 5: {keys[:5]}")
    print(f"Last 5: {keys[-5:]}")
    # Check có phải quantized không
    is_quantized = any('scale' in k or 'zero_point' in k for k in keys)
    print(f"Quantized? {is_quantized}")