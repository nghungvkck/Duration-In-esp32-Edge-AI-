"""
export_onnx.py: Export pruned MelStudent model (float32) sang ONNX.
Input:  best_pruned_student.pth  (float32, đã prune + fine-tune)
Output: model.onnx

Chạy: python export_onnx.py
"""
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
CKPT_PATH = "../notebook/model_94_no_smooth/best_pruned_student.pth"
ONNX_PATH = "model.onnx"
INPUT_SHAPE = (1, 1, 128, 16)   # batch, channel, mel_bins, time_frames
OPSET_VERSION = 18


# ============================================================
# MODEL CLASS (khớp với lúc train)
# ============================================================
class MelStudent(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# BUILD PRUNED MODEL (channels: 1 → 19 → 38 → 76 → 153)
# ============================================================
def build_pruned_model():
    """Khởi tạo MelStudent với kiến trúc đã prune"""
    model = MelStudent()

    model.features[0]  = nn.Conv2d(1, 19, 3, padding=1)
    model.features[1]  = nn.BatchNorm2d(19)

    model.features[4]  = nn.Conv2d(19, 38, 3, padding=1)
    model.features[5]  = nn.BatchNorm2d(38)

    model.features[8]  = nn.Conv2d(38, 76, 3, padding=1)
    model.features[9]  = nn.BatchNorm2d(76)

    model.features[12] = nn.Conv2d(76, 153, 3, padding=1)
    model.features[13] = nn.BatchNorm2d(153)

    model.classifier[0] = nn.Linear(153, 128)

    return model


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("EXPORT ONNX")
    print("=" * 60)

    # --- Build + load ---
    model = build_pruned_model()
    state = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded: {CKPT_PATH}")

    # --- Info ---
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {n_params:,} (~{n_params * 4 / 1024:.1f} KB float32)")

    # --- Export ONNX ---
    dummy_input = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=['input'],
        output_names=['output'],
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamic_axes=None,          # fix shape cho ESP32
        verbose=False,
    )
    print(f"✅ Exported: {ONNX_PATH}")

    # --- Verify ---
    verify_onnx(ONNX_PATH)


# ============================================================
# VERIFY (không cần onnxruntime — tránh lỗi WSL)
# ============================================================
def verify_onnx(onnx_path):
    """Verify ONNX structure — chỉ dùng onnx, không dùng onnxruntime"""
    import onnx

    print("\n" + "=" * 60)
    print("VERIFY ONNX")
    print("=" * 60)

    # 1. Load + check structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX structure OK")

    # 2. Input / Output info
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"   Input:  {inp.name:10s} shape={shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"   Output: {out.name:10s} shape={shape}")

    # 3. Metadata
    print(f"   Opset:  {onnx_model.opset_import[0].version}")
    print(f"   IR:     {onnx_model.ir_version}")

    size_kb = Path(onnx_path).stat().st_size / 1024
    print(f"   File size: {size_kb:.1f} KB")

    # 4. Nodes + op types
    n_nodes = len(onnx_model.graph.node)
    op_types = sorted(set(node.op_type for node in onnx_model.graph.node))
    print(f"   Nodes:  {n_nodes}")
    print(f"   Ops:    {op_types}")

    print("\n✅ ONNX ready — sang bước quantize ESP-PPQ")


if __name__ == "__main__":
    main()