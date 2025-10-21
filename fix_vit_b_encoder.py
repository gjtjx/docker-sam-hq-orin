#!/usr/bin/env python3
"""
修复 vit_b encoder ONNX 模型：
1. 输入名称从 'image' 改为 'images' 以保持一致性
2. 验证输出维度正确
"""

import torch
import onnx
import shutil
import os
from segment_anything import sam_model_registry

print("=" * 80)
print("修复 vit_b encoder ONNX 模型")
print("=" * 80)

# 加载模型
checkpoint = "sam_hq_vit_b.pth"
model_type = "vit_b"

print(f"\n加载模型: {checkpoint}")
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.eval()

# 备份原文件
encoder_onnx_path = f"sam_hq_{model_type}_encoder.onnx"
backup_path = f"{encoder_onnx_path}.backup_vit_b"
print(f"\n备份原文件: {backup_path}")
shutil.copy(encoder_onnx_path, backup_path)

# 导出 encoder（保持原始输出格式）
print(f"\n重新导出 encoder...")
dummy_input = torch.randn(1, 3, 1024, 1024)

# 创建包装器，输出 5维 interm_embeddings (与 vit_tiny/vit_l 一致)
class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x):
        features = self.encoder(x)
        # vit_b 返回的是 (image_embeddings, [interm_features])
        image_embeddings = features[0]
        interm_features_list = features[1]
        
        if len(interm_features_list) > 0:
            interm_features = interm_features_list[0]  # [B, H, W, C]
            # 关键修复：输出 5维格式 [4, B, H, W, C]，与其他模型一致
            # 先 unsqueeze 到 5维
            interm_embeddings = interm_features.unsqueeze(0)  # [1, B, H, W, C]
            # 重复到 4 层
            interm_embeddings = interm_embeddings.repeat(4, 1, 1, 1, 1)  # [4, B, H, W, C]
        else:
            interm_embeddings = interm_features_list
        
        return image_embeddings, interm_embeddings

wrapped_encoder = EncoderWrapper(sam_model.image_encoder)
wrapped_encoder.eval()

torch.onnx.export(
    wrapped_encoder,
    dummy_input,
    encoder_onnx_path,
    input_names=['images'],  # 统一改为 'images'
    output_names=['image_embeddings', 'interm_embeddings'],
    dynamic_axes={
        'images': {0: 'batch'},
        'image_embeddings': {0: 'batch'},
        'interm_embeddings': {0: 'batch'}  # vit_b 是 4 维，第一维是 batch
    },
    opset_version=17,
    do_constant_folding=True
)

# 验证导出的模型
print(f"\n验证 ONNX 模型...")
model = onnx.load(encoder_onnx_path)

print(f"\n输入:")
for inp in model.graph.input:
    dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {dims}")

print(f"\n输出:")
for out in model.graph.output:
    dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {dims}")

file_size_mb = os.path.getsize(encoder_onnx_path) / (1024 * 1024)
print(f"\n✅ vit_b encoder 已重新导出: {file_size_mb:.2f} MB")
print(f"   输入名称已统一为 'images'")

print("\n" + "=" * 80)
print("✅ 修复完成！")
print("=" * 80)
