#!/usr/bin/env python3
"""
修复 encoder ONNX 导出，确保 interm_embeddings 输出格式正确
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'segment_anything'))

from segment_anything import sam_model_registry


class EncoderWrapper(torch.nn.Module):
    """Encoder 包装器，确保输出格式正确"""
    
    def __init__(self, encoder, model_type):
        super().__init__()
        self.encoder = encoder
        self.model_type = model_type
    
    def forward(self, x):
        # 调用原始 encoder
        image_embeddings, interm_embeddings = self.encoder(x)
        
        # 确保 interm_embeddings 是 5 维: [num_layers, batch, H, W, C]
        # 不同模型可能输出不同格式
        if len(interm_embeddings) > 0:
            interm_emb = interm_embeddings[0]  # 取第一个 intermediate embedding
            
            # interm_emb 应该是 [B, H, W, C] 格式
            # 添加 num_layers 维度并复制: [B, H, W, C] -> [4, B, H, W, C]
            if interm_emb.dim() == 4:
                # 直接在第 0 维添加
                interm_emb = interm_emb.unsqueeze(0)  # [1, B, H, W, C]
                interm_emb = interm_emb.repeat(4, 1, 1, 1, 1)  # [4, B, H, W, C]
            else:
                # 如果不是 4 维，说明格式异常
                raise ValueError(f"Unexpected interm_embeddings shape: {interm_emb.shape}")
        
        return image_embeddings, interm_emb


def export_encoder_fixed(model_type, checkpoint_path, output_path):
    """
    重新导出 encoder ONNX，修复 interm_embeddings 格式
    
    Args:
        model_type: 模型类型 (vit_tiny, vit_b, vit_l, vit_h)
        checkpoint_path: 模型权重路径
        output_path: 输出 ONNX 路径
    """
    print(f"\n{'='*80}")
    print(f"修复导出: {model_type} encoder")
    print(f"{'='*80}")
    print(f"权重文件: {checkpoint_path}")
    print(f"输出文件: {output_path}")
    
    # 加载模型
    print("\n加载模型...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).cuda()
    sam.eval()
    
    # 创建包装器
    encoder_wrapper = EncoderWrapper(sam.image_encoder, model_type).cuda()
    encoder_wrapper.eval()
    
    # 创建 dummy 输入
    dummy_input = torch.randn(1, 3, 1024, 1024).cuda()
    
    # 测试输出形状
    print("\n测试输出形状...")
    with torch.no_grad():
        test_embeddings, test_interm = encoder_wrapper(dummy_input)
        print(f"  image_embeddings: {test_embeddings.shape}")
        print(f"  interm_embeddings: {test_interm.shape}")
    
    # 导出 ONNX
    print("\n导出 ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['image_embeddings', 'interm_embeddings'],
            dynamic_axes={
                'images': {0: 'batch'},
                'image_embeddings': {0: 'batch'},
                'interm_embeddings': {1: 'batch'},
            }
        )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ 导出成功!")
    print(f"   文件: {output_path}")
    print(f"   大小: {file_size:.2f} MB")
    
    # 验证 ONNX
    print("\n验证 ONNX...")
    import onnx
    model = onnx.load(output_path)
    
    print("  输出:")
    for out in model.graph.output:
        dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {dims}")
    
    return True


def main():
    workspace_dir = "/media/user/Disk2/gjt/sam-hq"
    
    # 要修复的模型（跳过 vit_b，它已经是正确的）
    models_to_fix = [
        ('vit_tiny', 'sam_hq_vit_tiny.pth'),
        ('vit_l', 'sam_hq_vit_l.pth'),
    ]
    
    print(f"{'='*80}")
    print(f"修复 Encoder ONNX 导出")
    print(f"{'='*80}")
    print(f"目标: 确保 interm_embeddings 输出为 [4, batch, 64, 64, C] 格式")
    print(f"模型: {', '.join([m[0] for m in models_to_fix])}")
    print(f"{'='*80}\n")
    
    results = {}
    
    for model_type, checkpoint_name in models_to_fix:
        checkpoint_path = f"{workspace_dir}/{checkpoint_name}"
        output_path = f"{workspace_dir}/sam_hq_{model_type}_encoder.onnx"
        
        # 备份原文件
        if os.path.exists(output_path):
            backup_path = f"{output_path}.backup"
            print(f"备份原文件: {backup_path}")
            os.rename(output_path, backup_path)
        
        try:
            success = export_encoder_fixed(model_type, checkpoint_path, output_path)
            results[model_type] = success
        except Exception as e:
            print(f"\n❌ {model_type} 导出失败: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = False
            
            # 恢复备份
            backup_path = f"{output_path}.backup"
            if os.path.exists(backup_path):
                print(f"恢复备份文件...")
                os.rename(backup_path, output_path)
        
        print(f"\n{'='*80}\n")
    
    # 总结
    print(f"\n{'='*80}")
    print(f"修复总结")
    print(f"{'='*80}")
    
    success_count = sum(1 for s in results.values() if s)
    for model_type, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {model_type:12s}: {status}")
    
    print(f"\n成功: {success_count}/{len(results)}")
    print(f"{'='*80}\n")
    
    if success_count > 0:
        print("💡 提示: 现在可以重新运行评估脚本")
        print("   python evaluate_all_models.py")
    
    return success_count == len(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
