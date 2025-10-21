#!/usr/bin/env python3
"""
使用 convert_to_tensorrt_python.py 批量转换 vit_l、vit_tiny、vit_h 的 encoder
"""

import os
import sys
import subprocess
import time

def convert_encoder(model_type, workspace_dir="/media/user/Disk2/gjt/sam-hq"):
    """
    使用 convert_to_tensorrt_python.py 转换单个 encoder
    
    Args:
        model_type: 模型类型 (vit_b, vit_l, vit_tiny, vit_h)
        workspace_dir: 工作目录
    """
    # 特殊处理：vit_h 的 encoder 在嵌套目录中
    if model_type == "vit_h":
        onnx_file = f"{workspace_dir}/sam_hq_{model_type}_encoder-onnx/sam_hq_{model_type}_encoder.onnx"
    else:
        onnx_file = f"{workspace_dir}/sam_hq_{model_type}_encoder.onnx"
    
    output_dir = f"{workspace_dir}/tensorrt_engines"
    
    # 检查 ONNX 文件是否存在
    if not os.path.exists(onnx_file):
        print(f"❌ ONNX 文件不存在: {onnx_file}")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(onnx_file) / (1024 * 1024)  # MB
    print(f"\n{'='*80}")
    print(f"开始转换: {model_type} encoder")
    print(f"ONNX 文件: {onnx_file}")
    print(f"文件大小: {file_size:.2f} MB")
    
    # 如果是 vit_h，检查外部数据文件
    if model_type == "vit_h" and file_size < 10:
        # 检查是否有外部数据文件
        from pathlib import Path
        external_files = list(Path(workspace_dir).glob("onnx__MatMul_*"))
        if external_files:
            print(f"✓ 检测到 {len(external_files)} 个外部数据文件")
        else:
            print(f"⚠️  警告: vit_h encoder 文件太小 ({file_size:.2f} MB)")
            print(f"   预期大小应该在 2GB 以上，当前可能缺少外部数据")
    
    # 特殊处理：vit_h 使用 bf16，其余使用 fp16
    precision = "bf16" if model_type == "vit_h" else "fp16"
    
    print(f"输出目录: {output_dir}")
    print(f"精度模式: {precision.upper()}")
    print(f"{'='*80}\n")
    
    # 构建命令 - 只转换 encoder
    cmd = [
        "python",
        f"{workspace_dir}/convert_to_tensorrt_python.py",
        "--encoder-onnx", onnx_file,
        "--output-dir", output_dir,
        "--precision", precision,
        "--workspace", "6144",  # 6GB workspace for large models
        "--skip-decoder"  # 跳过 decoder 转换
    ]
    
    print(f"执行命令:")
    print(f"  {' '.join(cmd)}\n")
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行转换
    try:
        result = subprocess.run(
            cmd,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            check=False  # 不立即抛出异常，我们自己检查
        )
        
        elapsed_time = time.time() - start_time
        
        # 打印输出
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("标准错误输出:")
            print(result.stderr)
        
        # 检查输出文件（vit_h 使用 bf16，其余使用 fp16）
        precision_suffix = "bf16" if model_type == "vit_h" else "fp16"
        engine_file = f"{output_dir}/sam_hq_{model_type}_encoder_{precision_suffix}.engine"
        
        # 注意：convert_to_tensorrt_python.py 使用固定名称 sam_hq_vit_b_encoder_*.engine
        # 我们需要重命名
        default_engine = f"{output_dir}/sam_hq_vit_b_encoder_{precision_suffix}.engine"
        
        if os.path.exists(default_engine):
            # 重命名为对应的模型名称
            if default_engine != engine_file:
                os.rename(default_engine, engine_file)
                print(f"✓ 重命名引擎文件: {os.path.basename(default_engine)} -> {os.path.basename(engine_file)}")
        
        if os.path.exists(engine_file):
            engine_size = os.path.getsize(engine_file) / (1024 * 1024)  # MB
            print(f"\n✅ {model_type} encoder 转换成功!")
            print(f"   用时: {elapsed_time:.2f} 秒")
            print(f"   精度: {precision_suffix.upper()}")
            print(f"   引擎文件: {engine_file}")
            print(f"   引擎大小: {engine_size:.2f} MB")
            return True
        else:
            print(f"\n❌ {model_type} encoder 转换失败")
            print(f"   用时: {elapsed_time:.2f} 秒")
            print(f"   返回码: {result.returncode}")
            return False
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {model_type} encoder 转换出错!")
        print(f"   用时: {elapsed_time:.2f} 秒")
        print(f"   错误: {e}")
        return False


def main():
    workspace_dir = "/media/user/Disk2/gjt/sam-hq"
    
    # 确保输出目录存在
    engine_dir = f"{workspace_dir}/tensorrt_engines"
    os.makedirs(engine_dir, exist_ok=True)
    
    # 要转换的模型列表（包含 vit_b）
    models = ["vit_tiny", "vit_b", "vit_l", "vit_h"]#
    
    print(f"{'='*80}")
    print(f"批量转换 Encoder 到 TensorRT")
    print(f"{'='*80}")
    print(f"使用脚本: convert_to_tensorrt_python.py")
    print(f"模型列表: {', '.join(models)}")
    print(f"输出目录: {engine_dir}")
    print(f"精度模式: vit_h=BF16, 其余=FP16")
    print(f"工作空间: 6GB")
    print(f"注意: vit_h 的 ONNX 路径为 sam_hq_vit_h_encoder-onnx/sam_hq_vit_h_encoder.onnx")
    print(f"{'='*80}\n")
    
    total_start_time = time.time()
    results = {}
    
    # 按顺序转换每个模型
    for model_type in models:
        success = convert_encoder(model_type, workspace_dir)
        results[model_type] = success
        
        if not success:
            print(f"\n⚠️  {model_type} encoder 转换失败")
            # 询问是否继续
            response = input(f"是否继续转换下一个模型? (y/n): ")
            if response.lower() != 'y':
                print("转换中止")
                break
        
        print(f"\n{'='*80}\n")
    
    total_elapsed = time.time() - total_start_time
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"转换完成总结")
    print(f"{'='*80}")
    print(f"总用时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)")
    print(f"\n结果:")
    
    success_count = 0
    for model_type, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        precision_suffix = "bf16" if model_type == "vit_h" else "fp16"
        print(f"  {model_type:12s}: {status}")
        if success:
            success_count += 1
            engine_file = f"{engine_dir}/sam_hq_{model_type}_encoder_{precision_suffix}.engine"
            if os.path.exists(engine_file):
                size = os.path.getsize(engine_file) / (1024 * 1024)
                print(f"                精度: {precision_suffix.upper()}")
                print(f"                引擎大小: {size:.2f} MB")
    
    print(f"\n成功: {success_count}/{len(results)}")
    print(f"{'='*80}\n")
    
    # 列出所有 encoder 引擎文件
    if success_count > 0:
        print("所有 encoder 引擎文件:")
        import glob
        engine_files = sorted(glob.glob(f"{engine_dir}/*_encoder_*.engine"))
        for ef in engine_files:
            size = os.path.getsize(ef) / (1024 * 1024)
            print(f"  {os.path.basename(ef):40s}  {size:8.2f} MB")
    
    return success_count == len(models)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
