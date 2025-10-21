#!/usr/bin/env python3
"""
使用Python TensorRT API转换ONNX模型为TensorRT引擎
支持encoder和decoder的转换
"""

import tensorrt as trt
import numpy as np
import os
import argparse


class TensorRTConverter:
    """TensorRT模型转换器"""
    
    def __init__(self, precision='fp16', max_workspace_size=4096):
        """
        初始化转换器
        
        Args:
            precision: 精度模式 ('fp32', 'fp16', 'int8')
            max_workspace_size: 最大工作空间大小 (MB)
        """
        self.precision = precision.lower()
        self.max_workspace_size = max_workspace_size * (1024 ** 2)  # 转换为字节
        
        # 创建logger
        self.logger = trt.Logger(trt.Logger.INFO)
        
        print(f"TensorRT版本: {trt.__version__}")
        print(f"精度模式: {self.precision}")
        print(f"工作空间大小: {max_workspace_size} MB")
    
    def build_engine_from_onnx(self, onnx_path, engine_path, 
                               dynamic_shapes=None, 
                               min_shapes=None, 
                               opt_shapes=None, 
                               max_shapes=None):
        """
        从ONNX模型构建TensorRT引擎
        
        Args:
            onnx_path: ONNX模型路径
            engine_path: 输出引擎路径
            dynamic_shapes: 动态形状的输入名称列表
            min_shapes: 最小形状字典 {input_name: shape}
            opt_shapes: 最优形状字典 {input_name: shape}
            max_shapes: 最大形状字典 {input_name: shape}
        
        Returns:
            bool: 是否成功
        """
        print("\n" + "="*80)
        print(f"🔧 转换ONNX模型到TensorRT引擎")
        print("="*80)
        print(f"  输入ONNX: {onnx_path}")
        print(f"  输出引擎: {engine_path}")
        
        if not os.path.exists(onnx_path):
            print(f"  ❌ ONNX文件不存在: {onnx_path}")
            return False
        
        # 创建builder
        builder = trt.Builder(self.logger)
        
        # 创建network
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # 创建ONNX parser
        parser = trt.OnnxParser(network, self.logger)
        
        # 解析ONNX模型
        print(f"  📖 解析ONNX模型...")
        
        # 检查是否有外部数据文件（针对大模型如 vit_h）
        onnx_dir = os.path.dirname(onnx_path) or '.'
        
        # 方法1: 使用 parse_from_file 可以自动处理外部数据
        # 这比手动读取文件更好，因为它会自动加载外部数据
        success = parser.parse_from_file(onnx_path)
        
        if not success:
            print(f"  ❌ ONNX解析失败:")
            for i in range(parser.num_errors):
                error = parser.get_error(i)
                print(f"    错误 {i}: {error}")
            
            # 尝试方法2: 使用 onnx 库先加载再序列化
            print(f"\n  🔄 尝试使用 onnx 库加载（支持外部数据）...")
            try:
                import onnx
                # 加载 ONNX 模型，自动处理外部数据
                onnx_model = onnx.load(onnx_path)
                # 序列化为字节流
                onnx_bytes = onnx_model.SerializeToString()
                # 使用序列化的字节流解析
                if not parser.parse(onnx_bytes):
                    print(f"  ❌ 二次解析也失败")
                    for i in range(parser.num_errors):
                        error = parser.get_error(i)
                        print(f"    错误 {i}: {error}")
                    return False
                else:
                    print(f"  ✅ 使用 onnx 库加载成功")
            except Exception as e:
                print(f"  ❌ onnx 库加载失败: {e}")
                return False
        else:
            print(f"  ✅ ONNX解析成功")
        
        # 打印网络信息
        print(f"\n  📊 网络信息:")
        print(f"    输入数量: {network.num_inputs}")
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            print(f"      输入 {i}: {input_tensor.name}, 形状: {input_tensor.shape}, 类型: {input_tensor.dtype}")
        
        print(f"    输出数量: {network.num_outputs}")
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            print(f"      输出 {i}: {output_tensor.name}, 形状: {output_tensor.shape}, 类型: {output_tensor.dtype}")
        
        # 创建builder config
        config = builder.create_builder_config()
        
        # 设置工作空间大小
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace_size)
        
        # 设置精度
        if self.precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print(f"  ✅ 启用FP16精度")
            else:
                print(f"  ⚠️  平台不支持FP16，使用FP32")
        elif self.precision == 'bf16':
            # BF16 支持（TensorRT 8.6+）
            try:
                if hasattr(trt.BuilderFlag, 'BF16'):
                    config.set_flag(trt.BuilderFlag.BF16)
                    print(f"  ✅ 启用BF16精度")
                else:
                    print(f"  ⚠️  TensorRT版本不支持BF16，使用FP16")
                    if builder.platform_has_fast_fp16:
                        config.set_flag(trt.BuilderFlag.FP16)
            except Exception as e:
                print(f"  ⚠️  BF16设置失败: {e}，使用FP16")
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == 'int8':
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                print(f"  ✅ 启用INT8精度")
            else:
                print(f"  ⚠️  平台不支持INT8，使用FP32")
        
        # 配置动态形状
        if dynamic_shapes and (min_shapes or opt_shapes or max_shapes):
            print(f"\n  🔄 配置动态形状:")
            profile = builder.create_optimization_profile()
            
            # 获取所有输入张量信息
            input_is_shape = {}
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                input_is_shape[input_tensor.name] = input_tensor.is_shape_tensor
            
            for input_name in dynamic_shapes:
                min_shape = min_shapes.get(input_name) if min_shapes else None
                opt_shape = opt_shapes.get(input_name) if opt_shapes else None
                max_shape = max_shapes.get(input_name) if max_shapes else None
                
                if min_shape and opt_shape and max_shape:
                    # 检查是否为shape tensor
                    is_shape_tensor = input_is_shape.get(input_name, False)
                    
                    if is_shape_tensor:
                        # 对于shape tensor，需要使用set_shape_input
                        # 输入的是值的范围，而不是形状
                        profile.set_shape_input(input_name, min_shape, opt_shape, max_shape)
                        print(f"    {input_name} (shape input):")
                        print(f"      min values: {min_shape}")
                        print(f"      opt values: {opt_shape}")
                        print(f"      max values: {max_shape}")
                    else:
                        # 对于普通tensor，使用set_shape
                        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                        print(f"    {input_name}:")
                        print(f"      min: {min_shape}")
                        print(f"      opt: {opt_shape}")
                        print(f"      max: {max_shape}")
            
            config.add_optimization_profile(profile)
        
        # 构建引擎
        print(f"\n  🔨 构建TensorRT引擎 (可能需要几分钟)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print(f"  ❌ 引擎构建失败")
            return False
        
        print(f"  ✅ 引擎构建成功")
        
        # 保存引擎
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        engine_size_mb = os.path.getsize(engine_path) / (1024 ** 2)
        print(f"  💾 引擎已保存: {engine_path} ({engine_size_mb:.1f} MB)")
        
        print("="*80)
        return True
    
    def convert_encoder(self, onnx_path, engine_path, batch_size=1):
        """
        转换encoder模型
        
        Args:
            onnx_path: encoder ONNX路径
            engine_path: 输出引擎路径
            batch_size: batch大小（默认1）
        
        Returns:
            bool: 是否成功
        """
        print("\n" + "🎯 转换Encoder模型")
        
        # Encoder的batch维度是动态的，需要配置
        dynamic_shapes = ['images']  # 修正：实际名称是 'images'
        
        min_shapes = {
            'images': (batch_size, 3, 1024, 1024)
        }
        
        opt_shapes = {
            'images': (batch_size, 3, 1024, 1024)
        }
        
        max_shapes = {
            'images': (batch_size, 3, 1024, 1024)
        }
        
        return self.build_engine_from_onnx(
            onnx_path=onnx_path,
            engine_path=engine_path,
            dynamic_shapes=dynamic_shapes,
            min_shapes=min_shapes,
            opt_shapes=opt_shapes,
            max_shapes=max_shapes
        )
    
    def convert_decoder(self, onnx_path, engine_path, 
                       min_points=1, opt_points=3, max_points=10,
                       min_img_size=256, opt_img_size=1024, max_img_size=2048):
        """
        转换decoder模型（支持动态点数）
        
        Args:
            onnx_path: decoder ONNX路径
            engine_path: 输出引擎路径
            min_points: 最小点数
            opt_points: 最优点数
            max_points: 最大点数
            min_img_size: 最小图像尺寸
            opt_img_size: 最优图像尺寸
            max_img_size: 最大图像尺寸
        
        Returns:
            bool: 是否成功
        """
        print("\n" + "🎯 转换Decoder模型 (动态形状)")
        
        # Decoder的point_coords、point_labels和orig_im_size都需要配置
        # 注意: orig_im_size是shape input tensor，需要特殊处理
        dynamic_shapes = ['point_coords', 'point_labels', 'orig_im_size']
        
        min_shapes = {
            'point_coords': (1, min_points, 2),
            'point_labels': (1, min_points),
            'orig_im_size': (min_img_size, min_img_size)  # 值的范围，不是形状
        }
        
        opt_shapes = {
            'point_coords': (1, opt_points, 2),
            'point_labels': (1, opt_points),
            'orig_im_size': (opt_img_size, opt_img_size)
        }
        
        max_shapes = {
            'point_coords': (1, max_points, 2),
            'point_labels': (1, max_points),
            'orig_im_size': (max_img_size, max_img_size)
        }
        
        return self.build_engine_from_onnx(
            onnx_path=onnx_path,
            engine_path=engine_path,
            dynamic_shapes=dynamic_shapes,
            min_shapes=min_shapes,
            opt_shapes=opt_shapes,
            max_shapes=max_shapes
        )


def main():
    parser = argparse.ArgumentParser(description='使用Python TensorRT API转换ONNX模型')
    parser.add_argument('--encoder-onnx', type=str, default='sam_hq_vit_b_encoder.onnx',
                       help='Encoder ONNX模型路径')
    parser.add_argument('--decoder-onnx', type=str, default='sam_hq_official_decoder_trt_fixed.onnx',
                       help='Decoder ONNX模型路径')
    parser.add_argument('--output-dir', type=str, default='tensorrt_engines',
                       help='输出目录')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'bf16', 'int8'],
                       help='精度模式')
    parser.add_argument('--workspace', type=int, default=4096,
                       help='最大工作空间大小 (MB)')
    parser.add_argument('--min-points', type=int, default=1,
                       help='Decoder最小点数')
    parser.add_argument('--opt-points', type=int, default=3,
                       help='Decoder最优点数')
    parser.add_argument('--max-points', type=int, default=10,
                       help='Decoder最大点数')
    parser.add_argument('--min-img-size', type=int, default=256,
                       help='Decoder最小图像尺寸')
    parser.add_argument('--opt-img-size', type=int, default=1024,
                       help='Decoder最优图像尺寸')
    parser.add_argument('--max-img-size', type=int, default=4096,
                       help='Decoder最大图像尺寸')
    parser.add_argument('--skip-encoder', action='store_true',
                       help='跳过encoder转换')
    parser.add_argument('--skip-decoder', action='store_true',
                       help='跳过decoder转换')
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = TensorRTConverter(
        precision=args.precision,
        max_workspace_size=args.workspace
    )
    
    success_count = 0
    total_count = 0
    
    # 转换encoder
    if not args.skip_encoder:
        total_count += 1
        encoder_engine_path = os.path.join(
            args.output_dir, 
            f'sam_hq_vit_b_encoder_{args.precision}.engine'
        )
        
        if converter.convert_encoder(args.encoder_onnx, encoder_engine_path):
            success_count += 1
            print(f"\n✅ Encoder转换成功: {encoder_engine_path}")
        else:
            print(f"\n❌ Encoder转换失败")
    
    # 转换decoder
    if not args.skip_decoder:
        total_count += 1
        decoder_engine_path = os.path.join(
            args.output_dir,
            f'sam_hq_official_decoder_{args.precision}.engine'
        )
        
        if converter.convert_decoder(
            args.decoder_onnx, 
            decoder_engine_path,
            min_points=args.min_points,
            opt_points=args.opt_points,
            max_points=args.max_points,
            min_img_size=args.min_img_size,
            opt_img_size=args.opt_img_size,
            max_img_size=args.max_img_size
        ):
            success_count += 1
            print(f"\n✅ Decoder转换成功: {decoder_engine_path}")
        else:
            print(f"\n❌ Decoder转换失败")
    
    # 总结
    print("\n" + "="*80)
    print("📊 转换总结")
    print("="*80)
    print(f"  成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 所有模型转换成功！")
        print("\n下一步: 运行以下命令测试模型:")
        print(f"  python compare_all_models.py \\")
        print(f"    --image <your_image.jpg> \\")
        print(f"    --points \"x1,y1,1;x2,y2,1\" \\")
        print(f"    --encoder-trt {args.output_dir}/sam_hq_vit_b_encoder_{args.precision}.engine \\")
        print(f"    --decoder-trt {args.output_dir}/sam_hq_official_decoder_{args.precision}.engine")
    else:
        print("\n⚠️  部分模型转换失败，请检查错误信息")
    
    print("="*80)


if __name__ == '__main__':
    main()
