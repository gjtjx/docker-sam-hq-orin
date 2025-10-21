#!/usr/bin/env python3
"""
完整模型对比：PyTorch、ONNX CUDA、TensorRT
验证一致性并统计性能，生成可视化对比图
直接使用TensorRT引擎文件进行推理
"""

import numpy as np
import cv2
import torch
import onnxruntime as ort
import tensorrt as trt
import argparse
import time
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import os
warnings.filterwarnings("ignore")


class TensorRTInference:
    """TensorRT推理混合策略版 - Encoder用torch快速，Decoder用numpy稳定"""
    
    def __init__(self, engine_path, use_fast_mode='auto'):
        """
        初始化TensorRT引擎
        
        Args:
            engine_path: 引擎文件路径
            use_fast_mode: 'auto'(自动检测), 'fast'(torch tensor), 'stable'(numpy)
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"无法加载TensorRT引擎: {engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        # 创建专用CUDA stream
        self.stream = torch.cuda.Stream()
        
        # 获取输入输出信息
        self.input_names = []
        self.output_names = []
        self.output_specs = {}
        self.has_dynamic_output = False
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_specs[name] = {'shape': shape, 'dtype': dtype}
                if -1 in shape:
                    self.has_dynamic_output = True
        
        # 自动选择模式
        if use_fast_mode == 'auto':
            # decoder有动态输出，使用稳定模式
            self.use_fast_mode = not self.has_dynamic_output
        else:
            self.use_fast_mode = (use_fast_mode == 'fast')
        
        # 预分配buffers
        self._gpu_buffers = {}  # torch tensor buffers
        self._numpy_buffers = {}  # numpy buffers
    
    def _get_torch_dtype(self, trt_dtype):
        """TensorRT dtype转torch dtype"""
        dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }
        return dtype_map.get(trt_dtype, torch.float32)
    
    def _get_numpy_dtype(self, trt_dtype):
        """TensorRT dtype转numpy dtype"""
        dtype_map = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT32: np.int32,
            trt.DataType.INT64: np.int64,
            trt.DataType.BOOL: bool,
        }
        return dtype_map.get(trt_dtype, np.float32)
    
    def infer(self, inputs, output_shapes_hint=None, return_tensors=False):
        """
        执行推理
        
        Args:
            inputs: 输入字典 {name: numpy.ndarray或torch.Tensor}
            output_shapes_hint: 可选的输出shape提示
            return_tensors: 是否返回torch tensor（默认返回numpy）
            
        Returns:
            list: 输出数组列表
        """
        # 根据模式选择实现
        if self.use_fast_mode:
            return self._infer_fast(inputs, output_shapes_hint, return_tensors)
        else:
            return self._infer_stable(inputs, output_shapes_hint)
    
    def _infer_fast(self, inputs, output_shapes_hint=None, return_tensors=False):
        """快速模式 - 使用torch GPU tensor"""
        with torch.cuda.stream(self.stream):
            # 转换输入为GPU tensor
            for name, data in inputs.items():
                if name not in self.input_names:
                    continue
                
                if isinstance(data, np.ndarray):
                    tensor = torch.from_numpy(data).cuda()
                elif isinstance(data, torch.Tensor):
                    tensor = data.cuda() if not data.is_cuda else data
                else:
                    raise TypeError(f"不支持的输入类型: {type(data)}")
                
                tensor = tensor.contiguous()
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, tensor.data_ptr())
            
            # 分配输出buffers
            output_tensors = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = self._get_torch_dtype(self.output_specs[name]['dtype'])
                
                # 复用GPU buffer
                key = (name, shape, dtype)
                if key in self._gpu_buffers:
                    output_tensor = self._gpu_buffers[key]
                else:
                    output_tensor = torch.empty(shape, dtype=dtype, device='cuda')
                    self._gpu_buffers[key] = output_tensor
                
                output_tensors[name] = output_tensor
                self.context.set_tensor_address(name, output_tensor.data_ptr())
            
            # 执行推理
            success = self.context.execute_async_v3(self.stream.cuda_stream)
            if not success:
                raise RuntimeError("TensorRT推理失败")
        
        # 同步stream
        self.stream.synchronize()
        
        # 返回结果
        if return_tensors:
            return [output_tensors[name] for name in self.output_names]
        else:
            return [output_tensors[name].cpu().numpy() for name in self.output_names]
    
    def _infer_stable(self, inputs, output_shapes_hint=None):
        """稳定模式 - 使用numpy + ctypes（支持动态shape）"""
        # 设置输入
        for name, data in inputs.items():
            if name not in self.input_names:
                continue
            
            # 确保是numpy数组
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            
            # 确保是contiguous
            data = np.ascontiguousarray(data)
            
            # 复用或分配buffer
            key = (name, data.shape, data.dtype)
            if key in self._numpy_buffers:
                np.copyto(self._numpy_buffers[key], data)
                data = self._numpy_buffers[key]
            else:
                self._numpy_buffers[key] = data
            
            self.context.set_input_shape(name, data.shape)
            self.context.set_tensor_address(name, data.ctypes.data)
        
        # 分配输出
        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            
            # 处理动态shape
            if -1 in shape:
                if output_shapes_hint and name in output_shapes_hint:
                    shape = output_shapes_hint[name]
                    self.context.set_tensor_shape(name, shape)
                else:
                    raise RuntimeError(f"输出'{name}'需要output_shapes_hint")
            
            dtype = self._get_numpy_dtype(self.output_specs[name]['dtype'])
            
            # 复用或分配buffer
            key = (name, shape, dtype)
            if key in self._numpy_buffers:
                output = self._numpy_buffers[key]
            else:
                output = np.empty(shape, dtype=dtype)
                self._numpy_buffers[key] = output
            
            outputs[name] = output
            self.context.set_tensor_address(name, output.ctypes.data)
        
        # 执行推理
        success = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        
        if not success:
            raise RuntimeError("TensorRT推理失败")
        
        return [outputs[name] for name in self.output_names]
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, '_gpu_buffers'):
            self._gpu_buffers.clear()
        if hasattr(self, '_numpy_buffers'):
            self._numpy_buffers.clear()


def preprocess_image_for_onnx(image, target_length=1024):
    """预处理图像用于ONNX encoder"""
    transformer = ResizeLongestSide(target_length)
    
    input_image = transformer.apply_image(image)
    h, w = input_image.shape[:2]
    
    padh = target_length - h
    padw = target_length - w
    input_image = np.pad(input_image, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)
    
    input_image_torch = input_image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)
    input_image_torch = (input_image_torch - mean) / std
    
    return input_image_torch


def apply_coords(coords, original_size, target_length=1024):
    """变换点坐标到模型输入空间"""
    old_h, old_w = original_size
    scale = target_length * 1.0 / max(old_h, old_w)
    new_h, new_w = int(old_h * scale + 0.5), int(old_w * scale + 0.5)
    
    coords = coords.copy().astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    
    return coords


def benchmark_pytorch(checkpoint_path, image, points, labels, model_type='vit_b', num_runs=20, warmup=10):
    """PyTorch GPU基准测试"""
    print("\n" + "="*80)
    print("🔥 PyTorch GPU 基准测试")
    print("="*80)
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam.to(device=device)
    
    predictor = SamPredictor(sam)
    
    # 预热
    print(f"  预热 {warmup} 次...")
    for _ in range(warmup):
        predictor.set_image(image)
        _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=False, hq_token_only=True)
    
    # 测试encoder
    encoder_times = []
    for i in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        predictor.set_image(image)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        encoder_times.append((time.time() - t0) * 1000)
    
    # 测试decoder
    decoder_times = []
    predictor.set_image(image)  # 设置一次
    for i in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=False, hq_token_only=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        decoder_times.append((time.time() - t0) * 1000)
    
    encoder_mean = np.mean(encoder_times)
    encoder_std = np.std(encoder_times)
    decoder_mean = np.mean(decoder_times)
    decoder_std = np.std(decoder_times)
    total_mean = encoder_mean + decoder_mean
    
    print(f"\n  ✅ 结果 ({num_runs} 次运行):")
    print(f"    Encoder: {encoder_mean:.2f} ± {encoder_std:.2f} ms")
    print(f"    Decoder: {decoder_mean:.2f} ± {decoder_std:.2f} ms")
    print(f"    总耗时:  {total_mean:.2f} ms")
    
    mask_pytorch = masks[0]
    fg_pct = 100 * mask_pytorch.sum() / mask_pytorch.size
    print(f"    前景像素: {mask_pytorch.sum():,} ({fg_pct:.2f}%)")
    
    return {
        'encoder_mean': encoder_mean,
        'encoder_std': encoder_std,
        'decoder_mean': decoder_mean,
        'decoder_std': decoder_std,
        'total_mean': total_mean,
        'mask': mask_pytorch,
        'encoder_times': encoder_times,
        'decoder_times': decoder_times
    }


def benchmark_onnx_cuda(encoder_path, decoder_path, image, points, labels, num_runs=20, warmup=10):
    """测试ONNX CUDA性能"""
    print("\n" + "="*80)
    print("⚡ ONNX CUDA 基准测试")
    print("="*80)
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # 准备输入
    input_image_onnx = preprocess_image_for_onnx(image, target_length=1024)
    original_size = image.shape[:2]
    
    onnx_coord = np.concatenate([points, [[-1, -1]]], axis=0)[None, :, :]
    onnx_label = np.concatenate([labels, [-1]], axis=0)[None, :].astype(np.float32)
    onnx_coord = apply_coords(onnx_coord[0], original_size)[None, :, :].astype(np.float32)
    
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.array([0], dtype=np.float32)
    orig_im_size = np.array(original_size, dtype=np.float32)
    
    # 加载模型
    print(f"  加载ONNX模型...")
    encoder_session = ort.InferenceSession(encoder_path, providers=providers)
    decoder_session = ort.InferenceSession(decoder_path, providers=providers)
    
    print(f"  Encoder Provider: {encoder_session.get_providers()[0]}")
    print(f"  Decoder Provider: {decoder_session.get_providers()[0]}")
    
    # 预热
    print(f"  预热 {warmup} 次...")
    for _ in range(warmup):
        encoder_outputs = encoder_session.run(None, {'images': input_image_onnx})
        image_embeddings = encoder_outputs[0]
        interm_embeddings = encoder_outputs[1]
        
        decoder_inputs = {
            'image_embeddings': image_embeddings,
            'interm_embeddings': interm_embeddings,
            'point_coords': onnx_coord,
            'point_labels': onnx_label,
            'mask_input': mask_input,
            'has_mask_input': has_mask_input,
            'orig_im_size': orig_im_size,
        }
        _ = decoder_session.run(None, decoder_inputs)
    
    # 测试encoder
    encoder_times = []
    for i in range(num_runs):
        t0 = time.time()
        encoder_outputs = encoder_session.run(None, {'images': input_image_onnx})
        encoder_times.append((time.time() - t0) * 1000)
    
    image_embeddings = encoder_outputs[0]
    interm_embeddings = encoder_outputs[1]
    
    # 测试decoder
    decoder_inputs = {
        'image_embeddings': image_embeddings,
        'interm_embeddings': interm_embeddings,
        'point_coords': onnx_coord,
        'point_labels': onnx_label,
        'mask_input': mask_input,
        'has_mask_input': has_mask_input,
        'orig_im_size': orig_im_size,
    }
    
    decoder_times = []
    for i in range(num_runs):
        t0 = time.time()
        decoder_outputs = decoder_session.run(None, decoder_inputs)
        decoder_times.append((time.time() - t0) * 1000)
    
    encoder_mean = np.mean(encoder_times)
    encoder_std = np.std(encoder_times)
    decoder_mean = np.mean(decoder_times)
    decoder_std = np.std(decoder_times)
    total_mean = encoder_mean + decoder_mean
    
    print(f"\n  ✅ 结果 ({num_runs} 次运行):")
    print(f"    Encoder: {encoder_mean:.2f} ± {encoder_std:.2f} ms")
    print(f"    Decoder: {decoder_mean:.2f} ± {decoder_std:.2f} ms")
    print(f"    总耗时:  {total_mean:.2f} ms")
    
    masks_onnx = decoder_outputs[0]
    mask_onnx = (masks_onnx[0, 0] > 0.0)
    fg_pct = 100 * mask_onnx.sum() / mask_onnx.size
    print(f"    前景像素: {mask_onnx.sum():,} ({fg_pct:.2f}%)")
    
    return {
        'encoder_mean': encoder_mean,
        'encoder_std': encoder_std,
        'decoder_mean': decoder_mean,
        'decoder_std': decoder_std,
        'total_mean': total_mean,
        'mask': mask_onnx,
        'encoder_times': encoder_times,
        'decoder_times': decoder_times
    }


def benchmark_tensorrt(encoder_engine_path, decoder_onnx_path, image, points, labels, num_runs=20, warmup=10):
    """测试TensorRT Encoder + ONNX Decoder混合模式"""
    print("\n" + "="*80)
    print("🚀 TensorRT Encoder + ONNX Decoder 混合模式")
    print("="*80)
    
    # 加载TensorRT Encoder引擎
    print(f"  加载TensorRT Encoder引擎...")
    encoder_engine = TensorRTInference(encoder_engine_path)
    print(f"  ✓ TensorRT引擎加载完成")
    
    # 加载ONNX Decoder
    print(f"  加载ONNX Decoder...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    decoder_session = ort.InferenceSession(decoder_onnx_path, providers=providers)
    print(f"  Decoder Provider: {decoder_session.get_providers()[0]}")
    print(f"  ✓ ONNX Decoder加载完成")
    
    # 准备输入（使用ONNX预处理）
    input_image_onnx = preprocess_image_for_onnx(image, target_length=1024)
    original_size = image.shape[:2]
    
    # 准备decoder输入
    onnx_coord = np.concatenate([points, [[-1, -1]]], axis=0)[None, :, :]
    onnx_label = np.concatenate([labels, [-1]], axis=0)[None, :].astype(np.float32)
    onnx_coord = apply_coords(onnx_coord[0], original_size)[None, :, :].astype(np.float32)
    
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.array([0], dtype=np.float32)
    orig_im_size = np.array(original_size, dtype=np.float32)  # ONNX decoder期望float32
    
    # 预热
    print(f"  预热 {warmup} 次...")
    for _ in range(warmup):
        # TensorRT Encoder
        encoder_outputs = encoder_engine.infer({'images': input_image_onnx})
        image_embeddings = encoder_outputs[0]
        interm_embeddings = encoder_outputs[1]
        
        # ONNX Decoder
        decoder_inputs = {
            'image_embeddings': image_embeddings,
            'interm_embeddings': interm_embeddings,
            'point_coords': onnx_coord,
            'point_labels': onnx_label,
            'mask_input': mask_input,
            'has_mask_input': has_mask_input,
            'orig_im_size': orig_im_size,
        }
        _ = decoder_session.run(None, decoder_inputs)
    
    # 测试encoder (TensorRT)
    encoder_times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        encoder_outputs = encoder_engine.infer({'images': input_image_onnx})
        torch.cuda.synchronize()
        encoder_times.append((time.time() - t0) * 1000)
    
    image_embeddings = encoder_outputs[0]
    interm_embeddings = encoder_outputs[1]
    
    # 测试decoder (ONNX)
    decoder_inputs = {
        'image_embeddings': image_embeddings,
        'interm_embeddings': interm_embeddings,
        'point_coords': onnx_coord,
        'point_labels': onnx_label,
        'mask_input': mask_input,
        'has_mask_input': has_mask_input,
        'orig_im_size': orig_im_size,
    }
    
    decoder_times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        decoder_outputs = decoder_session.run(None, decoder_inputs)
        torch.cuda.synchronize()
        decoder_times.append((time.time() - t0) * 1000)
    
    encoder_mean = np.mean(encoder_times)
    encoder_std = np.std(encoder_times)
    decoder_mean = np.mean(decoder_times)
    decoder_std = np.std(decoder_times)
    total_mean = encoder_mean + decoder_mean
    
    print(f"\n  ✅ 结果 ({num_runs} 次运行):")
    print(f"    TensorRT Encoder: {encoder_mean:.2f} ± {encoder_std:.2f} ms")
    print(f"    ONNX Decoder:     {decoder_mean:.2f} ± {decoder_std:.2f} ms")
    print(f"    总耗时:           {total_mean:.2f} ms")
    
    # 处理输出mask
    masks_hybrid = decoder_outputs[0]
    mask_hybrid = (masks_hybrid[0, 0] > 0.0)
    fg_pct = 100 * mask_hybrid.sum() / mask_hybrid.size
    print(f"    前景像素: {mask_hybrid.sum():,} ({fg_pct:.2f}%)")
    
    return {
        'encoder_mean': encoder_mean,
        'encoder_std': encoder_std,
        'decoder_mean': decoder_mean,
        'decoder_std': decoder_std,
        'total_mean': total_mean,
        'mask': mask_hybrid,
        'encoder_times': encoder_times,
        'decoder_times': decoder_times
    }


def benchmark_tensorrt_full(encoder_engine_path, decoder_engine_path, image, points, labels, num_runs=20, warmup=10):
    """测试纯TensorRT推理（Encoder + Decoder都使用TensorRT）"""
    print("\n" + "="*80)
    print("🚀 纯TensorRT推理 (Encoder + Decoder)")
    print("="*80)
    
    # 加载TensorRT Encoder引擎
    print(f"  加载TensorRT Encoder引擎...")
    encoder_engine = TensorRTInference(encoder_engine_path)
    print(f"  ✓ TensorRT Encoder加载完成")
    
    # 加载TensorRT Decoder引擎
    print(f"  加载TensorRT Decoder引擎...")
    decoder_engine = TensorRTInference(decoder_engine_path)
    print(f"  ✓ TensorRT Decoder加载完成")
    
    # 准备输入（使用ONNX预处理）
    input_image_onnx = preprocess_image_for_onnx(image, target_length=1024)
    original_size = image.shape[:2]
    
    # 准备decoder输入
    onnx_coord = np.concatenate([points, [[-1, -1]]], axis=0)[None, :, :]
    onnx_label = np.concatenate([labels, [-1]], axis=0)[None, :].astype(np.float32)
    onnx_coord = apply_coords(onnx_coord[0], original_size)[None, :, :].astype(np.float32)
    
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.array([0.0], dtype=np.float32)
    orig_im_size = np.array(original_size, dtype=np.int64)  # TensorRT decoder期望int64
    
    # 计算decoder输出shape（基于原始图像尺寸）
    h, w = original_size
    decoder_output_shapes = {
        'masks': (1, 1, h, w),
        'iou_predictions': (1, 1),
        'low_res_masks': (1, 1, 256, 256),
    }
    
    # 预热
    print(f"  预热 {warmup} 次...")
    for _ in range(warmup):
        # TensorRT Encoder
        encoder_outputs = encoder_engine.infer({'images': input_image_onnx})
        image_embeddings = encoder_outputs[0]
        interm_embeddings = encoder_outputs[1]
        
        # TensorRT Decoder
        decoder_inputs = {
            'image_embeddings': image_embeddings,
            'interm_embeddings': interm_embeddings,
            'point_coords': onnx_coord,
            'point_labels': onnx_label,
            'mask_input': mask_input,
            'has_mask_input': has_mask_input,
            'orig_im_size': orig_im_size,
        }
        _ = decoder_engine.infer(decoder_inputs, output_shapes_hint=decoder_output_shapes)
    
    # 测试encoder (TensorRT)
    encoder_times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        encoder_outputs = encoder_engine.infer({'images': input_image_onnx})
        torch.cuda.synchronize()
        encoder_times.append((time.time() - t0) * 1000)
    
    image_embeddings = encoder_outputs[0]
    interm_embeddings = encoder_outputs[1]
    
    # 测试decoder (TensorRT)
    decoder_inputs = {
        'image_embeddings': image_embeddings,
        'interm_embeddings': interm_embeddings,
        'point_coords': onnx_coord,
        'point_labels': onnx_label,
        'mask_input': mask_input,
        'has_mask_input': has_mask_input,
        'orig_im_size': orig_im_size,
    }
    
    decoder_times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        decoder_outputs = decoder_engine.infer(decoder_inputs, output_shapes_hint=decoder_output_shapes)
        torch.cuda.synchronize()
        decoder_times.append((time.time() - t0) * 1000)
    
    encoder_mean = np.mean(encoder_times)
    encoder_std = np.std(encoder_times)
    decoder_mean = np.mean(decoder_times)
    decoder_std = np.std(decoder_times)
    total_mean = encoder_mean + decoder_mean
    
    print(f"\n  ✅ 结果 ({num_runs} 次运行):")
    print(f"    TensorRT Encoder: {encoder_mean:.2f} ± {encoder_std:.2f} ms")
    print(f"    TensorRT Decoder: {decoder_mean:.2f} ± {decoder_std:.2f} ms")
    print(f"    总耗时:           {total_mean:.2f} ms")
    
    # 处理输出mask
    masks_trt = decoder_outputs[0]
    mask_trt = (masks_trt[0, 0] > 0.0)
    fg_pct = 100 * mask_trt.sum() / mask_trt.size
    print(f"    前景像素: {mask_trt.sum():,} ({fg_pct:.2f}%)")
    
    return {
        'encoder_mean': encoder_mean,
        'encoder_std': encoder_std,
        'decoder_mean': decoder_mean,
        'decoder_std': decoder_std,
        'total_mean': total_mean,
        'mask': mask_trt,
        'encoder_times': encoder_times,
        'decoder_times': decoder_times
    }


def calculate_iou(mask1, mask2):
    """计算IoU"""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def calculate_dice(mask1, mask2):
    """计算Dice系数"""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    return 2 * intersection / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0


def create_visualization(image, points, labels, results, output_path):
    """创建可视化对比图"""
    print("\n" + "="*80)
    print("📊 生成可视化对比图")
    print("="*80)
    
    # 根据结果数量调整布局
    num_methods = len(results)
    if num_methods == 4:
        fig = plt.figure(figsize=(24, 20))
        rows, cols = 4, 4
    else:
        fig = plt.figure(figsize=(24, 16))
        rows, cols = 3, 4
    
    # 颜色映射
    colors = {1: 'green', 0: 'red'}
    
    # 1. 原始图像
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i, (point, label) in enumerate(zip(points, labels)):
        ax1.plot(point[0], point[1], 'o', markersize=12, 
                color=colors[label], markeredgecolor='white', markeredgewidth=2)
        ax1.text(point[0], point[1]-30, f'P{i+1}', color='white', 
                fontsize=12, ha='center', weight='bold',
                bbox=dict(boxstyle='round', facecolor=colors[label], alpha=0.7))
    ax1.set_title('原始图像 + 输入点', fontsize=14, weight='bold')
    ax1.axis('off')
    
    # 2-4. PyTorch结果
    if 'pytorch' in results:
        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_rgb = np.zeros_like(image)
        mask_rgb[results['pytorch']['mask']] = [0, 255, 0]
        ax2.imshow(mask_rgb, alpha=0.5)
        ax2.set_title(f'PyTorch GPU\n总耗时: {results["pytorch"]["total_mean"]:.2f}ms', 
                     fontsize=12, weight='bold')
        ax2.axis('off')
        
        ax3 = plt.subplot(3, 4, 3)
        ax3.imshow(results['pytorch']['mask'], cmap='gray')
        ax3.set_title(f'PyTorch Mask\n前景: {results["pytorch"]["mask"].sum():,} 像素', 
                     fontsize=12)
        ax3.axis('off')
        
        # PyTorch时序分布
        ax4 = plt.subplot(3, 4, 4)
        ax4.hist(results['pytorch']['encoder_times'], bins=20, alpha=0.7, label='Encoder', color='blue')
        ax4.hist(results['pytorch']['decoder_times'], bins=20, alpha=0.7, label='Decoder', color='orange')
        ax4.set_xlabel('时间 (ms)')
        ax4.set_ylabel('频次')
        ax4.set_title('PyTorch 时序分布', fontsize=12, weight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5-7. ONNX CUDA结果
    if 'onnx' in results:
        ax5 = plt.subplot(3, 4, 5)
        ax5.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_rgb = np.zeros_like(image)
        mask_rgb[results['onnx']['mask']] = [0, 255, 0]
        ax5.imshow(mask_rgb, alpha=0.5)
        speedup = results['pytorch']['total_mean'] / results['onnx']['total_mean'] if 'pytorch' in results else 1.0
        ax5.set_title(f'ONNX CUDA\n总耗时: {results["onnx"]["total_mean"]:.2f}ms ({speedup:.2f}x)', 
                     fontsize=12, weight='bold', color='green' if speedup > 1 else 'black')
        ax5.axis('off')
        
        ax6 = plt.subplot(3, 4, 6)
        ax6.imshow(results['onnx']['mask'], cmap='gray')
        if 'pytorch' in results:
            iou = calculate_iou(results['pytorch']['mask'], results['onnx']['mask'])
            dice = calculate_dice(results['pytorch']['mask'], results['onnx']['mask'])
            ax6.set_title(f'ONNX Mask\nIoU: {iou*100:.2f}% | Dice: {dice*100:.2f}%', 
                         fontsize=12, color='green' if iou > 0.95 else 'orange')
        else:
            ax6.set_title(f'ONNX Mask\n前景: {results["onnx"]["mask"].sum():,} 像素', fontsize=12)
        ax6.axis('off')
        
        # ONNX时序分布
        ax7 = plt.subplot(3, 4, 7)
        ax7.hist(results['onnx']['encoder_times'], bins=20, alpha=0.7, label='Encoder', color='blue')
        ax7.hist(results['onnx']['decoder_times'], bins=20, alpha=0.7, label='Decoder', color='orange')
        ax7.set_xlabel('时间 (ms)')
        ax7.set_ylabel('频次')
        ax7.set_title('ONNX CUDA 时序分布', fontsize=12, weight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8-10. TensorRT混合模式结果 (TensorRT Enc + ONNX Dec)
    if 'tensorrt' in results:
        ax8 = plt.subplot(rows, cols, 9)
        ax8.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_rgb = np.zeros_like(image)
        mask_rgb[results['tensorrt']['mask']] = [0, 255, 0]
        ax8.imshow(mask_rgb, alpha=0.5)
        speedup = results['pytorch']['total_mean'] / results['tensorrt']['total_mean'] if 'pytorch' in results else 1.0
        ax8.set_title(f'TensorRT Enc + ONNX Dec\n总耗时: {results["tensorrt"]["total_mean"]:.2f}ms ({speedup:.2f}x)', 
                     fontsize=12, weight='bold', color='green' if speedup > 1 else 'black')
        ax8.axis('off')
        
        ax9 = plt.subplot(rows, cols, 10)
        ax9.imshow(results['tensorrt']['mask'], cmap='gray')
        if 'pytorch' in results:
            iou = calculate_iou(results['pytorch']['mask'], results['tensorrt']['mask'])
            dice = calculate_dice(results['pytorch']['mask'], results['tensorrt']['mask'])
            ax9.set_title(f'TensorRT-Hybrid Mask\nIoU: {iou*100:.2f}% | Dice: {dice*100:.2f}%', 
                         fontsize=12, color='green' if iou > 0.95 else 'orange')
        else:
            ax9.set_title(f'TensorRT-Hybrid Mask\n前景: {results["tensorrt"]["mask"].sum():,} 像素', fontsize=12)
        ax9.axis('off')
        
        # TensorRT混合时序分布
        ax10 = plt.subplot(rows, cols, 11)
        ax10.hist(results['tensorrt']['encoder_times'], bins=20, alpha=0.7, label='TRT Encoder', color='blue')
        ax10.hist(results['tensorrt']['decoder_times'], bins=20, alpha=0.7, label='ONNX Decoder', color='orange')
        ax10.set_xlabel('时间 (ms)')
        ax10.set_ylabel('频次')
        ax10.set_title('TensorRT混合 时序分布', fontsize=12, weight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
    
    # 11-13. 纯TensorRT结果 (TensorRT Enc + TensorRT Dec)
    if 'tensorrt_full' in results:
        ax11 = plt.subplot(rows, cols, 13)
        ax11.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_rgb = np.zeros_like(image)
        mask_rgb[results['tensorrt_full']['mask']] = [0, 255, 0]
        ax11.imshow(mask_rgb, alpha=0.5)
        speedup = results['pytorch']['total_mean'] / results['tensorrt_full']['total_mean'] if 'pytorch' in results else 1.0
        ax11.set_title(f'纯TensorRT (Enc+Dec)\n总耗时: {results["tensorrt_full"]["total_mean"]:.2f}ms ({speedup:.2f}x)', 
                      fontsize=12, weight='bold', color='darkgreen' if speedup > 1 else 'black')
        ax11.axis('off')
        
        ax12 = plt.subplot(rows, cols, 14)
        ax12.imshow(results['tensorrt_full']['mask'], cmap='gray')
        if 'pytorch' in results:
            iou = calculate_iou(results['pytorch']['mask'], results['tensorrt_full']['mask'])
            dice = calculate_dice(results['pytorch']['mask'], results['tensorrt_full']['mask'])
            ax12.set_title(f'纯TensorRT Mask\nIoU: {iou*100:.2f}% | Dice: {dice*100:.2f}%', 
                          fontsize=12, color='green' if iou > 0.95 else 'orange')
        else:
            ax12.set_title(f'纯TensorRT Mask\n前景: {results["tensorrt_full"]["mask"].sum():,} 像素', fontsize=12)
        ax12.axis('off')
        
        # 纯TensorRT时序分布
        ax13 = plt.subplot(rows, cols, 15)
        ax13.hist(results['tensorrt_full']['encoder_times'], bins=20, alpha=0.7, label='TRT Encoder', color='blue')
        ax13.hist(results['tensorrt_full']['decoder_times'], bins=20, alpha=0.7, label='TRT Decoder', color='red')
        ax13.set_xlabel('时间 (ms)')
        ax13.set_ylabel('频次')
        ax13.set_title('纯TensorRT 时序分布', fontsize=12, weight='bold')
        ax13.legend()
        ax13.grid(True, alpha=0.3)
    
    # 性能对比柱状图
    ax_perf = plt.subplot(rows, cols, 8)
    methods = []
    encoder_means = []
    decoder_means = []
    total_means = []
    
    for method_name, method_key in [('PyTorch', 'pytorch'), ('ONNX', 'onnx'), 
                                     ('TRT混合', 'tensorrt'), ('纯TRT', 'tensorrt_full')]:
        if method_key in results:
            methods.append(method_name)
            encoder_means.append(results[method_key]['encoder_mean'])
            decoder_means.append(results[method_key]['decoder_mean'])
            total_means.append(results[method_key]['total_mean'])
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax_perf.bar(x - width, encoder_means, width, label='Encoder', color='steelblue')
    ax_perf.bar(x, decoder_means, width, label='Decoder', color='coral')
    ax_perf.bar(x + width, total_means, width, label='总耗时', color='mediumseagreen')
    
    ax_perf.set_ylabel('时间 (ms)', fontsize=11)
    ax_perf.set_title('性能对比', fontsize=12, weight='bold')
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(methods, fontsize=9, rotation=15)
    ax_perf.legend(fontsize=9)
    ax_perf.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (enc, dec, tot) in enumerate(zip(encoder_means, decoder_means, total_means)):
        ax_perf.text(i - width, enc + 2, f'{enc:.1f}', ha='center', va='bottom', fontsize=7)
        ax_perf.text(i, dec + 2, f'{dec:.1f}', ha='center', va='bottom', fontsize=7)
        ax_perf.text(i + width, tot + 2, f'{tot:.1f}', ha='center', va='bottom', fontsize=7)
    
    # 加速比对比
    ax_speedup = plt.subplot(rows, cols, 12)
    if 'pytorch' in results:
        baseline = results['pytorch']['total_mean']
        speedups = []
        speedup_labels = []
        colors_speedup = []
        
        for method_name, method_key in [('ONNX', 'onnx'), ('TRT混合', 'tensorrt'), ('纯TRT', 'tensorrt_full')]:
            if method_key in results:
                speedup = baseline / results[method_key]['total_mean']
                speedups.append(speedup)
                speedup_labels.append(method_name)
                colors_speedup.append('green' if speedup > 1 else 'red')
        
        bars = ax_speedup.barh(speedup_labels, speedups, color=colors_speedup, alpha=0.7)
        ax_speedup.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
        ax_speedup.set_xlabel('加速比', fontsize=11)
        ax_speedup.set_title('相对PyTorch的加速比', fontsize=12, weight='bold')
        ax_speedup.legend(fontsize=9)
        ax_speedup.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            ax_speedup.text(speedup + 0.05, i, f'{speedup:.2f}x', va='center', fontsize=10, weight='bold')
    
    plt.suptitle('SAM-HQ 模型性能与一致性对比\nPyTorch vs ONNX CUDA vs TensorRT混合 vs 纯TensorRT', 
                 fontsize=16, weight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存对比图: {output_path}")
    
    return output_path


def print_summary(results):
    """打印汇总信息"""
    print("\n" + "="*80)
    print("📈 性能与一致性汇总")
    print("="*80)
    
    # 性能表格
    print("\n性能对比:")
    print("-" * 80)
    print(f"{'方法':<20} {'Encoder (ms)':<20} {'Decoder (ms)':<20} {'总耗时 (ms)':<15} {'加速比':<10}")
    print("-" * 80)
    
    baseline = results['pytorch']['total_mean'] if 'pytorch' in results else None
    
    for method_name, method_key in [('PyTorch GPU', 'pytorch'), ('ONNX CUDA', 'onnx'), 
                                     ('TensorRT混合', 'tensorrt'), ('纯TensorRT', 'tensorrt_full')]:
        if method_key in results:
            r = results[method_key]
            speedup = f"{baseline / r['total_mean']:.2f}x" if baseline and baseline != r['total_mean'] else "-"
            print(f"{method_name:<20} {r['encoder_mean']:>6.2f} ± {r['encoder_std']:>5.2f}  "
                  f"{r['decoder_mean']:>6.2f} ± {r['decoder_std']:>5.2f}  "
                  f"{r['total_mean']:>10.2f}     {speedup:<10}")
    
    # 一致性表格
    if 'pytorch' in results:
        print("\n一致性验证 (相对PyTorch):")
        print("-" * 80)
        print(f"{'方法':<20} {'IoU (%)':<15} {'Dice (%)':<15} {'前景像素差异':<20}")
        print("-" * 80)
        
        pytorch_mask = results['pytorch']['mask']
        pytorch_fg = pytorch_mask.sum()
        
        for method_name, method_key in [('ONNX CUDA', 'onnx'), ('TensorRT混合', 'tensorrt'), 
                                         ('纯TensorRT', 'tensorrt_full')]:
            if method_key in results:
                mask = results[method_key]['mask']
                iou = calculate_iou(pytorch_mask, mask) * 100
                dice = calculate_dice(pytorch_mask, mask) * 100
                fg_diff = mask.sum() - pytorch_fg
                fg_diff_pct = (fg_diff / pytorch_fg) * 100
                
                status = "✅" if iou >= 95.0 else "⚠️"
                print(f"{method_name:<20} {iou:>7.2f} {status}      {dice:>7.2f}        "
                      f"{fg_diff:>+8,} ({fg_diff_pct:>+6.2f}%)")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='SAM-HQ模型完整对比测试')
    parser.add_argument('--model-type', type=str, default='vit_b', 
                       choices=['vit_tiny', 'vit_b', 'vit_l', 'vit_h'],
                       help='模型类型')
    parser.add_argument('--checkpoint', type=str, default='sam_hq_vit_b.pth', help='PyTorch checkpoint')
    parser.add_argument('--encoder-onnx', type=str, default='sam_hq_vit_b_encoder.onnx', help='ONNX encoder')
    parser.add_argument('--decoder-onnx', type=str, default='sam_hq_official_decoder.onnx', help='ONNX decoder')
    parser.add_argument('--encoder-trt', type=str, default='tensorrt_engines/sam_hq_vit_b_encoder_fp16.engine', 
                       help='TensorRT encoder engine')
    parser.add_argument('--decoder-trt', type=str, default='tensorrt_engines/sam_hq_official_decoder_fp16.engine', 
                       help='TensorRT decoder engine')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--points', type=str, required=True, 
                       help='输入点，格式: x1,y1,label1;x2,y2,label2 (label: 1=前景, 0=背景)')
    parser.add_argument('--runs', type=int, default=20, help='每个测试的运行次数')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    parser.add_argument('--output', type=str, default='model_comparison.jpg', help='输出对比图路径')
    parser.add_argument('--skip-pytorch', action='store_true', help='跳过PyTorch测试')
    parser.add_argument('--skip-onnx', action='store_true', help='跳过ONNX测试')
    parser.add_argument('--skip-tensorrt', action='store_true', help='跳过TensorRT测试')
    parser.add_argument('--skip-tensorrt-full', action='store_true', help='跳过纯TensorRT测试（仅测试混合模式）')
    
    args = parser.parse_args()
    
    # 自动检测vit_h的嵌套路径
    if args.model_type == 'vit_h':
        # 检查是否是嵌套目录结构
        nested_encoder = f'sam_hq_{args.model_type}_encoder-onnx/sam_hq_{args.model_type}_encoder.onnx'
        if os.path.exists(nested_encoder) and args.encoder_onnx == 'sam_hq_vit_b_encoder.onnx':
            args.encoder_onnx = nested_encoder
            print(f"✅ 检测到vit_h嵌套路径: {nested_encoder}")
    
    # 解析点
    points = []
    labels = []
    for point_str in args.points.split(';'):
        parts = point_str.strip().split(',')
        if len(parts) == 3:
            x, y, label = map(int, parts)
            points.append([x, y])
            labels.append(label)
    
    points = np.array(points)
    labels = np.array(labels)
    
    # 读取图像
    print("\n" + "="*80)
    print("🖼️  加载测试图像")
    print("="*80)
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ 无法读取图像: {args.image}")
        return
    
    print(f"  图像尺寸: {image.shape[:2]}")
    print(f"  输入点数: {len(points)}")
    print(f"  点坐标: {points.tolist()}")
    print(f"  点标签: {labels.tolist()} (1=前景, 0=背景)")
    
    # 存储结果
    results = {}
    
    # 测试PyTorch
    if not args.skip_pytorch and os.path.exists(args.checkpoint):
        try:
            results['pytorch'] = benchmark_pytorch(args.checkpoint, image, points, labels, 
                                                   args.model_type, args.runs, args.warmup)
        except Exception as e:
            print(f"❌ PyTorch测试失败: {e}")
    elif args.skip_pytorch:
        print("\n⏭️  跳过PyTorch测试")
    
    # 测试ONNX CUDA
    if not args.skip_onnx and os.path.exists(args.encoder_onnx) and os.path.exists(args.decoder_onnx):
        try:
            results['onnx'] = benchmark_onnx_cuda(args.encoder_onnx, args.decoder_onnx, 
                                                  image, points, labels, args.runs, args.warmup)
        except Exception as e:
            print(f"❌ ONNX测试失败: {e}")
    elif args.skip_onnx:
        print("\n⏭️  跳过ONNX测试")
    
    # 测试TensorRT混合模式 (TensorRT Encoder + ONNX Decoder)
    if not args.skip_tensorrt and os.path.exists(args.encoder_trt) and os.path.exists(args.decoder_onnx):
        try:
            results['tensorrt'] = benchmark_tensorrt(args.encoder_trt, args.decoder_onnx, 
                                                     image, points, labels, args.runs, args.warmup)
        except Exception as e:
            print(f"❌ TensorRT混合模式测试失败: {e}")
            import traceback
            traceback.print_exc()
    elif args.skip_tensorrt:
        print("\n⏭️  跳过TensorRT混合模式测试")
    
    # 测试纯TensorRT (TensorRT Encoder + TensorRT Decoder)
    if (not args.skip_tensorrt and not args.skip_tensorrt_full and 
        os.path.exists(args.encoder_trt) and os.path.exists(args.decoder_trt)):
        try:
            print("\n提示: 纯TensorRT模式需要正确处理动态输出shape")
            results['tensorrt_full'] = benchmark_tensorrt_full(args.encoder_trt, args.decoder_trt, 
                                                               image, points, labels, args.runs, args.warmup)
        except Exception as e:
            print(f"❌ 纯TensorRT测试失败: {e}")
            import traceback
            traceback.print_exc()
    elif args.skip_tensorrt or args.skip_tensorrt_full:
        print("\n⏭️  跳过纯TensorRT测试")
    
    # 生成可视化
    if results:
        create_visualization(image, points, labels, results, args.output)
        print_summary(results)
    else:
        print("\n❌ 没有成功的测试结果")
    
    print("\n✅ 测试完成！")


if __name__ == '__main__':
    main()
