#!/usr/bin/env python3
"""
SAM 分割服务 API

支持 TensorRT 引擎推理和多种 prompt 模式
- 全图分割
- 点 prompt
- 框 prompt
"""

import sys
sys.path.insert(0, '/media/user/Disk2/gjt/segment-anything')

import io
import os
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ONNX Runtime 导入
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: ONNX Runtime 不可用")

# TensorRT 导入（不再需要 PyCUDA，使用 PyTorch 的 CUDA 管理）
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("警告: TensorRT 不可用，将使用 PyTorch")

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import (
    MaskData,
    generate_crop_boxes,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
    calculate_stability_score,
    rle_to_mask,
    batched_mask_to_box,
    mask_to_rle_pytorch,
    is_box_near_crop_edge,
    batch_iterator,
    remove_small_regions,
)

app = Flask(__name__, static_folder='.')
CORS(app)

# 模型配置 - SAM-HQ 混合模式：TensorRT Encoder + ONNX Decoder
MODEL_CONFIGS = {
    "vit_tiny": {
        "type": "vit_tiny",
        "checkpoint": "sam_hq_vit_tiny.pth",
        "encoder_engine": "/app/models/sam_hq_vit_tiny_encoder_bf16.engine",
        "decoder_onnx": "/app/models/sam_hq_vit_tiny_decoder.onnx",
        "precision": "BF16",
        "description": "SAM-HQ ViT-Tiny BF16 - 最快，适合实时应用",
    },
    "vit_b": {
        "type": "vit_b",
        "checkpoint": "sam_hq_vit_b.pth",
        "encoder_engine": "/app/models/sam_hq_vit_b_encoder_fp16.engine",
        "decoder_onnx": "/app/models/sam_hq_vit_b_decoder.onnx",
        "precision": "FP16",
        "description": "SAM-HQ ViT-B FP16 - 平衡速度与精度",
    },
    "vit_l": {
        "type": "vit_l",
        "checkpoint": "/app/models/sam_hq_vit_l.pth",
        "encoder_engine": "/app/models/tensorrt_engines/sam_hq_vit_l_encoder_fp16.engine",
        "decoder_onnx": "/app/models/sam_hq_vit_l_decoder.onnx",
        "precision": "FP16",
        "description": "SAM-HQ ViT-L FP16 - 高精度分割",
    },
    "vit_h": {
        "type": "vit_h",
        "checkpoint": "/app/models/sam_hq_vit_h.pth",
        "encoder_engine": "/app/models//sam_hq_vit_h_encoder_bf16.engine",
        "decoder_onnx": "/app/models/sam_hq_vit_h_decoder.onnx",
        "precision": "BF16",
        "description": "SAM-HQ ViT-H BF16 - 最高精度",
    },
}

# 全局模型缓存
loaded_models = {}


class HybridSAMPredictor:
    """SAM-HQ 混合预测器：TensorRT Encoder + ONNX Decoder
    
    这个预测器结合了 TensorRT 加速的 image encoder 和 ONNX Runtime 的 decoder，
    提供最佳的性能和精度平衡。
    """
    
    def __init__(self, encoder_engine_path: str, decoder_onnx_path: str, model_type: str):
        """初始化混合预测器
        
        Args:
            encoder_engine_path: TensorRT encoder 引擎路径
            decoder_onnx_path: ONNX decoder 模型路径  
            model_type: 模型类型 (vit_tiny, vit_b, vit_l, vit_h)
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT 不可用！")
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime 不可用！")
        
        self.model_type = model_type
        
        # 加载 TensorRT encoder
        print(f"  ⚡ 加载 TensorRT encoder: {encoder_engine_path}")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(encoder_engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"无法加载 TensorRT 引擎: {encoder_engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.input_name = self.engine.get_tensor_name(0)
        self.output0_name = self.engine.get_tensor_name(1)  # image_embeddings
        self.output1_name = self.engine.get_tensor_name(2)  # interm_embeddings
        
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output0_shape = tuple(self.engine.get_tensor_shape(self.output0_name))
        self.output1_shape = tuple(self.engine.get_tensor_shape(self.output1_name))
        
        print(f"    输入 '{self.input_name}': {self.input_shape}")
        print(f"    输出0 '{self.output0_name}': {self.output0_shape}")
        print(f"    输出1 '{self.output1_name}': {self.output1_shape}")
        
        # 使用 PyTorch 管理 CUDA 内存（而不是 PyCUDA）
        self.d_input = torch.empty(self.input_shape, dtype=torch.float32, device='cuda')
        self.d_output0 = torch.empty(self.output0_shape, dtype=torch.float32, device='cuda')
        self.d_output1 = torch.empty(self.output1_shape, dtype=torch.float32, device='cuda')
        
        # 绑定内存地址
        self.context.set_tensor_address(self.input_name, self.d_input.data_ptr())
        self.context.set_tensor_address(self.output0_name, self.d_output0.data_ptr())
        self.context.set_tensor_address(self.output1_name, self.d_output1.data_ptr())
        
        # 创建 PyTorch CUDA stream
        self.stream = torch.cuda.Stream()
        
        # 执行一次推理以建立 CUDA 上下文
        with torch.cuda.stream(self.stream):
            dummy_input = torch.zeros(self.input_shape, dtype=torch.float32, device='cuda')
            self.d_input.copy_(dummy_input)
            success = self.context.execute_async_v3(self.stream.cuda_stream)
            if not success:
                print("    警告: 初始化推理失败")
        torch.cuda.synchronize()
        
        # 加载 ONNX decoder（使用 CUDA provider）
        print(f"  🔧 加载 ONNX decoder: {decoder_onnx_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.decoder_session = ort.InferenceSession(
            decoder_onnx_path,
            sess_options=sess_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 检查 decoder 输入名称
        decoder_inputs = {inp.name: inp for inp in self.decoder_session.get_inputs()}
        self.interm_name = 'interm_features' if 'interm_features' in decoder_inputs else 'interm_embeddings'
        print(f"    Decoder 中间特征名称: {self.interm_name}")
        
        # SAM 预处理参数
        self.pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
        self.pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 3)
        self.img_size = 1024
        
        # 保存原始图像信息（用于坐标转换）
        self.original_size = None
        self.input_size = None
        self.scale = None
        self.new_h = None
        self.new_w = None
        
        # ✅ 优化：缓存的 encoder 输出（保持在 GPU）
        self.image_embeddings_gpu = None
        self.interm_embeddings_gpu = None
        
        # 兼容性：也保留 CPU 版本（用于不支持 IOBinding 的情况）
        self.image_embeddings = None
        self.interm_embeddings = None
        self.use_iobinding = True  # 默认启用 IOBinding 优化
    
    def set_image(self, image: np.ndarray):
        """设置图像并运行 encoder
        
        Args:
            image: RGB 图像，numpy array (H, W, 3)
        """
        self.original_size = (image.shape[0], image.shape[1])  # (height, width)
        
        # 预处理图像
        h, w = image.shape[:2]
        self.scale = self.img_size / max(h, w)
        self.new_h, self.new_w = int(h * self.scale), int(w * self.scale)
        
        resized = cv2.resize(image, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        padded[:self.new_h, :self.new_w] = resized
        
        normalized = (padded - self.pixel_mean) / self.pixel_std
        input_tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32))
        
        self.input_size = (self.img_size, self.img_size)
        
        # TensorRT encoder 推理（使用 PyTorch CUDA stream）
        with torch.cuda.stream(self.stream):
            # 将 numpy 数据复制到 GPU
            input_torch = torch.from_numpy(input_tensor).cuda()
            self.d_input.copy_(input_torch)
            
            # 执行 TensorRT 推理
            success = self.context.execute_async_v3(self.stream.cuda_stream)
            if not success:
                raise RuntimeError("TensorRT encoder 推理失败")
        
        # 同步并获取结果
        torch.cuda.synchronize()
        
        # ✅ 优化：保持数据在 GPU 上（torch tensor），避免 GPU→CPU→GPU 的往返
        self.image_embeddings_gpu = self.d_output0.clone()
        self.interm_embeddings_gpu = self.d_output1.clone()
        
        # ✅ 优化：在 GPU 上进行维度修复（比 CPU 快 50-100ms）
        if self.interm_name == 'interm_embeddings' and len(self.interm_embeddings_gpu.shape) == 4:
            self.interm_embeddings_gpu = self.interm_embeddings_gpu.unsqueeze(0)
            self.interm_embeddings_gpu = self.interm_embeddings_gpu.repeat(4, 1, 1, 1, 1)
            self.interm_embeddings_gpu = self.interm_embeddings_gpu.contiguous()
        
        # 兼容性：如果不使用 IOBinding，才传输到 CPU
        if not self.use_iobinding:
            self.image_embeddings = self.image_embeddings_gpu.cpu().numpy()
            self.interm_embeddings = self.interm_embeddings_gpu.cpu().numpy()
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ):
        """使用 prompts 进行预测
        
        Args:
            point_coords: 点坐标 (N, 2)，原始图像空间
            point_labels: 点标签 (N,)
            box: 边界框 (4,) [x1, y1, x2, y2]，原始图像空间
            mask_input: mask 输入 (1, 256, 256)
            multimask_output: 是否输出多个 mask
            return_logits: 是否返回 logits
        
        Returns:
            masks: (N, H, W) numpy array
            scores: (N,) numpy array
            logits: (N, 256, 256) numpy array
        """
        if self.image_embeddings_gpu is None:
            raise RuntimeError("请先调用 set_image()")
        
        # 转换坐标到 1024 空间
        if point_coords is not None:
            point_coords = point_coords * self.scale
            point_coords = point_coords.reshape(1, -1, 2).astype(np.float32)
            point_labels = point_labels.reshape(1, -1).astype(np.float32)
        
        if box is not None:
            box = box * self.scale
            # 转换为 point prompt 格式（SAM decoder 的标准格式）
            box_coords = np.array([[box[0], box[1]], [box[2], box[3]]])
            box_labels = np.array([2, 3])  # 2=左上角，3=右下角
            
            if point_coords is not None:
                point_coords = np.concatenate([point_coords[0], box_coords], axis=0).reshape(1, -1, 2)
                point_labels = np.concatenate([point_labels[0], box_labels], axis=0).reshape(1, -1)
            else:
                point_coords = box_coords.reshape(1, -1, 2).astype(np.float32)
                point_labels = box_labels.reshape(1, -1).astype(np.float32)
        
        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0], dtype=np.float32)
        else:
            has_mask_input = np.array([1], dtype=np.float32)
        
        orig_im_size = np.array(self.original_size, dtype=np.float32)
        
        # ✅ 优化：使用 IOBinding 进行零拷贝推理（节省 200-400ms）
        if self.use_iobinding:
            try:
                outputs = self._predict_with_iobinding(
                    point_coords, point_labels, mask_input, has_mask_input, orig_im_size
                )
            except Exception as e:
                print(f"⚠️ IOBinding 失败，回退到标准模式: {e}")
                self.use_iobinding = False
                outputs = self._predict_standard(
                    point_coords, point_labels, mask_input, has_mask_input, orig_im_size
                )
        else:
            outputs = self._predict_standard(
                point_coords, point_labels, mask_input, has_mask_input, orig_im_size
            )
        
        masks = outputs[0]  # (1, num_masks, H, W)
        iou_predictions = outputs[1]  # (1, num_masks)
        low_res_masks = outputs[2]  # (1, num_masks, 256, 256)
        
        # 提取结果
        masks = masks[0]  # (num_masks, H, W)
        scores = iou_predictions[0]  # (num_masks,)
        logits = low_res_masks[0]  # (num_masks, 256, 256)
        
        # 转换为 boolean mask
        masks = masks > 0
        
        if return_logits:
            return masks, scores, logits
        else:
            return masks, scores, None
    
    def _predict_with_iobinding(self, point_coords, point_labels, mask_input, has_mask_input, orig_im_size):
        """使用 IOBinding 进行零拷贝推理（优化版）
        
        通过 ONNX Runtime IOBinding，直接使用 GPU 上的 encoder 输出，
        避免 GPU→CPU→GPU 的往返传输，节省 200-400ms。
        """
        io_binding = self.decoder_session.io_binding()
        
        # ✅ 绑定 GPU 输入（大数据：image_embeddings 和 interm_embeddings）
        io_binding.bind_input(
            name='image_embeddings',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(self.image_embeddings_gpu.shape),
            buffer_ptr=self.image_embeddings_gpu.data_ptr()
        )
        
        io_binding.bind_input(
            name=self.interm_name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(self.interm_embeddings_gpu.shape),
            buffer_ptr=self.interm_embeddings_gpu.data_ptr()
        )
        
        # 小数据可以用 CPU 输入（传输开销可忽略）
        io_binding.bind_cpu_input('point_coords', point_coords)
        io_binding.bind_cpu_input('point_labels', point_labels)
        io_binding.bind_cpu_input('mask_input', mask_input)
        io_binding.bind_cpu_input('has_mask_input', has_mask_input)
        io_binding.bind_cpu_input('orig_im_size', orig_im_size)
        
        # 绑定输出（自动分配 GPU 内存）
        io_binding.bind_output('masks')
        io_binding.bind_output('iou_predictions')
        io_binding.bind_output('low_res_masks')
        
        # ✅ 执行推理（全部在 GPU，零拷贝）
        self.decoder_session.run_with_iobinding(io_binding)
        
        # 获取输出（从 GPU 传输到 CPU）
        return io_binding.copy_outputs_to_cpu()
    
    def _predict_standard(self, point_coords, point_labels, mask_input, has_mask_input, orig_im_size):
        """标准推理模式（回退方案）
        
        当 IOBinding 不可用或失败时使用。
        会将 GPU tensor 转换为 numpy 后传递给 ONNX decoder。
        """
        # 确保有 CPU 版本的数据
        if self.image_embeddings is None:
            self.image_embeddings = self.image_embeddings_gpu.cpu().numpy()
            self.interm_embeddings = self.interm_embeddings_gpu.cpu().numpy()
        
        decoder_inputs = {
            'image_embeddings': self.image_embeddings,
            self.interm_name: self.interm_embeddings,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'mask_input': mask_input,
            'has_mask_input': has_mask_input,
            'orig_im_size': orig_im_size
        }
        
        return self.decoder_session.run(None, decoder_inputs)


def get_gpu_memory_mb():
    """获取 SAM service 的 GPU 显存占用 (MB)
    
    返回当前进程在 GPU 上分配的显存总量。
    包括 PyTorch 模型权重、TensorRT 引擎、激活值等。
    """
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        # 方法1：尝试使用 pynvml (最准确 - 统计进程实际使用)
        try:
            import pynvml
            if not hasattr(get_gpu_memory_mb, '_pynvml_initialized'):
                pynvml.nvmlInit()
                get_gpu_memory_mb._pynvml_initialized = True
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # 获取当前进程的 GPU 内存使用
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            current_pid = os.getpid()
            
            for proc in processes:
                if proc.pid == current_pid:
                    return proc.usedGpuMemory / (1024 ** 2)
            
            # 如果没找到当前进程，fallback
            return torch.cuda.memory_reserved() / (1024 ** 2)
            
        except (ImportError, Exception):
            # pynvml 不可用，使用 PyTorch 的统计
            # memory_reserved 包含 PyTorch 内存池保留的所有显存
            # 比 memory_allocated 更接近实际使用（包含缓存）
            return torch.cuda.memory_reserved() / (1024 ** 2)
            
    except Exception as e:
        print(f"警告: 获取 GPU 内存失败: {e}")
        return 0.0


def load_model(model_key: str, use_tensorrt: bool = True) -> HybridSAMPredictor:
    """加载模型 - SAM-HQ 混合模式：TensorRT Encoder + ONNX Decoder
    
    Args:
        model_key: 模型键名 (vit_tiny, vit_b, vit_l, vit_h)
        use_tensorrt: 是否使用 TensorRT（当前版本总是使用）
    
    Returns:
        HybridSAMPredictor 实例
    """
    
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    config = MODEL_CONFIGS[model_key]
    model_type = config["type"]
    encoder_engine_path = Path(config["encoder_engine"])
    decoder_onnx_path = Path(config["decoder_onnx"])
    
    # 检查必需的文件
    if not encoder_engine_path.exists():
        raise FileNotFoundError(
            f"❌ TensorRT encoder 引擎不存在: {encoder_engine_path}\n"
            f"   请运行: python rebuild_vit_{model_type}_trt.py"
        )
    
    if not decoder_onnx_path.exists():
        raise FileNotFoundError(
            f"❌ ONNX decoder 不存在: {decoder_onnx_path}\n"
            f"   请运行 ONNX 导出脚本"
        )
    
    if not TRT_AVAILABLE:
        raise RuntimeError("❌ TensorRT 不可用！请安装: pip install tensorrt")
    
    if not ONNX_AVAILABLE:
        raise RuntimeError("❌ ONNX Runtime 不可用！请安装: pip install onnxruntime-gpu")
    
    print(f"\n🔧 加载模型: {model_key} (SAM-HQ 混合模式)")
    print(f"  ⚡ TensorRT Encoder: {encoder_engine_path.name} ({config['precision']})")
    print(f"  🔧 ONNX Decoder: {decoder_onnx_path.name}")
    
    # 创建混合预测器
    predictor = HybridSAMPredictor(
        encoder_engine_path=str(encoder_engine_path),
        decoder_onnx_path=str(decoder_onnx_path),
        model_type=model_type
    )
    
    print(f"  ✓ 模型加载成功")
    
    loaded_models[model_key] = predictor
    
    return predictor


def preload_models():
    """预加载所有可用的模型到显存 - SAM-HQ 优化版本
    
    按顺序加载：vit_tiny -> vit_b -> vit_l -> vit_h
    如果显存不足，可以选择只加载部分模型。
    """
    print("\n" + "="*80)
    print("🚀 开始预加载模型（SAM-HQ TensorRT + ONNX）")
    print("="*80)
    
    # 推荐的加载顺序：从小到大
    load_order = ['vit_tiny', 'vit_b', 'vit_l', 'vit_h']
    
    for model_key in load_order:
        if model_key not in MODEL_CONFIGS:
            continue
            
        config = MODEL_CONFIGS[model_key]
        encoder_engine_path = Path(config["encoder_engine"])
        decoder_onnx_path = Path(config["decoder_onnx"])
        
        # 检查必需的文件
        missing_files = []
        if not encoder_engine_path.exists():
            missing_files.append(f"TensorRT Engine: {encoder_engine_path}")
        if not decoder_onnx_path.exists():
            missing_files.append(f"ONNX Decoder: {decoder_onnx_path}")
        
        if missing_files:
            print(f"\n⚠ 跳过 {model_key}: 缺少文件")
            for f in missing_files:
                print(f"   ✗ {f}")
            if not encoder_engine_path.exists():
                print(f"   转换 TensorRT: python rebuild_vit_{config['type']}_trt.py")
            if not decoder_onnx_path.exists():
                print(f"   导出 ONNX: 运行 decoder 导出脚本")
            continue
        
        try:
            mem_before = get_gpu_memory_mb()
            
            # 使用 TensorRT + ONNX 混合模式
            if not TRT_AVAILABLE or not ONNX_AVAILABLE:
                print(f"\n✗ TensorRT 或 ONNX Runtime 不可用！")
                if not TRT_AVAILABLE:
                    print(f"   请安装 TensorRT: pip install tensorrt")
                if not ONNX_AVAILABLE:
                    print(f"   请安装 ONNX Runtime: pip install onnxruntime-gpu")
                continue
                
            print(f"\n📦 加载 {model_key} ({config['precision']}) - TensorRT + ONNX")
            load_model(model_key, use_tensorrt=True)
            mem_after = get_gpu_memory_mb()
            mem_used = mem_after - mem_before
            print(f"   ✓ 加载成功 | 显存使用: +{mem_used:.1f} MB (总计: {mem_after:.1f} MB)")
                
        except Exception as e:
            print(f"\n✗ {model_key} 加载失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"   继续加载其他模型...")
    
    print("\n" + "="*80)
    if len(loaded_models) > 0:
        print(f"✓ 预加载完成！成功加载 {len(loaded_models)} 个模型")
        print(f"📊 可用模型: {list(loaded_models.keys())}")
        final_mem = get_gpu_memory_mb()
        print(f"💾 总显存使用: {final_mem:.1f} MB")
    else:
        print(f"⚠ 警告: 没有成功加载任何模型！")
        print(f"   请检查模型文件是否存在")
    print("="*80 + "\n")
    
    return len(loaded_models)


def decode_base64_image(base64_str: str) -> np.ndarray:
    """解码 base64 图片"""
    img_data = base64.b64decode(base64_str.split(',')[1] if ',' in base64_str else base64_str)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img.convert('RGB'))


def resize_image_if_needed(image: np.ndarray, max_size: int = 1024) -> Tuple[np.ndarray, float]:
    """如果图像长边大于 max_size，则缩放图像，保持长宽比
    
    Args:
        image: 输入图像 (H, W, C)
        max_size: 最大尺寸
    
    Returns:
        resized_image: 缩放后的图像
        scale: 缩放比例（用于将坐标和mask缩放回原始尺寸）
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim <= max_size:
        # 不需要缩放
        return image, 1.0
    
    # 计算缩放比例
    scale = max_size / max_dim
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # 使用高质量插值缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    print(f"图像缩放: ({w}x{h}) -> ({new_w}x{new_h}), 比例: {scale:.3f}")
    
    return resized, scale


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """将 mask 编码为 base64"""
    # 确保 mask 是 uint8 类型
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # 编码为 PNG
    _, buffer = cv2.imencode('.png', mask_uint8)
    
    # 转换为 base64
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{mask_base64}"


def mask_to_rle(mask: np.ndarray) -> Dict:
    """将 mask 转换为 RLE (Run-Length Encoding) 格式
    
    RLE 格式大幅减小数据量，适合网络传输。
    格式: {"size": [h, w], "counts": [len1, len2, ...]}
    
    ⚡ 优化版本：使用 NumPy 向量化操作，比 Python 循环快 100x
    对于 12MP 图像（3024×4032），从 7000ms 降低到 ~30ms
    """
    # 确保是 boolean mask
    mask = mask.astype(bool)
    height, width = mask.shape
    
    # 展平为 1D（按列优先，与 COCO 格式一致）
    pixels = mask.T.flatten()
    pixels_uint8 = pixels.astype(np.uint8)
    
    # ⚡ 向量化 RLE 编码
    # RLE 规则：counts = [0的长度, 1的长度, 0的长度, 1的长度, ...]
    # 如果第一个像素是 1，则第一个 count 是 0（表示没有背景）
    
    # 在前后添加哨兵值（0），这样可以统一处理边界情况
    padded = np.concatenate([[0], pixels_uint8, [0]])
    
    # 找到所有值变化的位置（0→1 或 1→0）
    diffs = np.diff(padded.astype(np.int16))
    change_indices = np.where(diffs != 0)[0]
    
    # 计算每段的长度
    counts = np.diff(change_indices).tolist()
    
    # 如果第一个像素是 1，需要在开头插入 0（表示 0 长度的背景）
    if len(pixels_uint8) > 0 and pixels_uint8[0] == 1:
        counts.insert(0, 0)
    
    return {
        "size": [int(height), int(width)],
        "counts": counts
    }


def calculate_bbox(mask: np.ndarray) -> List[int]:
    """计算 mask 的边界框 [x, y, width, height]"""
    # 找到所有非零点
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return [
        int(cmin),
        int(rmin),
        int(cmax - cmin + 1),
        int(rmax - rmin + 1)
    ]


def visualize_masks(image: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> tuple:
    """可视化 masks - 每个 mask 单独进行 alpha 混合，返回图像和颜色列表"""
    h, w = image.shape[:2]
    
    # 从 float32 开始，保持精度
    overlay = image.astype(np.float32)
    
    # 存储每个 mask 的颜色
    colors = []
    
    # 为每个 mask 单独进行 alpha 混合
    for i, mask in enumerate(masks):
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 生成随机颜色（先生成 uint8，再转为 float32）
        color = np.random.randint(50, 255, 3, dtype=np.uint8).astype(np.float32)
        colors.append([int(color[0]), int(color[1]), int(color[2])])  # 保存为整数列表
        
        # 创建当前 mask 的彩色层
        colored_mask = np.zeros_like(overlay, dtype=np.float32)
        mask_bool = mask > 0
        for c in range(3):
            colored_mask[:, :, c][mask_bool] = color[c]
        
        # 对当前 overlay 应用这个 mask：overlay = overlay * (1 - alpha * mask) + colored_mask * alpha * mask
        # 这样每个像素点只在 mask 区域内进行混合
        mask_float = mask_bool.astype(np.float32)
        overlay = overlay * (1 - alpha * mask_float[:, :, np.newaxis]) + colored_mask * alpha * mask_float[:, :, np.newaxis]
    
    return overlay.astype(np.uint8), colors


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "tensorrt_available": TRT_AVAILABLE,
        "models": list(MODEL_CONFIGS.keys())
    })


@app.route('/api/segment', methods=['POST'])
def segment():
    """分割接口"""
    try:
        data = request.json
        print(f"\n=== 收到分割请求 ===")
        print(f"模型: {data.get('model', 'vit_b')}")
        print(f"提示类型: {data.get('prompt_type', 'auto')}")
        print(f"使用 TensorRT: {data.get('use_tensorrt', True)}")
        
        # 解析参数
        model_key = data.get('model', 'vit_b')  # 默认使用 vit_b（最快）
        image_base64 = data.get('image')
        prompt_type = data.get('prompt_type', 'auto')  # auto, point, box
        points = data.get('points', [])  # [[x, y, label], ...]
        boxes = data.get('boxes', [])  # [[x1, y1, x2, y2], ...]
        use_tensorrt = data.get('use_tensorrt', True)
        alpha = data.get('alpha', 0.5)
        
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400
        
        if model_key not in MODEL_CONFIGS:
            return jsonify({"error": f"Invalid model: {model_key}"}), 400
        
        # 记录开始时间
        start_time = time.time()
        mem_before = get_gpu_memory_mb()
        
        # 解码图片
        original_image = decode_base64_image(image_base64)
        orig_h, orig_w = original_image.shape[:2]
        print(f"原始图像尺寸: {orig_w}x{orig_h}")
        
        # ⚠️ 重要: SamPredictor.set_image() 会自动处理图像尺寸
        # 它会将图像 resize 到长边 1024 并 pad 到 1024x1024
        # 因此我们直接使用原始图像，不需要手动缩放
        # 坐标也使用原始图像的坐标系，SamPredictor 会自动转换
        
        # 对于超大图像（如 4K+），可以选择性地预缩放以节省内存
        # 但通常不需要，因为 SAM 会处理
        MAX_INPUT_SIZE = 2048  # 可选的最大输入尺寸限制
        
        if max(orig_h, orig_w) > MAX_INPUT_SIZE:
            # 只对超大图像进行预缩放
            scale = MAX_INPUT_SIZE / max(orig_h, orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"图像预缩放: ({orig_w}x{orig_h}) -> ({new_w}x{new_h}), 比例: {scale:.3f}")
        else:
            image = original_image
            scale = 1.0
            new_h, new_w = orig_h, orig_w
        
        # 缩放坐标（仅当图像被预缩放时）
        if scale != 1.0:
            if points:
                points = [[p[0] * scale, p[1] * scale] + (p[2:] if len(p) > 2 else [1]) 
                         for p in points]
                print(f"点坐标预缩放: {len(points)} 个点，比例: {scale:.3f}")
            
            if boxes:
                boxes = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] 
                        for b in boxes]
                print(f"框坐标预缩放: {len(boxes)} 个框，比例: {scale:.3f}")
        
        # 使用预加载的模型（不再动态加载）
        if model_key not in loaded_models:
            return jsonify({
                "error": f"模型 {model_key} 未加载。可用模型: {list(loaded_models.keys())}"
            }), 400
        
        predictor = loaded_models[model_key]
        
        # 设置图片（SamPredictor 会自动处理尺寸转换）
        # ⚠️ 重要：必须同步 GPU 以获得准确计时
        torch.cuda.synchronize()
        encode_start = time.time()
        predictor.set_image(image)
        torch.cuda.synchronize()  # 等待 encoder 完成
        encode_time = time.time() - encode_start
        
        print(f"图像编码完成，predictor.original_size: {predictor.original_size}, predictor.input_size: {predictor.input_size}")
        
        # 根据 prompt 类型进行分割
        torch.cuda.synchronize()  # 确保 encoder 完成
        predict_start = time.time()
        
        if prompt_type == 'auto':
            # 自动分割全图
            from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
            
            # 优化参数以提升速度
            mask_generator = SamAutomaticMaskGenerator(
                model=predictor.model,
                points_per_side=16,  # 从 32 降低到 16，减少 75% 的采样点
                pred_iou_thresh=0.88,  # 提高阈值，过滤低质量 mask
                stability_score_thresh=0.95,  # 提高稳定性阈值
                crop_n_layers=0,  # 禁用裁剪层以加速
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )
            
            print(f"开始自动分割，采样点: 16x16 = 256...")
            masks_data = mask_generator.generate(image)
            print(f"生成了 {len(masks_data)} 个 mask")
            
            masks = [m['segmentation'] for m in masks_data]
            masks = np.array(masks)
            
            result_info = {
                "num_masks": len(masks),
                "mode": "automatic"
            }
            
        elif prompt_type == 'point':
            # 点 prompt（使用原始坐标，SamPredictor 会自动转换）
            if not points:
                return jsonify({"error": "No points provided"}), 400
            
            points_array = np.array([[p[0], p[1]] for p in points])
            labels_array = np.array([p[2] if len(p) > 2 else 1 for p in points])
            
            print(f"点坐标 (原始图像空间): {points_array.tolist()}")
            
            masks_result, scores, logits = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True,
            )
            
            print(f"预测完成，生成了 {len(masks_result)} 个 mask，尺寸: {masks_result.shape}")
            
            # 选择最佳 mask
            best_idx = np.argmax(scores)
            masks = masks_result[[best_idx]]
            
            result_info = {
                "num_masks": len(masks),
                "mode": "point",
                "scores": scores.tolist(),
                "best_score": float(scores[best_idx])
            }
            
        elif prompt_type == 'box':
            # 框 prompt（使用原始坐标，SamPredictor 会自动转换）
            if not boxes:
                return jsonify({"error": "No boxes provided"}), 400
            
            all_masks = []
            all_scores = []
            
            for box in boxes:
                box_array = np.array(box)
                
                print(f"框坐标 (原始图像空间): {box_array.tolist()}")
                
                masks_result, scores, logits = predictor.predict(
                    box=box_array,
                    multimask_output=False,
                )
                
                all_masks.append(masks_result[0])
                all_scores.append(scores[0])
            
            masks = np.array(all_masks)
            
            result_info = {
                "num_masks": len(masks),
                "mode": "box",
                "scores": [float(s) for s in all_scores]
            }
        else:
            return jsonify({"error": f"Invalid prompt_type: {prompt_type}"}), 400
        
        torch.cuda.synchronize()  # 等待 decoder 完成
        predict_time = time.time() - predict_start
        
        print(f"预测完成，mask 尺寸: {masks.shape}")
        
        # SamPredictor 返回的 mask 已经是输入图像的尺寸
        # 如果图像被预缩放过，需要将 mask 缩放回原始尺寸
        if scale != 1.0:
            print(f"将 mask 缩放回原始尺寸: ({new_w}x{new_h}) -> ({orig_w}x{orig_h})")
            resized_masks = []
            for mask in masks:
                # 使用最近邻插值保持 mask 的二值特性
                resized_mask = cv2.resize(
                    mask.astype(np.uint8), 
                    (orig_w, orig_h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                resized_masks.append(resized_mask)
            masks = np.array(resized_masks)
            
            # 用于可视化的图像也使用原始尺寸
            vis_image = original_image
        else:
            # 没有预缩放，mask 已经是原始尺寸
            vis_image = original_image
        
        print(f"最终 mask 尺寸: {masks.shape}, 图像尺寸: {vis_image.shape[:2]}")
        
        # ✅ 优化方案 2：只返回 mask 图像的 Base64（不返回 overlay）
        # 使用更快的图像格式（PNG 压缩级别 1）+ 缓存优化
        encode_start = time.time()
        
        # 生成随机颜色（用于前端渲染）
        colors = []
        for i in range(len(masks)):
            color = np.random.randint(50, 255, 3, dtype=np.uint8)
            colors.append([int(color[0]), int(color[1]), int(color[2])])
        
        # 为每个 mask 生成 Base64 编码（黑白图像）
        masks_data = []
        for i, mask in enumerate(masks):
            # 转换为 0/255 的灰度图
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # 编码为 PNG（压缩级别 1 = 快速）
            success, buffer = cv2.imencode('.png', mask_uint8, 
                                          [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not success:
                raise ValueError(f"Failed to encode mask {i}")
            
            mask_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 计算边界框和面积
            bbox = calculate_bbox(mask)
            area = int(mask.sum())
            
            masks_data.append({
                "mask": f"data:image/png;base64,{mask_base64}",
                "bbox": bbox,
                "area": area,
                "color": colors[i]
            })
        
        encode_time_total = time.time() - encode_start
        
        # 统计信息
        total_time = time.time() - start_time
        mem_after = get_gpu_memory_mb()
        
        print(f"✅ Mask 编码完成: {len(masks_data)} 个 mask, 耗时: {encode_time_total*1000:.2f}ms")
        
        return jsonify({
            "success": True,
            "result": {
                "masks": masks_data,
                "info": result_info
            },
            "image_info": {
                "original_width": orig_w,
                "original_height": orig_h,
                "processed_width": new_w,
                "processed_height": new_h,
                "predictor_original_size": predictor.original_size,
                "predictor_input_size": predictor.input_size,
                "pre_scale": scale,
                "was_pre_scaled": scale != 1.0
            },
            "performance": {
                "total_time_ms": round(total_time * 1000, 2),
                "image_encode_ms": round(encode_time * 1000, 2),
                "predict_time_ms": round(predict_time * 1000, 2),
                "mask_encode_ms": round(encode_time_total * 1000, 2),
                "memory_mb": round(mem_after, 2),
                "memory_delta_mb": round(mem_after - mem_before, 2),
                "optimization": "mask_only_base64 (fast PNG encoding)"
            },
            "model_info": {
                "model": model_key,
                "config": MODEL_CONFIGS[model_key],
                "tensorrt_used": use_tensorrt and TRT_AVAILABLE
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    models_info = []
    
    for key, config in MODEL_CONFIGS.items():
        encoder_engine_path = Path(config["encoder_engine"])
        decoder_onnx_path = Path(config["decoder_onnx"])
        
        # 检查模型是否已加载
        is_loaded = key in loaded_models
        
        # 检查文件是否都存在
        encoder_exists = encoder_engine_path.exists()
        decoder_exists = decoder_onnx_path.exists()
        all_files_exist = encoder_exists and decoder_exists
        
        models_info.append({
            "key": key,
            "name": f"{config['type'].upper()} {config['precision']}",
            "type": config["type"],
            "precision": config["precision"],
            "description": config["description"],
            "loaded": is_loaded,
            "encoder_exists": encoder_exists,
            "decoder_exists": decoder_exists,
            "all_files_exist": all_files_exist,
            "encoder_size_mb": round(encoder_engine_path.stat().st_size / (1024**2), 2) if encoder_exists else None,
            "decoder_size_mb": round(decoder_onnx_path.stat().st_size / (1024**2), 2) if decoder_exists else None,
            "total_size_mb": round((encoder_engine_path.stat().st_size + decoder_onnx_path.stat().st_size) / (1024**2), 2) if all_files_exist else None
        })
    
    return jsonify({
        "models": models_info,
        "loaded_models": list(loaded_models.keys()),  # 新增：已加载的模型列表
        "tensorrt_available": TRT_AVAILABLE
    })


@app.route('/')
def index():
    """服务前端页面"""
    return send_from_directory('.', 'index.html')


@app.route('/favicon.ico')
def favicon():
    """Favicon - 返回 204 避免 404 错误"""
    return '', 204


if __name__ == '__main__':
    import argparse
    import atexit
    
    # 注册清理函数
    def cleanup_cuda():
        """清理 CUDA 资源"""
        # PyTorch 会自动管理 CUDA 资源，无需手动清理
        print("✓ 清理完成")
    
    atexit.register(cleanup_cuda)
    
    parser = argparse.ArgumentParser(
        description="SAM-HQ 分割服务 - TensorRT + ONNX 优化版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启动服务（默认端口 5000）
  python3 sam_service.py
  
  # 指定端口
  python3 sam_service.py --port 8080
  
  # 调试模式
  python3 sam_service.py --debug
  
  # 只加载指定模型
  python3 sam_service.py --models vit_tiny
  python3 sam_service.py --models vit_tiny vit_b vit_l
  
可用模型:
  vit_tiny - SAM-HQ ViT-Tiny BF16 (最快，~7ms，推荐实时应用)
  vit_b    - SAM-HQ ViT-B FP16 (快速，平衡性能)
  vit_l    - SAM-HQ ViT-L FP16 (~11ms，高精度)
  vit_h    - SAM-HQ ViT-H BF16 (最高精度)
"""
    )
    parser.add_argument('--host', default='0.0.0.0', help='主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5001, help='端口 (默认: 5001)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--models', nargs='*', choices=['vit_tiny', 'vit_b', 'vit_l', 'vit_h'],
                       help='指定要加载的模型（默认加载所有）')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 SAM-HQ 分割服务 - TensorRT + ONNX 优化版本")
    print("="*80)
    print(f"📍 服务地址: http://{args.host}:{args.port}")
    print(f"🔧 TensorRT: {'✓ 可用' if TRT_AVAILABLE else '✗ 不可用'}")
    print(f"🔧 ONNX Runtime: {'✓ 可用' if ONNX_AVAILABLE else '✗ 不可用'}")
    print(f"🎮 CUDA: {'✓ 可用' if torch.cuda.is_available() else '✗ 不可用'}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"💾 GPU: {gpu_name} ({total_mem:.1f} GB)")
    
    print(f"📦 可用模型配置: {list(MODEL_CONFIGS.keys())}")
    
    if args.models:
        print(f"📌 将只加载指定模型: {args.models}")
        # 过滤模型配置
        filtered_configs = {k: v for k, v in MODEL_CONFIGS.items() if k in args.models}
        MODEL_CONFIGS.clear()
        MODEL_CONFIGS.update(filtered_configs)
    
    print("="*80)
    
    # 预加载模型
    num_loaded = preload_models()
    
    if num_loaded == 0:
        print("\n" + "="*80)
        print("⚠ 警告: 没有成功加载任何模型！")
        print("="*80)
        print("\n可能的原因:")
        print("1. TensorRT 引擎文件不存在")
        print("   解决: 运行 ./convert_with_trtexec.sh 生成引擎")
        print("")
        print("2. PyTorch 模型文件 (.pth) 不存在")
        print("   解决: 从 SAM 官方下载模型文件到 /opt/segment-anything/models/ 目录")
        print("")
        print("3. 显存不足")
        print("   解决: 关闭其他 GPU 程序，或只加载小模型")
        print("   例如: python3 sam_service.py --models vit_b")
        print("="*80 + "\n")
        
        import sys
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ 服务已就绪！")
    print("="*80)
    print(f"🌐 访问地址: http://{args.host}:{args.port}")
    print(f"📖 API 文档: http://{args.host}:{args.port}/api/models")
    print(f"🎨 Web UI: http://{args.host}:{args.port}/")
    print("")
    print("💡 提示:")
    print("  - 使用 vit_tiny 获得最快速度 (~7ms)")
    print("  - 使用 vit_l 获得平衡性能 (~11ms)")
    print("  - 使用 vit_h 获得最高精度")
    print("  - 按 Ctrl+C 停止服务")
    print("="*80 + "\n")
    
    # 使用单线程模式避免 CUDA 上下文问题
    # Jetson Orin 上推荐使用单线程以获得更好的稳定性
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=False)
