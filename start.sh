#!/bin/bash
# SAM-HQ 容器启动脚本
# 在运行时验证CUDA环境并启动服务

echo "🚀 启动 SAM-HQ 服务..."
echo "========================================"

# 验证CUDA环境
echo "🔧 验证CUDA环境..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    显存: {props.total_memory / (1024**3):.1f} GB')
        print(f'    计算能力: {props.major}.{props.minor}')
"

# 检查CUDA是否可用
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "✅ CUDA环境验证通过"
    echo "🎮 将使用GPU加速"
else
    echo "❌ CUDA不可用！"
    echo "请检查以下项目："
    echo "  1. 使用 --gpus all 参数启动容器"
    echo "  2. NVIDIA驱动已正确安装"
    echo "  3. nvidia-docker 或 nvidia-container-toolkit 已配置"
    echo "  4. Docker版本支持GPU"
    echo ""
    echo "示例启动命令："
    echo "  docker run --gpus all -p 5001:5001 sam-hq-cuda"
    echo ""
    echo "⚠️  将尝试以CPU模式继续启动..."
fi

# 验证其他关键依赖
echo ""
echo "🔍 验证其他依赖..."
python3 -c "
try:
    import cv2
    print(f'✓ OpenCV {cv2.__version__}')
except ImportError as e:
    print(f'✗ OpenCV 导入失败: {e}')

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f'✓ ONNX Runtime {ort.__version__}')
    print(f'  可用提供者: {providers}')
    if 'CUDAExecutionProvider' in providers:
        print('  ✓ CUDA执行提供者可用')
    else:
        print('  ⚠️  CUDA执行提供者不可用')
except ImportError as e:
    print(f'✗ ONNX Runtime 导入失败: {e}')

try:
    import flask
    print('✓ Flask 可用')
except ImportError as e:
    print(f'✗ Flask 导入失败: {e}')

try:
    import pycuda
    print('✓ PyCUDA 可用')
except ImportError as e:
    print(f'✗ PyCUDA 导入失败: {e}')
"

echo ""
echo "🌐 启动 SAM-HQ 服务..."
echo "========================================"

# 启动主服务
exec python3 /app/sam-hq/sam_service.py --models vit_h