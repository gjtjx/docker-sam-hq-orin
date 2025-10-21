FROM dustynv/l4t-pytorch:r36.4.0

# 首先在root用户下配置pip镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.timeout 300 && \
    pip config set global.retries 3 && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 创建非 root 用户
RUN useradd -m appuser

# 为用户也配置pip镜像源
RUN mkdir -p /home/appuser/.config/pip && \
    echo "[global]" > /home/appuser/.config/pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /home/appuser/.config/pip/pip.conf && \
    echo "timeout = 300" >> /home/appuser/.config/pip/pip.conf && \
    echo "retries = 3" >> /home/appuser/.config/pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> /home/appuser/.config/pip/pip.conf && \
    chown -R appuser:appuser /home/appuser/.config

USER appuser

# 设置工作目录
WORKDIR /app

# 将依赖文件复制到容器内
COPY onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl .

# 验证基础镜像中PyTorch的CUDA支持
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"

# 测试pip镜像源连接
RUN pip list --format=freeze | head -5 && echo "✓ pip镜像源连接正常"

# 分批安装依赖，提高成功率，强制使用清华镜像源
# 第一批：基础科学计算依赖
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    coloredlogs flatbuffers numpy packaging protobuf sympy

# 第二批：应用依赖（不包括torch/torchvision，使用基础镜像预装的版本）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    opencv-python==4.10.0.84 flask flask-cors pycuda timm

# 第三批：安装sam-hq（使用--no-deps避免torch依赖覆盖）
RUN pip install --no-cache-dir --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    git+https://github.com/SysCV/sam-hq.git

# 第四批：安装自编译的onnxruntime-gpu（使用--no-deps避免依赖冲突）
RUN pip install --no-cache-dir --no-deps /app/onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl

# 验证核心依赖导入
RUN python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
RUN python3 -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"
RUN python3 -c "import onnxruntime as ort; print(f'✓ ONNX Runtime {ort.__version__} - 提供者: {ort.get_available_providers()}')"
RUN python3 -c "import flask; import pycuda; print('✓ Flask 和 PyCUDA 导入成功')"
# 复制项目代码和启动脚本
COPY . .

# 设置环境变量
ENV ENVIRONMENT=production
# 确保CUDA环境变量正确设置
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 暴露端口
EXPOSE 5001

# 指定容器启动时执行启动脚本（运行时验证CUDA）
CMD ["/app/start.sh"]
