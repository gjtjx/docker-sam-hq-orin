# docker-sam-hq-orin
docker-sam-hq-orin

onnx2trt:
docker exec -it trt_convert bash 进入dustynv/l4t-pytorch:r36.4.0 镜像，重新转换，注意文件路径挂载

VITH
/usr/src/tensorrt/bin/trtexec --onnx=/trt_convert/sam_hq/models/sam_hq_vit_h_encoder-onnx/sam_hq_vit_h_encoder.onnx --saveEngine=/trt_convert/sam_hq/models/sam_hq_vit_h_encoder_bf16.plan --bf16 --memPoolSize=workspace:12G 

VITL
/usr/src/tensorrt/bin/trtexec --onnx=/trt_convert/sam_hq/models/sam_hq_vit_l_encoder.onnx --saveEngine=/trt_convert/sam_hq/models/sam_hq_vit_l_encoder_fp16.plan --fp16 --memPoolSize=workspace:12G

VITB
/usr/src/tensorrt/bin/trtexec --onnx=/trt_convert/sam_hq/models/sam_hq_vit_b_encoder.onnx --saveEngine=/trt_convert/sam_hq/models/sam_hq_vit_b_encoder_fp16.plan --fp16 --memPoolSize=workspace:12G

VITTiny
/usr/src/tensorrt/bin/trtexec --onnx=/trt_convert/sam_hq/models/sam_hq_vit_tiny_encoder.onnx --saveEngine=/trt_convert/sam_hq/models/sam_hq_vit_tiny_encoder_bf16.plan --bf16 --memPoolSize=workspace:12G 

Dockerfile
From dustynv/l4t-pytorch:r36.4.0

# 创建非 root 用户
RUN useradd -m appuser
USER appuser

# 设置工作目录
WORKDIR /app

# 将依赖文件复制到容器内
COPY onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl .

# 安装依赖（使用国内镜像源可加速）
RUN pip install /app/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --no-deps opencv-python==4.10.0.84  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install  --no-cache-dir flask flask_cors pycuda timm onnxruntime-gpu git+https://github.com/SysCV/sam-hq.git --ignore-installed blinker -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 设置环境变量
ENV ENVIRONMENT=production

# 暴露端口
EXPOSE 5001

# 指定容器启动时执行的命令
CMD ["python3", "/app/sam_hq/sam_service.py", "--models", "vit_h"]


#创建容器
docker build -t sam-hq-service:latest .

#查看容器
docker ps -a

# 删除容器
docker rm -f sam-hq-container

# 保存
docker save -o sam-hq-service.tar sam-hq-service:latest

# 加载
docker load -i /path/sam-hq-service.tar

#运行容器
docker run -d --name sam-hq-service --runtime=nvidia --network host -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility sam-hq-cuda

#查看日志
docker logs -f sam-hq-container

#trt
docker start trt_convert
docker exec -it trt_convert bash



Onnxruntime-gpu 安装问题
1.Cmake 升级：
# 添加 Kitware 官方仓库
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

# 安装最新 CMakesudo apt update
sudo apt install -y cmake
2.编译安装Onnxruntime-gpu

# 安装依赖sudo apt install -y cmake build-essential python3-dev libprotobuf-dev protobuf-compiler

# 克隆源码
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime


#swap 内存设置19GB

sudo fallocate -l 19G /var/swapfile
sudo chmod 666 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
可选：永久生效 
`echo '/var/swapfile swap swap defaults 0 0'

# 配置编译
./build.sh --config Release --build_shared_lib --parallel 4 --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --use_tensorrt --tensorrt_home /usr/src/tensorrt --skip_tests --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 --cmake_extra_defines CMAKE_CUDA_FLAGS="-allow-unsupported-compiler" --cmake_extra_defines onnxruntime_NVCC_THREADS=1 --build_wheel

# 安装cd build/Linux/Release
sudo pip3 install dist/*.whl


