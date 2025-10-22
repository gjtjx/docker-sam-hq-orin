import base64
import json
from pathlib import Path
import requests
import numpy as np
from PIL import Image
from datetime import datetime

# 配置
SERVER_URL = "https://segmentation.ensightful.xyz"
IMAGE_PATH = "C:/Users/JietianGUO/Downloads/yujie_images/D_17599118105627208.jpg"#填写路径
MODEL = "vit_h"

# 读取并编码图片
with open(IMAGE_PATH, "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    image_base64 = f"data:image/jpeg;base64,{image_data}"

# 请求数据
data = {
    "model": MODEL,
    "prompt_type": "point",  # 可选: "auto", "point", "box"
    "image": image_base64,
    "use_tensorrt": True,
    "alpha": 0.5,
    "points": [[1465, 1683, 1], [614, 2122, 1],[2539, 1965, 1]],  # 如果使用 point 模式
    # "boxes": [[50, 50, 200, 200]],  # 如果使用 box 模式
}

# 发送请求
print("发送分割请求...")
response = requests.post(f"{SERVER_URL}/api/segment", json=data)

# 创建保存目录
output_dir = Path("./mask_outputs")
output_dir.mkdir(exist_ok=True)

# 打印结果
if response.status_code == 200:
    result = response.json()
    print(f"成功! 生成了 {result['result']['info']['num_masks']} 个 mask")
    print(f"总耗时: {result['performance']['total_time_ms']} ms")
    print(f"显存使用: {result['performance']['memory_mb']} MB")
    
    # 保存masks到本地
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(IMAGE_PATH).stem
    
    # 如果有masks数据，保存它们
    if 'masks' in result['result']:
        masks_data = result['result']['masks']
        for idx, mask_obj in enumerate(masks_data):
            # 获取 mask 的 base64 数据（新的 API 格式）
            mask_base64 = mask_obj['mask']  # mask_obj 是一个字典，包含 mask、bbox、area、color
            
            # 解码base64图片
            mask_bytes = base64.b64decode(mask_base64.split(',')[1] if ',' in mask_base64 else mask_base64)
            
            # 保存为PNG文件
            output_path = output_dir / f"{image_name}_mask_{idx}_{timestamp}.png"
            with open(output_path, "wb") as f:
                f.write(mask_bytes)
            
            # 打印 mask 信息
            bbox = mask_obj.get('bbox', [])
            area = mask_obj.get('area', 0)
            color = mask_obj.get('color', [])
            print(f"已保存 mask {idx} 到: {output_path}")
            print(f"  - 边界框: {bbox}")
            print(f"  - 面积: {area} 像素")
            print(f"  - 颜色: RGB{tuple(color)}")
    
    # 如果有叠加图片，也保存
    if 'overlay' in result['result']:
        overlay_base64 = result['result']['overlay']
        overlay_bytes = base64.b64decode(overlay_base64.split(',')[1] if ',' in overlay_base64 else overlay_base64)
        overlay_path = output_dir / f"{image_name}_overlay_{timestamp}.png"
        with open(overlay_path, "wb") as f:
            f.write(overlay_bytes)
        print(f"已保存叠加图片到: {overlay_path}")
    
    print(f"\n所有文件已保存到目录: {output_dir.absolute()}")
else:
    print(f"错误: {response.status_code}")
    print(response.text)