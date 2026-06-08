# yolo_train1

基于 Ultralytics YOLOv8 的图像分类模型，使用 CIFAR-10 数据集进行训练。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 训练)
- ultralytics

## 安装依赖

```bash
pip install ultralytics
```

## 快速开始

### 训练模型

```bash
python yolov8_train.py
```

### 使用预训练模型推理

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/classify/yolov8_output/cifar10_yolov8n/weights/best.pt')

# 预测
results = model('image.jpg')
print(results[0].probs)  # 显示各类别概率
```

## 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 100 | 训练轮数 |
| batch | 128 | 批大小 |
| imgsz | 32 | 图像大小（CIFAR-10） |
| optimizer | AdamW | 优化器 |
| lr0 | 0.001 | 初始学习率 |
| cos_lr | True | 余弦学习率衰减 |

## 训练结果

- **Top-1 准确率**: 84.64%
- **Top-5 准确率**: 99.27%
- **训练时间**: ~41 分钟 (Tesla P4 GPU)
- **模型大小**: 2.99 MB

## 项目结构

```
yolo_train1/
├── yolov8_train.py    # 训练脚本
├── .gitignore         # Git 忽略文件
└── README.md          # 本文件
```

## 硬件要求

- **GPU**: NVIDIA GPU (推荐 4GB+ 显存)
- **内存**: 8GB+
- **存储**: 2GB+ (不含数据集)

## 参考资料

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)

## 许可证

MIT License
