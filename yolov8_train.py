"""
YOLOv8 图像分类训练脚本
支持 CIFAR-10/CIFAR-100 数据集
"""

import os
import torch
from ultralytics import YOLO

def main():
    # 检查 CUDA 可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备：{device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'显存：{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    # 创建输出目录
    output_dir = './yolov8_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 YOLOv8 分类模型（nano 版本，速度最快）
    model = YOLO('yolov8n-cls.pt')
    
    # 训练配置
    results = model.train(
        data='cifar10',  # 使用内置的 CIFAR-10 数据集
        epochs=100,
        batch=128,
        imgsz=32,  # CIFAR-10 图像大小
        device=device,
        project=output_dir,
        name='cifar10_yolov8n',
        verbose=True,
        cache=True,  # 缓存数据加速训练
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,  # 余弦学习率衰减
        augment=True,
        dropout=0.0,
    )
    
    # 验证模型
    metrics = model.val()
    print(f"验证准确率: {metrics.top1}")
    
    # 保存模型
    model.save(os.path.join(output_dir, 'cifar10_yolov8n_best.pt'))
    print("训练完成！")

if __name__ == '__main__':
    main()
