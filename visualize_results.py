"""
YOLOv8 训练结果可视化脚本
导出多种格式的训练结果：图像、表格、报告等
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TrainingVisualizer:
    def __init__(self, results_path, output_dir='./visualizations'):
        """
        初始化可视化器
        
        Args:
            results_path: results.csv 文件路径
            output_dir: 输出目录
        """
        self.results_path = results_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取数据
        self.df = pd.read_csv(results_path)
        print(f"✓ 已加载训练数据：{len(self.df)} epochs")
        
    def plot_accuracy_curves(self):
        """绘制准确率曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Top-1 准确率
        ax1 = axes[0]
        ax1.plot(self.df['epoch'], self.df['metrics/accuracy_top1'] * 100, 
                 'b-', linewidth=2, label='Top-1 Accuracy')
        ax1.fill_between(self.df['epoch'], 
                         self.df['metrics/accuracy_top1'] * 100 - 1,
                         self.df['metrics/accuracy_top1'] * 100 + 1,
                         alpha=0.2, color='blue')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Top-1 Accuracy Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 标注最终准确率
        final_acc = self.df['metrics/accuracy_top1'].iloc[-1] * 100
        ax1.axhline(y=final_acc, color='r', linestyle='--', alpha=0.5)
        ax1.text(self.df['epoch'].iloc[-1] * 0.7, final_acc + 1, 
                f'Final: {final_acc:.2f}%', fontsize=10, color='red')
        
        # Top-5 准确率
        ax2 = axes[1]
        ax2.plot(self.df['epoch'], self.df['metrics/accuracy_top5'] * 100, 
                 'g-', linewidth=2, label='Top-5 Accuracy')
        ax2.fill_between(self.df['epoch'], 
                         self.df['metrics/accuracy_top5'] * 100 - 0.5,
                         self.df['metrics/accuracy_top5'] * 100 + 0.5,
                         alpha=0.2, color='green')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Top-5 Accuracy Curve', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # 标注最终准确率
        final_acc5 = self.df['metrics/accuracy_top5'].iloc[-1] * 100
        ax2.axhline(y=final_acc5, color='r', linestyle='--', alpha=0.5)
        ax2.text(self.df['epoch'].iloc[-1] * 0.7, final_acc5 - 0.5, 
                f'Final: {final_acc5:.2f}%', fontsize=10, color='red')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'accuracy_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已保存准确率曲线：{save_path}")
        
    def plot_loss_curves(self):
        """绘制损失曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.df['epoch'], self.df['train/loss'], 
                'b-', linewidth=2, label='Train Loss')
        ax.plot(self.df['epoch'], self.df['val/loss'], 
                'r-', linewidth=2, label='Validation Loss')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # 标注最小验证损失
        min_val_loss_idx = self.df['val/loss'].idxmin()
        min_val_loss = self.df['val/loss'].iloc[min_val_loss_idx]
        min_epoch = self.df['epoch'].iloc[min_val_loss_idx]
        ax.scatter([min_epoch], [min_val_loss], color='red', s=100, zorder=5)
        ax.annotate(f'Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_epoch}',
                   xy=(min_epoch, min_val_loss),
                   xytext=(min_epoch + 10, min_val_loss + 0.1),
                   fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'loss_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已保存损失曲线：{save_path}")
        
    def plot_learning_rate(self):
        """绘制学习率曲线"""
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(self.df['epoch'], self.df['lr/pg0'], 
                'purple', linewidth=2, label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule (Cosine Annealing)', 
                    fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'learning_rate.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已保存学习率曲线：{save_path}")
        
    def plot_training_time(self):
        """绘制训练时间统计"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 每个 epoch 的时间
        ax1 = axes[0]
        epoch_times = self.df['time'].diff().dropna()
        ax1.plot(self.df['epoch'].iloc[1:], epoch_times / 60, 
                'orange', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Time (minutes)', fontsize=12)
        ax1.set_title('Time per Epoch', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 累积训练时间
        ax2 = axes[1]
        ax2.plot(self.df['epoch'], self.df['time'] / 60, 
                'green', linewidth=2)
        ax2.fill_between(self.df['epoch'], 0, self.df['time'] / 60, 
                        alpha=0.3, color='green')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Cumulative Time (minutes)', fontsize=12)
        ax2.set_title('Cumulative Training Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 标注总时间
        total_time = self.df['time'].iloc[-1] / 60
        ax2.text(self.df['epoch'].iloc[-1] * 0.5, total_time * 0.8,
                f'Total: {total_time:.1f} min', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_time.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已保存训练时间图：{save_path}")
        
    def plot_accuracy_comparison(self):
        """绘制准确率对比条形图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 选择关键 epoch
        key_epochs = [1, 10, 20, 50, 75, 100]
        key_epochs = [e for e in key_epochs if e <= len(self.df)]
        
        top1_accs = [self.df[self.df['epoch'] == e]['metrics/accuracy_top1'].iloc[0] * 100 
                    for e in key_epochs]
        top5_accs = [self.df[self.df['epoch'] == e]['metrics/accuracy_top5'].iloc[0] * 100 
                    for e in key_epochs]
        
        x = np.arange(len(key_epochs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, top1_accs, width, label='Top-1', color='steelblue')
        bars2 = ax.bar(x + width/2, top5_accs, width, label='Top-5', color='coral')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy Comparison at Key Epochs', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(key_epochs)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'accuracy_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已保存准确率对比图：{save_path}")
        
    def plot_all_metrics(self):
        """绘制所有指标的综合图"""
        fig = plt.figure(figsize=(16, 10))
        
        # 2x2 布局
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 准确率
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.df['epoch'], self.df['metrics/accuracy_top1'] * 100, 
                'b-', linewidth=2, label='Top-1')
        ax1.plot(self.df['epoch'], self.df['metrics/accuracy_top5'] * 100, 
                'g-', linewidth=2, label='Top-5')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.df['epoch'], self.df['train/loss'], 
                'b-', linewidth=2, label='Train')
        ax2.plot(self.df['epoch'], self.df['val/loss'], 
                'r-', linewidth=2, label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Curves', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 学习率
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.df['epoch'], self.df['lr/pg0'], 
                'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 训练时间
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.df['epoch'], self.df['time'] / 60, 
                'orange', linewidth=2)
        ax4.fill_between(self.df['epoch'], 0, self.df['time'] / 60, 
                        alpha=0.3, color='orange')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (minutes)')
        ax4.set_title('Cumulative Training Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('YOLOv8 Training Results Summary', fontsize=16, fontweight='bold', y=0.98)
        
        save_path = os.path.join(self.output_dir, 'all_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已保存综合指标图：{save_path}")
        
    def export_excel_report(self):
        """导出 Excel 报告"""
        excel_path = os.path.join(self.output_dir, 'training_report.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 原始数据
            self.df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # 统计摘要
            summary_data = {
                'Metric': [
                    'Total Epochs',
                    'Final Top-1 Accuracy (%)',
                    'Final Top-5 Accuracy (%)',
                    'Best Top-1 Accuracy (%)',
                    'Best Top-5 Accuracy (%)',
                    'Final Train Loss',
                    'Final Val Loss',
                    'Min Val Loss',
                    'Total Training Time (min)',
                    'Avg Time per Epoch (sec)',
                    'Initial Learning Rate',
                    'Final Learning Rate'
                ],
                'Value': [
                    len(self.df),
                    f"{self.df['metrics/accuracy_top1'].iloc[-1] * 100:.2f}",
                    f"{self.df['metrics/accuracy_top5'].iloc[-1] * 100:.2f}",
                    f"{self.df['metrics/accuracy_top1'].max() * 100:.2f}",
                    f"{self.df['metrics/accuracy_top5'].max() * 100:.2f}",
                    f"{self.df['train/loss'].iloc[-1]:.4f}",
                    f"{self.df['val/loss'].iloc[-1]:.4f}",
                    f"{self.df['val/loss'].min():.4f}",
                    f"{self.df['time'].iloc[-1] / 60:.1f}",
                    f"{self.df['time'].diff().mean():.2f}",
                    f"{self.df['lr/pg0'].iloc[0]:.6f}",
                    f"{self.df['lr/pg0'].iloc[-1]:.6f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 关键 epoch 数据
            key_epochs = [1, 10, 20, 30, 50, 75, 100]
            key_epochs = [e for e in key_epochs if e <= len(self.df)]
            key_data = self.df[self.df['epoch'].isin(key_epochs)].copy()
            key_data['metrics/accuracy_top1'] = key_data['metrics/accuracy_top1'] * 100
            key_data['metrics/accuracy_top5'] = key_data['metrics/accuracy_top5'] * 100
            key_data.to_excel(writer, sheet_name='Key Epochs', index=False)
            
        print(f"✓ 已保存 Excel 报告：{excel_path}")
        
    def export_markdown_report(self):
        """导出 Markdown 报告"""
        md_path = os.path.join(self.output_dir, 'training_report.md')
        
        # 计算统计数据
        total_time = self.df['time'].iloc[-1] / 60
        avg_epoch_time = self.df['time'].diff().mean()
        
        report = f"""# YOLOv8 Training Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Training Summary

| Metric | Value |
|--------|-------|
| **Total Epochs** | {len(self.df)} |
| **Final Top-1 Accuracy** | {self.df['metrics/accuracy_top1'].iloc[-1] * 100:.2f}% |
| **Final Top-5 Accuracy** | {self.df['metrics/accuracy_top5'].iloc[-1] * 100:.2f}% |
| **Best Top-1 Accuracy** | {self.df['metrics/accuracy_top1'].max() * 100:.2f}% |
| **Best Top-5 Accuracy** | {self.df['metrics/accuracy_top5'].max() * 100:.2f}% |
| **Final Train Loss** | {self.df['train/loss'].iloc[-1]:.4f} |
| **Final Val Loss** | {self.df['val/loss'].iloc[-1]:.4f} |
| **Min Val Loss** | {self.df['val/loss'].min():.4f} |
| **Total Training Time** | {total_time:.1f} minutes |
| **Avg Time per Epoch** | {avg_epoch_time:.2f} seconds |

## 📈 Accuracy Progress

| Epoch | Top-1 Acc (%) | Top-5 Acc (%) | Train Loss | Val Loss |
|-------|---------------|---------------|------------|----------|
"""
        
        # 添加关键 epoch 数据
        key_epochs = [1, 10, 20, 30, 50, 75, 100]
        key_epochs = [e for e in key_epochs if e <= len(self.df)]
        
        for epoch in key_epochs:
            row = self.df[self.df['epoch'] == epoch].iloc[0]
            report += f"| {int(row['epoch'])} | {row['metrics/accuracy_top1'] * 100:.2f} | "
            report += f"{row['metrics/accuracy_top5'] * 100:.2f} | "
            report += f"{row['train/loss']:.4f} | {row['val/loss']:.4f} |\n"
        
        report += f"""
## 📁 Generated Files

- `accuracy_curves.png` - Top-1 and Top-5 accuracy curves
- `loss_curves.png` - Training and validation loss curves
- `learning_rate.png` - Learning rate schedule
- `training_time.png` - Training time statistics
- `accuracy_comparison.png` - Accuracy comparison bar chart
- `all_metrics.png` - Comprehensive metrics overview
- `training_report.xlsx` - Excel report with all data
- `training_report.md` - This markdown report

## 🎯 Model Performance

### Strengths
- ✅ Fast training time ({total_time:.1f} minutes)
- ✅ High Top-5 accuracy ({self.df['metrics/accuracy_top5'].iloc[-1] * 100:.2f}%)
- ✅ Stable convergence

### Training Characteristics
- Initial accuracy: {self.df['metrics/accuracy_top1'].iloc[0] * 100:.2f}%
- Final accuracy: {self.df['metrics/accuracy_top1'].iloc[-1] * 100:.2f}%
- Accuracy improvement: {(self.df['metrics/accuracy_top1'].iloc[-1] - self.df['metrics/accuracy_top1'].iloc[0]) * 100:.2f}%
- Loss reduction: {self.df['train/loss'].iloc[0] - self.df['train/loss'].iloc[-1]:.4f}

---
*Report generated by YOLOv8 Training Visualizer*
"""
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 已保存 Markdown 报告：{md_path}")
        
    def generate_all(self):
        """生成所有可视化"""
        print("\n" + "="*50)
        print("YOLOv8 训练结果可视化")
        print("="*50 + "\n")
        
        # 生成所有图表
        self.plot_accuracy_curves()
        self.plot_loss_curves()
        self.plot_learning_rate()
        self.plot_training_time()
        self.plot_accuracy_comparison()
        self.plot_all_metrics()
        
        # 导出报告
        self.export_excel_report()
        self.export_markdown_report()
        
        print("\n" + "="*50)
        print(f"✓ 所有可视化已保存到：{self.output_dir}")
        print("="*50 + "\n")


def main():
    # 设置路径
    results_path = './runs/classify/yolov8_output/cifar10_yolov8n/results.csv'
    output_dir = './visualizations'
    
    # 检查文件是否存在
    if not os.path.exists(results_path):
        print(f"错误：找不到结果文件 {results_path}")
        return
    
    # 创建可视化器并生成所有图表
    visualizer = TrainingVisualizer(results_path, output_dir)
    visualizer.generate_all()


if __name__ == '__main__':
    main()
