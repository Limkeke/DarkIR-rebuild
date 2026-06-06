import re
import matplotlib.pyplot as plt
import numpy as np

log_path = "/root/DarkIR-main/train/experiments/LOLv2-Synthetic/train_LOLv2-Synthetic_20260419_145312.log"

iters = []
pixel_losses = []
perceptual_losses = []
edge_losses = []
enhance_losses = []

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip()
        iter_match = re.search(r'iter:\s*([\d,]+)', line)
        pixel_match = re.search(r'l_pixel_loss:\s*([\d\.e\-]+)', line)
        perceptual_match = re.search(r'l_perceptual_loss:\s*([\d\.e\-]+)', line)
        edge_match = re.search(r'l_edge_loss:\s*([\d\.e\-]+)', line)
        enhance_match = re.search(r'l_enhance_loss:\s*([\d\.e\-]+)', line)

        if all([iter_match, pixel_match, perceptual_match, edge_match, enhance_match]):
            it = int(iter_match.group(1).replace(',', ''))
            iters.append(it)
            pixel_losses.append(float(pixel_match.group(1)))
            perceptual_losses.append(float(perceptual_match.group(1)))
            edge_losses.append(float(edge_match.group(1)))
            enhance_losses.append(float(enhance_match.group(1)))

# 2行2列子图
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=600)
axes = axes.flatten()

axes[0].plot(iters, pixel_losses, label='Pixel Loss', color='#1f77b4')
axes[0].set_title('Pixel Loss')
axes[0].grid(alpha=0.3)

axes[1].plot(iters, perceptual_losses, label='Perceptual Loss', color='#ff7f0e')
axes[1].set_title('Perceptual Loss')
axes[1].grid(alpha=0.3)

axes[2].plot(iters, edge_losses, label='Edge Loss', color='#2ca02c')
axes[2].set_title('Edge Loss')
axes[2].grid(alpha=0.3)

axes[3].plot(iters, enhance_losses, label='Enhance Loss', color='#d62728')
axes[3].set_title('Enhance Loss')
axes[3].grid(alpha=0.3)

fig.supxlabel('Iteration')
fig.supylabel('Loss Value')
fig.suptitle('Training Loss Components', fontsize=16)
plt.tight_layout()
plt.savefig("loss_components_subplots.png", dpi=600, bbox_inches='tight')
plt.close()

print("✅ 已生成子图版损失分量图 loss_components_subplots.png")