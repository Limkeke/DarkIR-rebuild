import re
import matplotlib.pyplot as plt
import os

log_path = "/root/DarkIR-main/train/experiments/LOLv2-Synthetic/train_LOLv2-Synthetic_20260419_145312.log"

iters = []
pixel_losses = []
perceptual_losses = []
edge_losses = []
enhance_losses = []

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip()
        # 匹配格式：iter: 343,800 这样的数字（带逗号）
        iter_match = re.search(r'iter:\s*([\d,]+)', line)
        pixel_match = re.search(r'l_pixel_loss:\s*([\d\.e\-]+)', line)
        perceptual_match = re.search(r'l_perceptual_loss:\s*([\d\.e\-]+)', line)
        edge_match = re.search(r'l_edge_loss:\s*([\d\.e\-]+)', line)
        enhance_match = re.search(r'l_enhance_loss:\s*([\d\.e\-]+)', line)

        # 所有损失都存在才记录
        if all([iter_match, pixel_match, perceptual_match, edge_match, enhance_match]):
            # 把iter里的逗号去掉，转成整数
            it = int(iter_match.group(1).replace(',', ''))
            iters.append(it)
            pixel_losses.append(float(pixel_match.group(1)))
            perceptual_losses.append(float(perceptual_match.group(1)))
            edge_losses.append(float(edge_match.group(1)))
            enhance_losses.append(float(enhance_match.group(1)))

if not iters:
    print("❌ 没有读到任何数据，请检查日志文件路径！")
    exit()

print(f"✅ 成功读取到 {len(iters)} 条数据，迭代范围：{iters[0]} ~ {iters[-1]}")

# --------------------- 图1：各损失分量 ---------------------
plt.figure(figsize=(12, 5))
plt.plot(iters, pixel_losses, label='Pixel Loss', linewidth=1.2)
plt.plot(iters, perceptual_losses, label='Perceptual Loss', linewidth=1.2)
plt.plot(iters, edge_losses, label='Edge Loss', linewidth=1.2)
plt.plot(iters, enhance_losses, label='Enhance Loss', linewidth=1.2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.title('Training Loss Components', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
# 关键修改：dpi=600
plt.savefig("loss_components.png", dpi=600, bbox_inches='tight')
plt.close()

# --------------------- 图2：总损失 ---------------------
total_loss = [p + pe + e + en for p, pe, e, en in zip(pixel_losses, perceptual_losses, edge_losses, enhance_losses)]
plt.figure(figsize=(12, 5))
plt.plot(iters, total_loss, label='Total Loss', linewidth=1.5, color='#1f77b4')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Total Loss', fontsize=12)
plt.title('Total Training Loss', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
# 关键修改：dpi=600
plt.savefig("total_loss.png", dpi=600, bbox_inches='tight')
plt.close()

print("\n✅ 已生成两张高清曲线（dpi=600）：")
print("loss_components.png（包含所有损失分量）")
print("total_loss.png（包含完整总损失曲线）")