# A project for personal graduation design by rebuilding DarkIR 
本项目基于 DarkIR 网络重构，用于低光照图像增强。


## Dependencies and Installation

- Python == 3.12.3
- PyTorch == 2.5.1
- CUDA == 12.4
- Other required packages in `requirements.txt`

```
# git clone this repository
git clone https://github.com/Limkeke/DarkIR-rebuild.git
cd DarkIR-main

# create python environment
python3 -m venv venv_DarkIR
source venv_DarkIR/bin/activate

# install python dependencies
pip install -r requirements.txt
```


## Datasets
The datasets used for training and/or evaluation are:

|Dataset     | Sets of images | Source  |
| -----------| :---------------:|------|
|LOLv2-real        | 689 training pairs / 100 test pairs | [Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view) |
|LOLv2-synth        | 900 training pairs / 100 test pairs | [Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view) |
|LOL      | 485 training pairs / 15 test pairs | [Official Site](https://daooshee.github.io/BMVC2018website/)  |
|Real-LOLBlur | 1354 unpaired images  | [LEDNet](https://github.com/sczhou/LEDNet)  |

You can download each specific dataset and put it on the `/data/datasets` folder for testing. 

## 数据集预处理（必做）
**LOLv2-Synthetic 数据集不做额外处理**
**针对 LOLv2-Real_captured 数据集**
LOLv2-Real_captured 数据集原始文件名存在前缀差异（`normal000xx.png` / `low000xx.png`），需统一重命名为纯数字格式 `000xx.png`，保证数据集配对加载。

### 批量重命名命令
```bash
# 1. 处理训练集 Normal（删除 normal 前缀）
cd /root/DarkIR-main/data/datasets/LOL-v2/Real_captured/train/Normal
for file in normal*.png; do mv "$file" "${file#normal}"; done

# 2. 处理训练集 Low（删除 low 前缀）
cd /root/DarkIR-main/data/datasets/LOL-v2/Real_captured/train/Low
for file in low*.png; do mv "$file" "${file#low}"; done

# 3. 处理测试集 Normal
cd /root/DarkIR-main/data/datasets/LOL-v2/Real_captured/test/Normal
for file in normal*.png; do mv "$file" "${file#normal}"; done

# 4. 处理测试集 Low
cd /root/DarkIR-main/data/datasets/LOL-v2/Real_captured/test/Low
for file in low*.png; do mv "$file" "${file#low}"; done
```

### 预处理效果
- 原文件名：`normal00001.png` / `low00001.png`
- 处理后：`00001.png`（Normal 与 Low 文件夹内文件名完全一致）

---

## 训练启动命令
### 1.LOLv2-Real_captured完成数据集预处理后，执行以下命令启动训练：
```bash
cd /root/DarkIR-main
PYTHONPATH=/root/DarkIR-main/train python train/basicsr/train.py -opt options/train/LOLv2-Real_captured.yml
```

### 2.训练 LOLv2-Syn 合成数据集
```bash
cd /root/DarkIR-main
PYTHONPATH=/root/DarkIR-main/train python train/basicsr/train.py -opt options/train/LOLv2-Synthetic.yml
```
---


## 测试启动命令
### 1.LOLv2-Real_captured完成测试后，执行以下命令启动测试：
```bash
cd /root/DarkIR-main
python testing.py -p options/test/LOLv2_Syn.yml
```

### 2.测试 LOLv2-Syn 合成数据集
```bash
cd /root/DarkIR-main
python testing.py -p options/test/LOLv2_Real.yml
```
---


## 项目说明
- 本项目为个人毕业设计，核心工作为 DarkIR 网络重构与适配(https://github.com/cidautai/DarkIR)
- 基于 BasicSR 框架开发，适配 PyTorch 深度学习环境(https://github.com/Zhengzkang/DarkIR)
- 支持 LOLv2 等低光照增强数据集训练

---

