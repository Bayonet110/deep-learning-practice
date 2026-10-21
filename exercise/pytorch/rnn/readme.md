# Hands-on Assignment 4：基于 RNN 的 Yelp 文本情感分类

## 一、实验目标

本实验使用 PyTorch 实现基于循环神经网络的 Yelp 评论情感分类器，完成正面评论和负面评论的二分类任务。

实验分别采用 Vanilla RNN、GRU、LSTM 三种网络结构，并使用 SGD 和 Adam 两种优化器进行训练，共完成六组对比实验。

主要目标包括：

1. 掌握文本分词、词表构建、Embedding 和 Padding 等基本文本预处理方法。
2. 理解 RNN、GRU、LSTM 三种循环神经网络的结构及其在文本分类任务中的应用。
3. 对比 SGD 和 Adam 优化器的训练收敛速度与分类效果。
4. 通过训练集、验证集和测试集的 Loss、Accuracy 评价模型性能，分析收敛与过拟合情况。

## 二、数据集

### 2.1 数据来源

本实验使用 Yelp Review 情感分类数据集，参考 ACL 2021 论文：

*Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization*

论文地址：https://aclanthology.org/2021.acl-long.8/

数据集来源于 Li et al. (2018) 的公开项目：

https://github.com/lijuncen/Sentiment-and-Style-Transfer

该数据集包含经过预处理的英文 Yelp 用户评论，按照情感标签划分为正面评论和负面评论。

其中：

- 标签 `0`：Negative，负面评论。
- 标签 `1`：Positive，正面评论。

### 2.2 数据集划分

| 数据集 | 样本数量 |
|---|---:|
| 训练集 | 443,259 |
| 验证集 | 4,000 |
| 测试集 | 1,000 |

训练集用于模型参数更新，验证集用于选择最佳模型，测试集用于最终分类性能评估。

### 2.3 文本预处理

首先使用小写转换和空格分词，将原始英文评论转换为 Token 序列。

然后根据训练集构建 Vocabulary，按照 Token 出现频率分配对应的整数索引。词表最大容量设置为 20,000，实际构建得到 9,599 个 Token。

其中：

- `<PAD>`：索引为 0，用于填充序列。
- `<UNK>`：索引为 1，用于表示词表外的单词。

每条文本最大长度设置为 100，超过长度的文本进行截断。

使用 PyTorch 的 `Dataset` 和 `DataLoader` 实现数据加载，并通过动态 Padding 将同一 Batch 内的序列补齐至相同长度。

同时使用 `pack_padded_sequence()`，减少 Padding 对循环神经网络隐藏状态的影响。

## 三、模型设计

### 3.1 网络结构

三种模型采用统一的文本分类结构：

```text
Input Text
    ↓
Tokenization
    ↓
Vocabulary Encoding
    ↓
Embedding
    ↓
RNN / GRU / LSTM
    ↓
Final Hidden State
    ↓
Fully Connected Layer
    ↓
Classification Logits
```

首先通过 Embedding 层将 Token ID 映射为稠密词向量，然后使用循环神经网络提取文本序列特征。

选取最后一层的最终隐藏状态作为文本表示，经过全连接层输出两个类别的 Logits。

训练时使用 `CrossEntropyLoss` 计算分类损失。

### 3.2 三种网络模型

**Vanilla RNN**

普通循环神经网络利用当前输入与上一时间步的隐藏状态更新当前隐藏状态，能够处理序列数据，但在较长序列中可能存在梯度消失和长期依赖问题。

**GRU**

门控循环单元在普通 RNN 的基础上引入更新门和重置门，通过门控机制控制历史信息的保留与更新，从而改善长期依赖问题。

**LSTM**

长短期记忆网络引入输入门、遗忘门、输出门以及细胞状态，通过控制信息的存储、遗忘和输出，更有效地建模较长距离的序列依赖关系。

本实验统一使用单层、单向循环神经网络，不额外引入 Attention 等机制，以比较三种基本网络结构的分类效果。

## 四、实验设置

### 4.1 超参数配置

| 参数 | 设置 |
|---|---|
| Vocabulary Size | 9,599 |
| Embedding Dimension | 128 |
| Hidden Dimension | 128 |
| RNN Layers | 1 |
| Max Sequence Length | 100 |
| Batch Size | 64 |
| Epochs | 5 |
| Loss Function | CrossEntropyLoss |
| SGD Learning Rate | 0.1 |
| Adam Learning Rate | 0.001 |
| Gradient Clipping | 1.0 |
| Random Seed | 42 |
| Training Device | CUDA |

### 4.2 优化器

本实验分别采用 SGD 和 Adam 优化器。

SGD 使用固定的全局学习率，根据当前梯度更新模型参数。

Adam 在动量估计的基础上进一步利用梯度平方的指数移动平均，为不同参数自适应调整更新步长。

为了比较优化器效果，两种优化器使用相同的数据集、模型结构、Batch Size 和训练轮数，并分别采用适合其优化机制的学习率。

### 4.3 训练与评估策略

六组实验统一训练 5 个 Epoch，每轮记录训练集和验证集上的 Loss 与 Accuracy。

训练过程中使用梯度裁剪缓解梯度爆炸问题。

根据验证集 Loss 选择最佳模型并保存其参数。训练结束后加载最佳模型，在完整训练集、验证集和测试集上重新计算评价指标。

## 五、实验结果

### 5.1 六组实验对比

| 模型 | 优化器 | Train Accuracy | Dev Accuracy | Test Accuracy |
|---|---|---:|---:|---:|
| RNN | SGD | 97.42% | 96.83% | 95.60% |
| RNN | Adam | 98.64% | 97.22% | 97.00% |
| GRU | SGD | 98.11% | 97.20% | 97.00% |
| GRU | Adam | 99.13% | 97.70% | **98.40%** |
| LSTM | SGD | 98.03% | 97.25% | 97.40% |
| LSTM | Adam | 99.15% | 97.70% | 98.10% |

六组实验均能够正常收敛，测试集准确率处于 95.60%～98.40% 之间。

其中 GRU + Adam 取得最高的测试集准确率 98.40%，LSTM + Adam 次之，测试集准确率为 98.10%。

### 5.2 SGD 与 Adam 对比

从训练结果可以观察到，Adam 在三种网络结构上均取得了更高的最终测试准确率。

其中：

- RNN：Adam 比 SGD 提高 1.40 个百分点。
- GRU：Adam 比 SGD 提高 1.40 个百分点。
- LSTM：Adam 比 SGD 提高 0.70 个百分点。

从训练曲线来看，Adam 通常在前几个 Epoch 就能使 Loss 快速下降，而 SGD 的损失下降相对平稳。

这说明在当前数据集和超参数设置下，Adam 具有更快的初期收敛速度。

### 5.3 三种网络结构对比

在 SGD 优化器下，三种模型的测试准确率分别为：

- RNN：95.60%
- GRU：97.00%
- LSTM：97.40%

GRU 和 LSTM 均优于普通 RNN。

在 Adam 优化器下，三种模型的测试准确率分别为：

- RNN：97.00%
- GRU：98.40%
- LSTM：98.10%

GRU 和 LSTM 同样表现较好。

这说明门控机制有助于循环神经网络提取文本序列中的情感信息。

不过，GRU 与 LSTM 的测试准确率差距较小，不能仅依据本次单次实验认定某一种网络结构在所有任务中均具有优势。

### 5.4 收敛与过拟合分析

六组模型的训练损失均随 Epoch 增加而下降，训练准确率逐渐提高，说明模型能够正常学习。

SGD 训练的三组模型整体表现出较平稳的收敛趋势，在第 5 个 Epoch 取得最低验证损失。

Adam 训练的模型收敛速度较快，但部分模型在后期出现过拟合迹象。

例如，GRU + Adam 的训练损失从第 3 轮的 0.0409 继续下降至第 5 轮的 0.0233，但验证损失由 0.0681 上升至 0.0886。

LSTM + Adam 也存在类似情况。

因此，本实验通过选择验证集 Loss 最低的模型进行最终测试，降低后期过拟合对测试效果的影响。

## 六、运行方法

安装所需依赖：

```bash
pip install torch matplotlib tqdm
```

在项目目录下执行：

```bash
python train.py
```

程序会依次完成六组实验，并将结果保存至 `outputs/` 目录。

主要输出包括：

- `training_history.csv`：每轮训练及验证指标。
- `training_curves.png`：训练与验证的 Loss、Accuracy 曲线。
- `best_model.pt`：验证集最优模型参数。
- `metrics.json`：最终评价指标与实验配置。
- `training_log.txt`：训练过程与最终评价结果。
- `experiment_summary.csv`：六组实验结果汇总。

## 七、实验总结

本实验基于 PyTorch 实现了 RNN、GRU、LSTM 三种文本情感分类模型，并使用 SGD 和 Adam 完成六组对比实验。

实验结果表明：

1. 三种循环神经网络均能够在 Yelp 情感分类数据集上正常收敛，取得较好的分类效果。
2. 相比普通 RNN，GRU 和 LSTM 在本次实验中整体取得了更好的分类准确率。
3. Adam 相比普通 SGD 表现出更快的初期收敛速度，并在三种模型上均获得了更高的测试准确率。
4. Adam 在训练后期可能出现过拟合现象，因此使用验证集选择最佳模型具有实际意义。

本实验加深了对循环神经网络、文本序列处理、反向传播和优化器工作机制的理解，也完成了从原始文本数据到神经网络训练、评估和结果分析的完整实现流程。

需要注意的是，本实验仅采用单次随机种子及固定超参数，且测试集规模为 1,000 条，因此实验结论主要适用于当前数据集和实验设置，不能直接推广为三种模型在所有 NLP 任务中的性能排序。