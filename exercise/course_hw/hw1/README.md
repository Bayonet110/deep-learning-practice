# 李宏毅 2023 Spring HW1：COVID-19 回归预测

独立学习版 PyTorch 实现，输入 `data/covid_train.csv` 和 `data/covid_test.csv`。

## 快速启动

```bash
pip install -r requirements.txt
python train.py
```

预期目录结构：

```
ml2023_hw1/
├── train.py
├── requirements.txt
├── data/
│   ├── covid_train.csv
│   └── covid_test.csv
└── outputs/  # 自动生成
    ├── best_model.pt
    ├── submission.csv
    └── loss_curve.png
```

模型：Linear(D,64)→ReLU→Linear(64,32)→ReLU→Linear(32,1)。使用 MSE、Adam、早停及训练子集拟合的 Z-score 标准化。输出 `submission.csv` 两列 `id,tested_positive`。

## 学习建议

1. 首次运行时观察验证曲线。
2. 修改 `LEARNING_RATE`、`WEIGHT_DECAY`、隐藏层维度，记录验证集 MSE。
3. 使用不同随机种子重复实验，理解验证集分数的方差。
4. 进一步实验相关性特征筛选；不能使用测试集标签或额外的预训练模型。

数据需自行从 Kaggle 官方竞赛页面获取。本实现不包含数据文件。
