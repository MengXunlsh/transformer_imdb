# Transformer 情感分类 — IMDB 电影评论

使用 Transformer Encoder 对 IMDB 电影评论进行情感二分类。

## 技术栈

- Python 3.12+
- PyTorch + HuggingFace datasets
- matplotlib

## 快速开始

```bash
pip install torch datasets matplotlib
python main.py
```

首次运行需下载 IMDB 数据集（约 80MB），后续离线缓存。

## 项目结构

```
transformer_imdb/
  README.md              项目说明
  main.py                 训练入口
  src/
    model.py              模型定义 (TransformerEncoder)
    data.py               数据加载、分词、构建词表
    train.py              训练循环 + checkpoint 保存
    utils.py              可视化工具
  data/
  models/
```

## 网络结构

```
Embedding(vocab_size, 128) -> PositionalEncoding -> TransformerEncoder(layers=2, heads=4) -> MeanPool -> Dropout(0.3) -> Linear(128, 1)
```

位置编码使用正弦函数注入词序信息。`src_key_padding_mask` 屏蔽 padding 位置。输出经 sigmoid 得到 0~1 概率。

## 训练配置

- 优化器: Adam, lr=0.0005
- 梯度裁剪: max_norm=1.0
- 学习率调度: StepLR (step=5, gamma=0.5)
- 损失: BCEWithLogitsLoss
- Epochs: 10

## 结果

| 指标 | 值 |
|------|-----|
| Epoch 1 Acc | 79.97% |
| 峰值 Acc | 86.11% (Epoch 5) |
| 最终 Acc | 85.09% |
| 收敛 | 5 轮即达峰值，后续轻微过拟合 |

## 可视化

```bash
python attention_viz.py
```

生成两张 Attention 热力图（`models/attention_短评论.png` / `attention_长评论.png`），展示模型在做判断时关注了哪些词。红橙色越深，该词对判断越重要。
