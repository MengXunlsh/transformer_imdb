"""
Attention 可视化脚本
用法: python attention_viz.py
"""
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.data  import build_dataloaders, tokenize
import torch
from src.model import TransformerClassifier
import re
import matplotlib.pyplot as plt


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE, "models")

    # 加载数据和词表
    print("加载数据...")
    _, test_loader, word2idx, vocab_size, _, _ = build_dataloaders(batch_size=1)

    # 加载模型
    ckpt = torch.load(
        os.path.join(MODEL_DIR, "transformer_imdb.pth"),
        map_location=device, weights_only=True
    )
    model = TransformerClassifier(
        vocab_size=vocab_size, embed_dim=128, num_heads=4,
        num_layers=2, dropout=0.3
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print("模型已加载\n")

    # 选取短评论和长评论各一条
    from datasets import load_dataset
    imdb = load_dataset("stanfordnlp/imdb")
    
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    short = None
    long = None
    for text, label in zip(imdb["test"]["text"], imdb["test"]["label"]):
        tokens = tokenize(text, word2idx)
        n = len(tokens)
        if short is None and 30 <= n <= 60:
            short = (text, label, tokens)
        if long is None and n >= 300:
            long = (text, label, tokens)
        if short and long:
            break

    # 画 Attention 热力图（取 Layer 1）
    for name, (text, label, tokens) in [("短评论", short), ("长评论", long)]:
        seq = torch.tensor([tokens[:512]], dtype=torch.long).to(device)
        words = [
            list(word2idx.keys())[list(word2idx.values()).index(t)]
            if t in word2idx.values() else "<unk>"
            for t in tokens[:30]  # 只展示前 30 个词
        ]

        _, attn_layers = model.forward_with_attention(seq)

        # 画 Layer 1 的注意力（选第一条样本）
        attn = attn_layers[0][0].cpu().detach().numpy()  # (seq_len, seq_len)
        attn = attn[:30, :30]  # 截取前 30 个词

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.pcolormesh(attn, cmap='YlOrRd', edgecolors='white', linewidth=0.5)
        ax.invert_yaxis()  # pcolormesh 默认 y 轴朝上，反转一下

        ax.set_xticks(range(len(words)))
        ax.set_yticks(range(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=6)
        ax.set_yticklabels(words, fontsize=6)

        sentiment = "正面" if label == 1 else "负面"
        ax.set_title(
            f'{name} — Layer 1 Attention ({sentiment})\n'
            f'{text[:100]}...',
            fontsize=10
        )
        plt.colorbar(im, ax=ax, shrink=0.8)

        out_path = os.path.join(MODEL_DIR, f'attention_{name}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"已保存: models/attention_{name}.png")

    print("\n全部完成!")


if __name__ == "__main__":
    main()
