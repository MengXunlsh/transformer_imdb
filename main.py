"""
Transformer 情感分类 — IMDB 电影评论
=========================================
用法: python main.py
"""
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

from src.data  import build_dataloaders
import torch
from src.model import TransformerClassifier
from src.train import run_training, save_metrics
from src.utils import plot_curves

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载数据
    print("\n加载 IMDB 数据...")
    train_loader, test_loader, word2idx, vocab_size, n_train, n_test = build_dataloaders()
    print(f"  训练集: {n_train} 条, 测试集: {n_test} 条")
    print(f"  词表大小: {vocab_size}")

    # 创建模型
    model = TransformerClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.3
    )
    print(f"\n网络: Transformer Encoder")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("\n开始训练...")
    train_losses, test_accs = run_training(
        model, train_loader, test_loader,
        epochs=10, lr=0.0005, device=device,
        model_dir=MODEL_DIR, checkpoint_every=5
    )

    # 保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
    }, os.path.join(MODEL_DIR, "transformer_imdb.pth"))
    save_metrics(train_losses, test_accs,
                 os.path.join(MODEL_DIR, "metrics.json"))
    print(f"\n训练完成，最终准确率: {test_accs[-1]:.2f}%")

    # 可视化
    print("\n生成可视化图表...")
    plot_curves(train_losses, test_accs,
                os.path.join(MODEL_DIR, "loss_acc_curve.png"))
    print("  已保存: models/loss_acc_curve.png")
    print("\n全部完成!")


if __name__ == "__main__":
    main()
