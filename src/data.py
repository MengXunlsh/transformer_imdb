"""
IMDB 数据加载与预处理
依赖: pip install datasets
"""
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import re


def build_vocab(texts, max_vocab=25000):
    word_counts = {}
    for text in texts:
        for word in re.findall(r'\b\w+\b', text.lower()):
            word_counts[word] = word_counts.get(word, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    sorted_words = sorted_words[:max_vocab]

    word2idx = {"<pad>": 0, "<unk>": 1}
    for word, _ in sorted_words:
        word2idx[word] = len(word2idx)

    return word2idx, len(word2idx)


def tokenize(text, word2idx, max_len=512):
    tokens = [word2idx.get(w, word2idx["<unk>"])
              for w in re.findall(r'\b\w+\b', text.lower())]
    return tokens[:max_len]


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx], self.word2idx)
        label  = self.labels[idx]
        return tokens, label


def collate_batch(batch):
    sequences = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    labels    = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences, labels


def build_dataloaders(batch_size=64, max_vocab=25000):
    imdb = load_dataset("stanfordnlp/imdb")
    word2idx, vocab_size = build_vocab(imdb["train"]["text"], max_vocab)

    train_ds = IMDBDataset(
        imdb["train"]["text"], imdb["train"]["label"], word2idx
    )
    test_ds = IMDBDataset(
        imdb["test"]["text"], imdb["test"]["label"], word2idx
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_batch
    )

    return train_loader, test_loader, word2idx, vocab_size, len(train_ds), len(test_ds)
