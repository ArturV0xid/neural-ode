from __future__ import annotations

import re
import json
from collections import Counter
from typing import Optional

import torch
import nltk
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

from .config import HyperParams

nltk.download('stopwords')

PAD_IDX = 0
CLS_IDX = 1
UNK_IDX = 2

SPECIAL_TOKENS: dict[str, int] = {"[PAD]": PAD_IDX, "[CLS]": CLS_IDX, "[UNK]": UNK_IDX}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


class WordTokenizer:
    def __init__(self, hp: HyperParams, min_word_len: int = 3) -> None:
        self.vocab_size = hp.vocab_size
        self.min_word_len = min_word_len
        self.word2idx: dict[str, int] = dict(SPECIAL_TOKENS)
        self.idx2word: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        
        self.nltk_tokenizer = WordPunctTokenizer()
        self.stop_words = set(stopwords.words("english"))

    def _preprocess(self, text: str) -> list[str]:
        text = text.lower()
        
        text = _URL_RE.sub("", text)
        text = _EMAIL_RE.sub("", text)
        
        tokens = self.nltk_tokenizer.tokenize(text)
        
        filtered_tokens = []
        for token in tokens:
            if not token.isalpha():
                continue
                
            if token in self.stop_words:
                continue
                
            if len(token) < self.min_word_len:
                continue
                
            filtered_tokens.append(token)
            
        return filtered_tokens

    def build_vocab(self, texts: list[str]) -> None:
        counts: Counter[str] = Counter(
            token for doc in texts for token in self._preprocess(doc)
        )
        slots = self.vocab_size - len(SPECIAL_TOKENS)
        for word, _ in counts.most_common(slots):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: int) -> list[int]:
        tokens = self._preprocess(text)
        ids = [CLS_IDX] + [self.word2idx.get(t, UNK_IDX) for t in tokens[: max_len - 1]]
        pad = max(0, max_len - len(ids))
        return ids + [PAD_IDX] * pad

    def decode(self, ids: list[int]) -> str:
        return " ".join(
            self.idx2word.get(i, "[UNK]") for i in ids if i not in (PAD_IDX,)
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"vocab_size": self.vocab_size, "word2idx": self.word2idx},
                f,
                indent=4,
            )

    @classmethod
    def load(cls, path: str, hp: HyperParams) -> WordTokenizer:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        tok = cls(hp=hp)
        tok.word2idx = payload["word2idx"]
        tok.idx2word = {int(idx): word for word, idx in payload["word2idx"].items()}
        return tok


def load_data(
    hp: HyperParams,
    tokenizer: Optional[WordTokenizer] = None,
) -> dict:
    kwargs = {"remove": ("headers", "footers", "quotes")}
    train_raw = fetch_20newsgroups(subset="train", **kwargs)
    test_raw = fetch_20newsgroups(subset="test", **kwargs)

    if tokenizer is None:
        tokenizer = WordTokenizer(hp=hp)
        tokenizer.build_vocab(train_raw.data)

    def _encode_batch(texts: list[str]) -> torch.Tensor:
        return torch.tensor(
            [tokenizer.encode(doc, hp.max_seq_len) for doc in texts],
            dtype=torch.long,
        )

    train_ds = TensorDataset(
        _encode_batch(train_raw.data),
        torch.tensor(train_raw.target, dtype=torch.long),
    )
    test_ds = TensorDataset(
        _encode_batch(test_raw.data),
        torch.tensor(test_raw.target, dtype=torch.long),
    )

    return {
        "tokenizer": tokenizer,
        "train_loader": DataLoader(
            train_ds, batch_size=hp.batch_size, shuffle=True
        ),
        "test_loader": DataLoader(
            test_ds, batch_size=hp.batch_size, shuffle=False
        ),
        "label_names": list(train_raw.target_names),
    }
