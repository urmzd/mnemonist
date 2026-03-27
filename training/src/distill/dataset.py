"""Dataset loading and preprocessing for distillation training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CodePairDataset(Dataset):
    """Dataset of (code, description) pairs for contrastive learning."""

    def __init__(self, path: Path, tokenizer, max_len: int = 512):
        self.pairs = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.pairs.append(obj)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        from src.models.code_encoder import preprocess_code

        code_text = preprocess_code(pair["code"], pair.get("language"))
        desc_text = pair["description"]

        code_enc = self.tokenizer(
            code_text, max_length=self.max_len, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        desc_enc = self.tokenizer(
            desc_text, max_length=self.max_len, truncation=True, padding="max_length",
            return_tensors="pt",
        )

        return {
            "code_input_ids": code_enc["input_ids"].squeeze(0),
            "code_attention_mask": code_enc["attention_mask"].squeeze(0),
            "desc_input_ids": desc_enc["input_ids"].squeeze(0),
            "desc_attention_mask": desc_enc["attention_mask"].squeeze(0),
        }


class CodeSimilarityDataset(Dataset):
    """Dataset of (code_a, code_b, similarity) triples."""

    def __init__(self, path: Path, tokenizer, max_len: int = 512):
        self.triples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path) as f:
            for line in f:
                self.triples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        triple = self.triples[idx]
        from src.models.code_encoder import preprocess_code

        a_text = preprocess_code(triple["code_a"], triple.get("language_a"))
        b_text = preprocess_code(triple["code_b"], triple.get("language_b"))

        a_enc = self.tokenizer(
            a_text, max_length=self.max_len, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        b_enc = self.tokenizer(
            b_text, max_length=self.max_len, truncation=True, padding="max_length",
            return_tensors="pt",
        )

        return {
            "a_input_ids": a_enc["input_ids"].squeeze(0),
            "a_attention_mask": a_enc["attention_mask"].squeeze(0),
            "b_input_ids": b_enc["input_ids"].squeeze(0),
            "b_attention_mask": b_enc["attention_mask"].squeeze(0),
            "similarity": torch.tensor(triple["similarity"], dtype=torch.float32),
        }


class MemoryQueryDataset(Dataset):
    """Dataset of (query, relevant, irrelevant) triples for memory retrieval."""

    def __init__(self, path: Path, tokenizer, max_len: int = 256):
        self.triples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                # Flatten: each relevant/irrelevant pair becomes a training sample
                for rel in obj.get("relevant", []):
                    for irr in obj.get("irrelevant", []):
                        self.triples.append({
                            "query": obj["query"],
                            "positive": rel["description"],
                            "negative": irr["description"],
                            "relevance": rel.get("relevance", 1.0),
                        })

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        t = self.triples[idx]

        q_enc = self.tokenizer(
            t["query"], max_length=self.max_len, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        p_enc = self.tokenizer(
            t["positive"], max_length=self.max_len, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        n_enc = self.tokenizer(
            t["negative"], max_length=self.max_len, truncation=True, padding="max_length",
            return_tensors="pt",
        )

        return {
            "query_input_ids": q_enc["input_ids"].squeeze(0),
            "query_attention_mask": q_enc["attention_mask"].squeeze(0),
            "positive_input_ids": p_enc["input_ids"].squeeze(0),
            "positive_attention_mask": p_enc["attention_mask"].squeeze(0),
            "negative_input_ids": n_enc["input_ids"].squeeze(0),
            "negative_attention_mask": n_enc["attention_mask"].squeeze(0),
            "relevance": torch.tensor(t["relevance"], dtype=torch.float32),
        }
