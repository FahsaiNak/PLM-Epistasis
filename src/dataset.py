"""
dataset.py
-----------
PyTorch Dataset for aligned HIV-1 protein sequences and continuous labels.
"""

from typing import List, Dict
import torch
from torch.utils.data import Dataset


class HIVSeqDataset(Dataset):
    """
    Dataset for protein sequences (possibly with gaps '-') and numeric labels.
    Automatically replaces gaps with a safe token (e.g. 'X').

    Parameters
    ----------
    sequences : list of str
        Protein sequences.
    labels : list of float
        Corresponding scalar regression labels.
    tokenizer : transformers.AutoTokenizer
        Tokenizer compatible with ProtBERT.
    max_len : int
        Maximum tokenized sequence length.
    """

    def __init__(self, sequences: List[str], labels: List[float],
                 tokenizer, max_len: int = 512):
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        seq_spaced = " ".join(list(seq))
        label = self.labels[idx]
        
        tokens = self.tokenizer(
            seq_spaced,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }
