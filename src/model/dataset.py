"""
Lectura de CoNLL + PyTorch Dataset/DataLoader.

Responsabilidades:
  - Parsear archivos .conll a listas de (tokens, labels)
  - Mapear tokens → ids y labels → ids usando los vocabularios
  - Construir tensores de caracteres para la CharCNN
  - Extraer features del gazetteer (is_departamento, is_municipio, is_aldea)
  - Collate con padding dinámico por batch
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.utils.gazetteer import get_geo_feature_vector

# ---------------------------------------------------------------------------
# Constantes de tokens especiales
# ---------------------------------------------------------------------------

PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"
PAD_LABEL   = "PAD"

PAD_CHAR    = "<PAD>"
UNK_CHAR    = "<UNK>"
PAD_CHAR_ID = 0
UNK_CHAR_ID = 1

# Longitud máxima de un token en caracteres (tokens más largos se truncan)
MAX_CHAR_LEN = 30


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

class RawSentence(NamedTuple):
    tokens: list[str]
    labels: list[str]


class Batch(NamedTuple):
    """Un mini-batch listo para el modelo."""
    token_ids:  Tensor   # (B, seq_len)             — word embedding lookup
    char_ids:   Tensor   # (B, seq_len, max_char)   — CharCNN input
    label_ids:  Tensor   # (B, seq_len)             — targets
    mask:       Tensor   # (B, seq_len) bool         — True donde hay token real
    geo_feats:  Tensor   # (B, seq_len, 3) float    — [is_dept, is_mun, is_aldea]


# ---------------------------------------------------------------------------
# Parsing CoNLL
# ---------------------------------------------------------------------------

def read_conll(path: Path) -> list[RawSentence]:
    """
    Lee un archivo CoNLL y retorna lista de (tokens, labels).
    Ignora líneas de comentario (# ...) y usa líneas vacías como separador.
    """
    sentences: list[RawSentence] = []
    tokens:    list[str]         = []
    labels:    list[str]         = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if tokens:
                    sentences.append(RawSentence(tokens, labels))
                    tokens = []
                    labels = []
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    tok, lbl = parts
                    tokens.append(tok)
                    labels.append(lbl)

    if tokens:
        sentences.append(RawSentence(tokens, labels))

    return sentences


# ---------------------------------------------------------------------------
# Vocabulario de caracteres (construido en runtime desde el vocab de tokens)
# ---------------------------------------------------------------------------

def build_char_vocab(token_vocab: dict[str, int]) -> dict[str, int]:
    """
    Construye vocabulario de caracteres a partir del vocabulario de tokens.
    Incluye todos los caracteres únicos presentes en el vocab + especiales.
    """
    chars: set[str] = set()
    for token in token_vocab:
        if token not in (PAD_TOKEN, UNK_TOKEN):
            chars.update(token)

    char_vocab: dict[str, int] = {PAD_CHAR: PAD_CHAR_ID, UNK_CHAR: UNK_CHAR_ID}
    for ch in sorted(chars):
        if ch not in char_vocab:
            char_vocab[ch] = len(char_vocab)

    return char_vocab


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class NERDataset(Dataset[tuple[list[int], list[list[int]], list[int], list[list[float]]]]):
    """
    Dataset de NER que convierte frases CoNLL a tensores de ids.

    Cada item: (token_ids, char_ids_por_token, label_ids, geo_feats_por_token)
    geo_feats_por_token: lista de [is_dept, is_mun, is_aldea] (floats) por token
    """

    def __init__(
        self,
        sentences:  list[RawSentence],
        token_vocab: dict[str, int],
        label_vocab: dict[str, int],
        char_vocab:  dict[str, int],
    ) -> None:
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self.char_vocab  = char_vocab

        self.pad_token_id = token_vocab[PAD_TOKEN]
        self.unk_token_id = token_vocab[UNK_TOKEN]
        self.pad_label_id = label_vocab[PAD_LABEL]

        self.data = [self._encode(s) for s in sentences]

    def _encode(
        self, sentence: RawSentence
    ) -> tuple[list[int], list[list[int]], list[int], list[list[float]]]:
        token_ids: list[int]          = []
        char_ids:  list[list[int]]    = []
        label_ids: list[int]          = []
        geo_feats: list[list[float]]  = []

        for tok, lbl in zip(sentence.tokens, sentence.labels):
            tok_lower = tok.lower()
            token_ids.append(
                self.token_vocab.get(tok_lower, self.unk_token_id)
            )
            # Caracteres: truncar a MAX_CHAR_LEN
            cids = [
                self.char_vocab.get(ch, UNK_CHAR_ID)
                for ch in tok[:MAX_CHAR_LEN]
            ]
            char_ids.append(cids)
            label_ids.append(self.label_vocab.get(lbl, self.pad_label_id))
            # Features del gazetteer para este token
            geo_feats.append(get_geo_feature_vector(tok))

        return token_ids, char_ids, label_ids, geo_feats

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[list[int], list[list[int]], list[int], list[list[float]]]:
        return self.data[idx]


# ---------------------------------------------------------------------------
# Collate: padding dinámico por batch
# ---------------------------------------------------------------------------

def collate_fn(
    batch: list[tuple[list[int], list[list[int]], list[int], list[list[float]]]],
    pad_token_id: int,
    pad_label_id: int,
) -> Batch:
    """
    Agrupa items en un batch con padding dinámico (longitud del item más largo).

    token_ids : (B, max_seq)
    char_ids  : (B, max_seq, max_char_len)
    label_ids : (B, max_seq)
    mask      : (B, max_seq)  True = token real
    geo_feats : (B, max_seq, 3)  features del gazetteer (ceros en posiciones de padding)
    """
    token_seqs = [torch.tensor(t, dtype=torch.long) for t, _, _, _ in batch]
    label_seqs = [torch.tensor(l, dtype=torch.long) for _, _, l, _ in batch]

    token_ids_padded = pad_sequence(
        token_seqs, batch_first=True, padding_value=pad_token_id
    )
    label_ids_padded = pad_sequence(
        label_seqs, batch_first=True, padding_value=pad_label_id
    )
    mask = token_ids_padded != pad_token_id

    B       = len(batch)
    max_seq = token_ids_padded.size(1)

    # Char padding: (B, max_seq, max_char)
    max_char = max(
        max((len(cids) for cids in char_seq), default=1)
        for _, char_seq, _, _ in batch
    )
    char_tensor = torch.full(
        (B, max_seq, max_char), PAD_CHAR_ID, dtype=torch.long
    )
    for b_idx, (_, char_seq, _, _) in enumerate(batch):
        for s_idx, cids in enumerate(char_seq):
            length = len(cids)
            if length > 0:
                char_tensor[b_idx, s_idx, :length] = torch.tensor(cids, dtype=torch.long)

    # Geo features padding: (B, max_seq, 3)  — ceros en posiciones de padding
    geo_tensor = torch.zeros((B, max_seq, 3), dtype=torch.float)
    for b_idx, (_, _, _, geo_seq) in enumerate(batch):
        for s_idx, geo in enumerate(geo_seq):
            geo_tensor[b_idx, s_idx] = torch.tensor(geo, dtype=torch.float)

    return Batch(
        token_ids=token_ids_padded,
        char_ids=char_tensor,
        label_ids=label_ids_padded,
        mask=mask,
        geo_feats=geo_tensor,
    )


# ---------------------------------------------------------------------------
# Factory: crea los tres DataLoaders de una vez
# ---------------------------------------------------------------------------

class DataConfig(NamedTuple):
    token_vocab: dict[str, int]
    label_vocab: dict[str, int]
    char_vocab:  dict[str, int]
    id2label:    dict[int, str]
    num_labels:  int
    vocab_size:  int
    char_vocab_size: int
    pad_token_id:    int
    pad_label_id:    int


def load_data(
    processed_dir: Path,
    batch_size:    int = 32,
    num_workers:   int = 0,
) -> tuple[DataLoader[Batch], DataLoader[Batch], DataLoader[Batch], DataConfig]:
    """
    Lee vocabularios y CoNLL, construye Datasets y DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, config)
    """
    proc = Path(processed_dir)

    token_vocab: dict[str, int] = json.loads((proc / "vocab.json").read_text(encoding="utf-8"))
    label_vocab: dict[str, int] = json.loads((proc / "label2id.json").read_text(encoding="utf-8"))
    id2label:    dict[int, str] = {v: k for k, v in label_vocab.items()}
    char_vocab = build_char_vocab(token_vocab)

    pad_token_id = token_vocab[PAD_TOKEN]
    pad_label_id = label_vocab[PAD_LABEL]

    def _make_loader(split: str, shuffle: bool) -> DataLoader[Batch]:
        sentences = read_conll(proc / f"{split}.conll")
        ds = NERDataset(sentences, token_vocab, label_vocab, char_vocab)

        def _collate(b: list) -> Batch:
            return collate_fn(b, pad_token_id, pad_label_id)

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
            num_workers=num_workers,
            pin_memory=False,
        )

    train_loader = _make_loader("train", shuffle=True)
    val_loader   = _make_loader("val",   shuffle=False)
    test_loader  = _make_loader("test",  shuffle=False)

    config = DataConfig(
        token_vocab     = token_vocab,
        label_vocab     = label_vocab,
        char_vocab      = char_vocab,
        id2label        = id2label,
        num_labels      = len(label_vocab),
        vocab_size      = len(token_vocab),
        char_vocab_size = len(char_vocab),
        pad_token_id    = pad_token_id,
        pad_label_id    = pad_label_id,
    )

    return train_loader, val_loader, test_loader, config
