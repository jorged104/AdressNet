"""
Motor de inferencia para el modelo Bi-LSTM-CRF.

Encapsula la carga del modelo y toda la lógica de predicción, independiente
de FastAPI para facilitar tests y reutilización.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.model.dataset import MAX_CHAR_LEN, PAD_CHAR_ID, UNK_CHAR_ID, build_char_vocab
from src.model.model import BiLSTMCRF
from src.model.train import load_checkpoint
from src.utils.gazetteer import get_geo_feature_vector

PROCESSED_DIR = Path("data/processed")
MODEL_PATH    = Path("models/best_model.pt")


# ---------------------------------------------------------------------------
# Tipos de resultado
# ---------------------------------------------------------------------------

class TokenResult:
    """Predicción para un token individual."""
    __slots__ = ("token", "label", "geo")

    def __init__(self, token: str, label: str, geo: list[float]) -> None:
        self.token = token
        self.label = label
        self.geo   = geo  # [is_dept, is_mun, is_aldea]

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "label": self.label,
            "gazetteer": {
                "is_departamento": bool(self.geo[0]),
                "is_municipio":    bool(self.geo[1]),
                "is_aldea":        bool(self.geo[2]),
            },
        }


class ParseResult:
    """Resultado completo del parsing de una dirección."""
    __slots__ = ("address", "tokens", "entities", "structured")

    def __init__(
        self,
        address:    str,
        tokens:     list[TokenResult],
        entities:   list[dict],
        structured: dict[str, str],
    ) -> None:
        self.address    = address
        self.tokens     = tokens
        self.entities   = entities
        self.structured = structured

    def to_dict(self) -> dict:
        return {
            "address":    self.address,
            "tokens":     [t.to_dict() for t in self.tokens],
            "entities":   self.entities,
            "structured": self.structured,
        }


# ---------------------------------------------------------------------------
# Extracción de entidades desde etiquetas BIO
# ---------------------------------------------------------------------------

def _extract_entities(tokens: list[str], labels: list[str]) -> list[dict]:
    """
    Convierte secuencia BIO en spans de entidad.
    Retorna lista de {"text", "label", "start_token", "end_token"}.
    """
    entities: list[dict] = []
    current_tokens: list[str] = []
    current_label:  str | None = None
    start_idx: int = 0

    for i, (tok, lbl) in enumerate(zip(tokens, labels)):
        if lbl.startswith("B-"):
            if current_label is not None:
                entities.append({
                    "text":        " ".join(current_tokens),
                    "label":       current_label,
                    "start_token": start_idx,
                    "end_token":   i - 1,
                })
            current_tokens = [tok]
            current_label  = lbl[2:]
            start_idx      = i

        elif lbl.startswith("I-") and current_label == lbl[2:]:
            current_tokens.append(tok)

        else:
            # "O" o cambio de entidad inesperado
            if current_label is not None:
                entities.append({
                    "text":        " ".join(current_tokens),
                    "label":       current_label,
                    "start_token": start_idx,
                    "end_token":   i - 1,
                })
            current_tokens = []
            current_label  = None

    if current_label is not None:
        entities.append({
            "text":        " ".join(current_tokens),
            "label":       current_label,
            "start_token": start_idx,
            "end_token":   len(tokens) - 1,
        })

    return entities


def _build_structured(entities: list[dict]) -> dict[str, str]:
    """
    Colapsa las entidades en un dict {TIPO → texto}.
    Si hay múltiples tokens del mismo tipo, se concatenan con " | ".
    """
    result: dict[str, str] = {}
    for ent in entities:
        label = ent["label"]
        if label not in result:
            result[label] = ent["text"]
        else:
            result[label] += " | " + ent["text"]
    return result


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    Carga el modelo una vez y expone predict() para uso en la API.

    Uso:
        predictor = Predictor()
        result = predictor.predict("3a Calle 5-23 Zona 1 Guatemala")
    """

    def __init__(
        self,
        model_path:    Path = MODEL_PATH,
        processed_dir: Path = PROCESSED_DIR,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {model_path}")
        if not processed_dir.exists():
            raise FileNotFoundError(f"Directorio de datos no encontrado: {processed_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vocabularios
        self.token_vocab: dict[str, int] = json.loads(
            (processed_dir / "vocab.json").read_text(encoding="utf-8")
        )
        label_vocab: dict[str, int] = json.loads(
            (processed_dir / "label2id.json").read_text(encoding="utf-8")
        )
        self.id2label: dict[int, str] = {v: k for k, v in label_vocab.items()}
        self.char_vocab = build_char_vocab(self.token_vocab)

        self.pad_token_id = self.token_vocab.get("<PAD>", 0)
        self.unk_token_id = self.token_vocab.get("<UNK>", 1)

        # Modelo
        self.model: BiLSTMCRF
        self.model_cfg: dict
        self.model, self.model_cfg, self.saved_metrics = load_checkpoint(
            model_path, self.device
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Inferencia principal
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, address: str) -> ParseResult:
        """Parsea una dirección y retorna etiquetas + entidades."""
        tokens = address.split()
        if not tokens:
            return ParseResult(address, [], [], {})

        labels = self._run_model([tokens])[0]
        geo_vecs = [get_geo_feature_vector(tok) for tok in tokens]

        token_results = [
            TokenResult(tok, lbl, geo)
            for tok, lbl, geo in zip(tokens, labels, geo_vecs)
        ]
        entities  = _extract_entities(tokens, labels)
        structured = _build_structured(entities)

        return ParseResult(address, token_results, entities, structured)

    @torch.no_grad()
    def predict_batch(self, addresses: list[str]) -> list[ParseResult]:
        """Parsea una lista de direcciones en un solo forward pass."""
        token_lists = [a.split() for a in addresses]
        if not any(token_lists):
            return [ParseResult(addr, [], [], {}) for addr in addresses]

        all_labels = self._run_model(token_lists)

        results = []
        for address, tokens, labels in zip(addresses, token_lists, all_labels):
            if not tokens:
                results.append(ParseResult(address, [], [], {}))
                continue
            geo_vecs = [get_geo_feature_vector(tok) for tok in tokens]
            token_results = [
                TokenResult(tok, lbl, geo)
                for tok, lbl, geo in zip(tokens, labels, geo_vecs)
            ]
            entities   = _extract_entities(tokens, labels)
            structured = _build_structured(entities)
            results.append(ParseResult(address, token_results, entities, structured))

        return results

    # ------------------------------------------------------------------
    # Construcción de tensores e inferencia
    # ------------------------------------------------------------------

    def _run_model(self, token_lists: list[list[str]]) -> list[list[str]]:
        """
        Codifica, hace padding y corre el modelo para un batch de frases.
        Retorna lista de listas de etiquetas (ya decodificadas como strings).
        """
        B = len(token_lists)
        max_seq  = max(len(t) for t in token_lists) if token_lists else 1
        max_char = 1

        # Pre-computar ids
        token_seqs: list[list[int]]         = []
        char_seqs:  list[list[list[int]]]   = []
        geo_seqs:   list[list[list[float]]] = []

        for tokens in token_lists:
            tids, cids_all, gids_all = [], [], []
            for tok in tokens:
                tids.append(self.token_vocab.get(tok.lower(), self.unk_token_id))
                cids = [self.char_vocab.get(ch, UNK_CHAR_ID) for ch in tok[:MAX_CHAR_LEN]]
                cids_all.append(cids)
                gids_all.append(get_geo_feature_vector(tok))
                if len(cids) > max_char:
                    max_char = len(cids)
            token_seqs.append(tids)
            char_seqs.append(cids_all)
            geo_seqs.append(gids_all)

        # Tensores con padding
        token_tensor = torch.full((B, max_seq), self.pad_token_id, dtype=torch.long)
        char_tensor  = torch.full((B, max_seq, max_char), PAD_CHAR_ID, dtype=torch.long)
        geo_tensor   = torch.zeros((B, max_seq, 3), dtype=torch.float)

        for b, (tids, cids_all, gids_all) in enumerate(zip(token_seqs, char_seqs, geo_seqs)):
            n = len(tids)
            token_tensor[b, :n] = torch.tensor(tids, dtype=torch.long)
            for s, (cids, geo) in enumerate(zip(cids_all, gids_all)):
                if cids:
                    char_tensor[b, s, :len(cids)] = torch.tensor(cids, dtype=torch.long)
                geo_tensor[b, s] = torch.tensor(geo, dtype=torch.float)

        mask = token_tensor != self.pad_token_id

        pred_ids: list[list[int]] = self.model(
            token_tensor.to(self.device),
            char_tensor.to(self.device),
            mask.to(self.device),
            geo_feats=geo_tensor.to(self.device),
        )

        # Decodificar a strings (solo posiciones reales)
        all_labels: list[list[str]] = []
        for b, pids in enumerate(pred_ids):
            seq_len = int(mask[b].sum().item())
            labels  = [self.id2label.get(p, "O") for p in pids[:seq_len]]
            labels  = ["O" if lbl == "PAD" else lbl for lbl in labels]
            all_labels.append(labels)

        return all_labels

    # ------------------------------------------------------------------
    # Metadatos del modelo
    # ------------------------------------------------------------------

    @property
    def info(self) -> dict:
        return {
            "model":         "BiLSTMCRF",
            "use_gazetteer": self.model_cfg.get("use_gazetteer", False),
            "vocab_size":    self.model_cfg.get("vocab_size"),
            "num_labels":    self.model_cfg.get("num_labels"),
            "lstm_hidden":   self.model_cfg.get("lstm_hidden"),
            "lstm_layers":   self.model_cfg.get("lstm_layers"),
            "val_f1":        self.saved_metrics.get("f1"),
            "device":        str(self.device),
        }
