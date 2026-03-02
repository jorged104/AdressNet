"""
Arquitectura Bi-LSTM + CharCNN + CRF para NER de direcciones guatemaltecas.

Flujo por token:
  caracteres → CharCNN → vec_char (90-dim)
  token      → Embedding → vec_word (300-dim)
  concat([vec_char, vec_word]) → Dropout → Bi-LSTM → Dropout → Linear → CRF
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchcrf import CRF


# ---------------------------------------------------------------------------
# Configuración del modelo (dataclass en lugar de argparse directamente)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Vocabularios
    vocab_size:      int = 1051     # incluyendo PAD y UNK
    char_vocab_size: int = 90       # ~70 chars únicos + 2 especiales
    num_labels:      int = 22       # etiquetas BIO + PAD

    # Word embedding
    word_emb_dim:    int = 300      # coincide con fastText
    word_emb_freeze: bool = False   # fine-tune durante entrenamiento

    # Char CNN
    char_emb_dim:    int = 30
    char_kernels:    tuple[int, ...] = (2, 3, 4)  # tamaños de kernel
    char_filters:    int = 30       # filtros por kernel → total=90

    # Bi-LSTM
    lstm_hidden:     int = 256      # por dirección → 512 total
    lstm_layers:     int = 2
    lstm_dropout:    float = 0.5    # entre capas LSTM (solo si layers > 1)

    # Regularización
    dropout:         float = 0.5

    # Ids especiales
    pad_token_id:    int = 0
    pad_label_id:    int = 0

    # Gazetteer (features externas del INE)
    use_gazetteer:   bool = True    # concatenar [is_dept, is_mun, is_aldea] al input del LSTM
    geo_feat_dim:    int  = 3       # dimensión del vector de features geográficas


# ---------------------------------------------------------------------------
# Módulo CharCNN
# ---------------------------------------------------------------------------

class CharCNN(nn.Module):
    """
    Codifica un token a partir de sus caracteres usando convoluciones paralelas.

    Input:  (B * seq_len, max_char)   — ids de caracteres
    Output: (B * seq_len, total_filters)  — representación por token
    """

    def __init__(self, char_vocab_size: int, emb_dim: int, kernels: tuple[int, ...], filters: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, emb_dim, padding_idx=0)
        # Una Conv1D + ReLU + MaxPool por cada tamaño de kernel
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=emb_dim, out_channels=filters, kernel_size=k, padding=k // 2),
                nn.ReLU(),
            )
            for k in kernels
        ])
        self.output_dim = filters * len(kernels)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        char_ids: (B, seq_len, max_char)
        Returns:  (B, seq_len, output_dim)
        """
        B, seq_len, max_char = char_ids.shape

        # Aplanar batch+seq para procesar todos los tokens de una vez
        x = char_ids.view(B * seq_len, max_char)           # (B*S, max_char)
        x = self.embedding(x)                               # (B*S, max_char, emb_dim)
        x = x.transpose(1, 2)                               # (B*S, emb_dim, max_char)

        # Convolución + max-over-time pooling por cada kernel
        pooled = []
        for conv in self.convs:
            c = conv(x)                                     # (B*S, filters, *)
            c = c.max(dim=2).values                         # (B*S, filters)
            pooled.append(c)

        out = torch.cat(pooled, dim=1)                      # (B*S, total_filters)
        return out.view(B, seq_len, self.output_dim)        # (B, S, total_filters)


# ---------------------------------------------------------------------------
# Modelo principal
# ---------------------------------------------------------------------------

class BiLSTMCRF(nn.Module):
    """
    Bi-LSTM + CharCNN + CRF para NER.

    Parámetros totales estimados:
      CharCNN:       ~90K
      Word Emb:      ~315K  (1051 × 300)
      Bi-LSTM:       ~2.8M  (2 capas, 390→512→512)
      Linear:        ~11K   (512 → 22)
      CRF:           ~484   (22 × 22 transiciones)
      Total:         ~3.2M
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Char encoder
        self.char_cnn = CharCNN(
            char_vocab_size=cfg.char_vocab_size,
            emb_dim=cfg.char_emb_dim,
            kernels=cfg.char_kernels,
            filters=cfg.char_filters,
        )

        # Word embedding (pre-entrenado o random)
        self.word_embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.word_emb_dim,
            padding_idx=cfg.pad_token_id,
        )

        # Dropout de entrada
        self.input_dropout = nn.Dropout(cfg.dropout)

        # Bi-LSTM
        # Input = word_emb + char_cnn [+ geo_feats si use_gazetteer]
        lstm_input_dim = cfg.word_emb_dim + self.char_cnn.output_dim
        if cfg.use_gazetteer:
            lstm_input_dim += cfg.geo_feat_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
        )

        # Dropout de salida
        self.output_dropout = nn.Dropout(cfg.dropout)

        # Proyección a espacio de etiquetas
        self.classifier = nn.Linear(cfg.lstm_hidden * 2, cfg.num_labels)

        # CRF  (no incluye PAD en las transiciones — se maneja con mask)
        self.crf = CRF(cfg.num_labels, batch_first=True)

    # ------------------------------------------------------------------
    # Forward: emissions + (opcionalmente) loss
    # ------------------------------------------------------------------

    def _get_emissions(
        self,
        token_ids: torch.Tensor,                    # (B, seq_len)
        char_ids:  torch.Tensor,                    # (B, seq_len, max_char)
        mask:      torch.Tensor,                    # (B, seq_len)  bool
        geo_feats: torch.Tensor | None = None,      # (B, seq_len, 3) float
    ) -> torch.Tensor:
        """Retorna emission scores: (B, seq_len, num_labels)."""
        word_emb  = self.word_embedding(token_ids)       # (B, S, word_emb_dim)
        char_emb  = self.char_cnn(char_ids)              # (B, S, char_out_dim)
        parts = [word_emb, char_emb]
        if self.cfg.use_gazetteer and geo_feats is not None:
            parts.append(geo_feats.float())              # (B, S, 3)
        x = torch.cat(parts, dim=-1)                     # (B, S, word+char[+3])
        x = self.input_dropout(x)

        # Empaquetar secuencias para eficiencia (ignora padding en LSTM)
        lengths = mask.sum(dim=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out  = self.output_dropout(lstm_out)
        emissions = self.classifier(lstm_out)             # (B, S, num_labels)
        return emissions

    def forward(
        self,
        token_ids: torch.Tensor,
        char_ids:  torch.Tensor,
        mask:      torch.Tensor,
        label_ids: torch.Tensor | None = None,
        geo_feats: torch.Tensor | None = None,   # (B, seq_len, 3)
    ) -> torch.Tensor | list[list[int]]:
        """
        Entrenamiento (label_ids != None): retorna loss escalar (neglog-likelihood CRF).
        Inferencia  (label_ids is None):   retorna lista de listas con ids predichos.
        """
        emissions = self._get_emissions(token_ids, char_ids, mask, geo_feats)

        if label_ids is not None:
            # CRF espera que el PAD tenga label_id fuera del rango o se maneje
            # con mask; usamos mask para ignorar posiciones de padding
            loss = -self.crf(emissions, label_ids, mask=mask, reduction="mean")
            return loss

        return self.crf.decode(emissions, mask=mask)

    # ------------------------------------------------------------------
    # Carga de fastText
    # ------------------------------------------------------------------

    @torch.no_grad()
    def load_fasttext_embeddings(
        self,
        ft_path: str,
        token_vocab: dict[str, int],
        verbose: bool = True,
    ) -> int:
        """
        Inicializa self.word_embedding con vectores fastText.

        ft_path: ruta al archivo .vec (formato texto: "token dim1 dim2 ...")
        Retorna el número de tokens del vocabulario que encontró en fastText.
        """
        import io

        hits = 0
        with io.open(ft_path, "r", encoding="utf-8", newline="\n", errors="ignore") as f:
            n_vecs, dim = map(int, f.readline().split())
            if dim != self.cfg.word_emb_dim:
                raise ValueError(
                    f"fastText dim={dim} no coincide con word_emb_dim={self.cfg.word_emb_dim}"
                )
            for line in f:
                parts = line.rstrip().split(" ")
                token = parts[0].lower()
                if token in token_vocab:
                    idx = token_vocab[token]
                    vec = torch.tensor([float(v) for v in parts[1:]], dtype=torch.float)
                    self.word_embedding.weight[idx] = vec
                    hits += 1

        if verbose:
            coverage = hits / max(len(token_vocab) - 2, 1) * 100  # -2 por PAD y UNK
            print(f"fastText: {hits}/{len(token_vocab)-2} tokens cubiertos ({coverage:.1f}%)")

        return hits


# ---------------------------------------------------------------------------
# Factory con configuración por defecto
# ---------------------------------------------------------------------------

def build_model(
    vocab_size:      int,
    char_vocab_size: int,
    num_labels:      int,
    pad_token_id:    int = 0,
    pad_label_id:    int = 0,
    word_emb_dim:    int = 300,
    lstm_hidden:     int = 256,
    lstm_layers:     int = 2,
    dropout:         float = 0.5,
    use_gazetteer:   bool = True,
) -> BiLSTMCRF:
    """Construye el modelo con la configuración dada."""
    cfg = ModelConfig(
        vocab_size      = vocab_size,
        char_vocab_size = char_vocab_size,
        num_labels      = num_labels,
        pad_token_id    = pad_token_id,
        pad_label_id    = pad_label_id,
        word_emb_dim    = word_emb_dim,
        lstm_hidden     = lstm_hidden,
        lstm_layers     = lstm_layers,
        dropout         = dropout,
        use_gazetteer   = use_gazetteer,
    )
    return BiLSTMCRF(cfg)
