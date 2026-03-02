"""
Loop de entrenamiento + evaluación para el Bi-LSTM-CRF.

Uso:
  uv run python -m src.model.train fit
  uv run python -m src.model.train fit --epochs 50 --batch-size 32 --fasttext path/cc.es.300.vec
  uv run python -m src.model.train evaluate --checkpoint models/best_model.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.optim as optim
import typer
from seqeval.metrics import classification_report, f1_score
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from src.model.dataset import Batch, load_data
from src.model.model import BiLSTMCRF, build_model

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")

app     = typer.Typer(name="train", help="Entrenamiento del Bi-LSTM-CRF.", add_completion=False)
console = Console(highlight=False, legacy_windows=False)


# ---------------------------------------------------------------------------
# Evaluación con seqeval (entity-level F1)
# ---------------------------------------------------------------------------

def evaluate(
    model:    BiLSTMCRF,
    loader:   object,
    id2label: dict[int, str],
    device:   torch.device,
    split:    str = "val",
) -> dict[str, float]:
    """
    Corre inferencia sobre un DataLoader y calcula métricas seqeval.
    Retorna dict con keys: f1, precision, recall.
    """
    model.eval()
    all_preds:  list[list[str]] = []
    all_labels: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            batch: Batch
            token_ids = batch.token_ids.to(device)
            char_ids  = batch.char_ids.to(device)
            mask      = batch.mask.to(device)
            geo_feats = batch.geo_feats.to(device)
            label_ids = batch.label_ids  # CPU — solo para comparar

            pred_ids: list[list[int]] = model(token_ids, char_ids, mask, geo_feats=geo_feats)

            for b_idx, preds in enumerate(pred_ids):
                seq_len = mask[b_idx].sum().item()
                true_ids = label_ids[b_idx, :int(seq_len)].tolist()

                pred_labels = [id2label.get(p, "O") for p in preds[:int(seq_len)]]
                true_labels = [id2label.get(t, "O") for t in true_ids]

                # seqeval ignora "PAD" si se filtra aquí
                pred_labels = ["O" if l == "PAD" else l for l in pred_labels]
                true_labels = ["O" if l == "PAD" else l for l in true_labels]

                all_preds.append(pred_labels)
                all_labels.append(true_labels)

    f1        = f1_score(all_labels, all_preds, zero_division=0)
    report    = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)

    # Extraer precision/recall del report agregado
    precision = report.get("weighted avg", {}).get("precision", 0.0)
    recall    = report.get("weighted avg", {}).get("recall", 0.0)

    return {
        "split":     split,
        "f1":        round(f1, 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
    }


# ---------------------------------------------------------------------------
# Guardado / carga de checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     BiLSTMCRF,
    metrics:   dict[str, float],
    epoch:     int,
    cfg_dict:  dict,
    path:      Path,
) -> None:
    """Guarda modelo + configuración + métricas en un único .pt."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":       epoch,
            "state_dict":  model.state_dict(),
            "metrics":     metrics,
            "model_cfg":   cfg_dict,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[BiLSTMCRF, dict, dict]:
    """
    Carga modelo desde checkpoint.
    Retorna (model, cfg_dict, metrics).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)  # nosec: checkpoint propio
    cfg  = ckpt["model_cfg"]

    model = build_model(
        vocab_size      = cfg["vocab_size"],
        char_vocab_size = cfg["char_vocab_size"],
        num_labels      = cfg["num_labels"],
        pad_token_id    = cfg["pad_token_id"],
        pad_label_id    = cfg["pad_label_id"],
        word_emb_dim    = cfg.get("word_emb_dim", 300),
        lstm_hidden     = cfg.get("lstm_hidden", 256),
        lstm_layers     = cfg.get("lstm_layers", 2),
        dropout         = cfg.get("dropout", 0.5),
        use_gazetteer   = cfg.get("use_gazetteer", False),  # False para checkpoints antiguos
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, cfg, ckpt["metrics"]


# ---------------------------------------------------------------------------
# Comando: fit
# ---------------------------------------------------------------------------

@app.command()
def fit(
    epochs:        int   = typer.Option(30,    "--epochs",        help="Épocas de entrenamiento."),
    batch_size:    int   = typer.Option(32,    "--batch-size",    help="Tamaño del batch."),
    lr:            float = typer.Option(1e-3,  "--lr",            help="Learning rate inicial."),
    lstm_hidden:   int   = typer.Option(256,   "--lstm-hidden",   help="Dim hidden del Bi-LSTM."),
    lstm_layers:   int   = typer.Option(2,     "--lstm-layers",   help="Capas del Bi-LSTM."),
    dropout:       float = typer.Option(0.5,   "--dropout",       help="Dropout rate."),
    patience:      int   = typer.Option(7,     "--patience",      help="Early stopping patience."),
    fasttext:      str   = typer.Option("",    "--fasttext",      help="Ruta al archivo .vec de fastText."),
    word_emb_dim:  int   = typer.Option(300,   "--word-emb-dim",  help="Dim word embedding."),
    use_gazetteer: bool  = typer.Option(True,  "--use-gazetteer", help="Activar features del gazetteer INE."),
) -> None:
    """Entrena el modelo Bi-LSTM-CRF y guarda el mejor checkpoint por val F1."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.rule("[bold blue]Entrenamiento Bi-LSTM-CRF[/bold blue]")
    console.print(
        f"Device: [magenta]{device}[/magenta]  |  "
        f"Épocas: [cyan]{epochs}[/cyan]  |  "
        f"Batch: [cyan]{batch_size}[/cyan]  |  "
        f"LR: [cyan]{lr}[/cyan]  |  "
        f"Patience: [cyan]{patience}[/cyan]"
    )

    # --- Datos ---
    console.print("\n[bold]Cargando datos...[/bold]")
    train_loader, val_loader, _, data_cfg = load_data(PROCESSED_DIR, batch_size=batch_size)
    console.print(
        f"  Train batches: [cyan]{len(train_loader)}[/cyan]  |  "
        f"Val batches: [cyan]{len(val_loader)}[/cyan]  |  "
        f"Vocab: [cyan]{data_cfg.vocab_size}[/cyan]  |  "
        f"Labels: [cyan]{data_cfg.num_labels}[/cyan]  |  "
        f"Chars: [cyan]{data_cfg.char_vocab_size}[/cyan]"
    )

    # --- Modelo ---
    console.print("\n[bold]Construyendo modelo...[/bold]")
    model = build_model(
        vocab_size      = data_cfg.vocab_size,
        char_vocab_size = data_cfg.char_vocab_size,
        num_labels      = data_cfg.num_labels,
        pad_token_id    = data_cfg.pad_token_id,
        pad_label_id    = data_cfg.pad_label_id,
        word_emb_dim    = word_emb_dim,
        lstm_hidden     = lstm_hidden,
        lstm_layers     = lstm_layers,
        dropout         = dropout,
        use_gazetteer   = use_gazetteer,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Parámetros totales: [cyan]{n_params:,}[/cyan]")

    # --- fastText ---
    if fasttext and Path(fasttext).exists():
        console.print(f"\n[bold]Cargando fastText desde {fasttext}...[/bold]")
        hits = model.load_fasttext_embeddings(fasttext, data_cfg.token_vocab)
        console.print(f"  Embeddings cargados: [cyan]{hits}[/cyan] tokens")
    elif fasttext:
        console.print(f"[yellow]Archivo fastText no encontrado: {fasttext}. Usando init aleatoria.[/yellow]")

    # --- Optimizador + scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # --- Loop de entrenamiento ---
    best_f1     = 0.0
    best_epoch  = 0
    patience_ct = 0
    history:    list[dict] = []

    cfg_dict = {
        "vocab_size":      data_cfg.vocab_size,
        "char_vocab_size": data_cfg.char_vocab_size,
        "num_labels":      data_cfg.num_labels,
        "pad_token_id":    data_cfg.pad_token_id,
        "pad_label_id":    data_cfg.pad_label_id,
        "word_emb_dim":    word_emb_dim,
        "lstm_hidden":     lstm_hidden,
        "lstm_layers":     lstm_layers,
        "dropout":         dropout,
        "use_gazetteer":   use_gazetteer,
    }

    console.print()
    for epoch in range(1, epochs + 1):
        # ---------- Train ----------
        model.train()
        total_loss   = 0.0
        n_batches    = 0
        t0           = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn(f"Época [cyan]{epoch}/{epochs}[/cyan]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[dim]{task.fields[loss]:.4f}[/dim]"),
            console=console,
            transient=True,
        ) as prog:
            task = prog.add_task("train", total=len(train_loader), loss=0.0)

            for batch in train_loader:
                batch: Batch
                token_ids = batch.token_ids.to(device)
                char_ids  = batch.char_ids.to(device)
                mask      = batch.mask.to(device)
                geo_feats = batch.geo_feats.to(device)
                label_ids = batch.label_ids.to(device)

                optimizer.zero_grad()
                loss: torch.Tensor = model(token_ids, char_ids, mask, label_ids, geo_feats=geo_feats)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches  += 1
                prog.update(task, advance=1, loss=total_loss / n_batches)

        avg_loss = total_loss / max(n_batches, 1)
        elapsed  = time.time() - t0

        # ---------- Validación ----------
        val_metrics = evaluate(model, val_loader, data_cfg.id2label, device, "val")
        val_f1      = val_metrics["f1"]
        scheduler.step(val_f1)

        row = {
            "epoch":  epoch,
            "loss":   round(avg_loss, 4),
            **val_metrics,
            "time_s": round(elapsed, 1),
        }
        history.append(row)

        flag = ""
        if val_f1 > best_f1:
            best_f1      = val_f1
            best_epoch   = epoch
            patience_ct  = 0
            flag         = "  [green]<-- best[/green]"
            save_checkpoint(
                model, val_metrics, epoch, cfg_dict,
                MODELS_DIR / "best_model.pt",
            )
        else:
            patience_ct += 1

        console.print(
            f"  E{epoch:03d} | loss={avg_loss:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} | "
            f"{elapsed:.1f}s{flag}"
        )

        if patience_ct >= patience:
            console.print(f"\n[yellow]Early stopping: {patience} épocas sin mejora.[/yellow]")
            break

    # --- Resumen ---
    console.rule()
    console.print(
        f"[bold green]Entrenamiento completo.[/bold green]  "
        f"Mejor val F1: [cyan]{best_f1:.4f}[/cyan] en época [cyan]{best_epoch}[/cyan]"
    )
    console.print(f"Checkpoint guardado en [cyan]models/best_model.pt[/cyan]")

    # Guardar historial
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Comando: evaluate
# ---------------------------------------------------------------------------

@app.command()
def evaluate_cmd(
    checkpoint: str = typer.Option("models/best_model.pt", "--checkpoint", help="Ruta al .pt"),
    split:      str = typer.Option("test", "--split", help="Split a evaluar: train|val|test"),
    batch_size: int = typer.Option(32, "--batch-size"),
) -> None:
    """Evalúa un checkpoint guardado sobre train/val/test y muestra reporte."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint)

    if not ckpt_path.exists():
        console.print(f"[red]Checkpoint no encontrado: {ckpt_path}[/red]")
        raise typer.Exit(1)

    console.rule(f"[bold blue]Evaluación — {split}[/bold blue]")

    model, cfg, saved_metrics = load_checkpoint(ckpt_path, device)
    _, val_loader, test_loader, data_cfg = load_data(PROCESSED_DIR, batch_size=batch_size)
    train_loader, _, _, _ = load_data(PROCESSED_DIR, batch_size=batch_size)

    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    if split not in loader_map:
        console.print(f"[red]Split inválido: {split}. Usa train|val|test.[/red]")
        raise typer.Exit(1)

    metrics = evaluate(model, loader_map[split], data_cfg.id2label, device, split)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", justify="right", style="green")
    for k, v in metrics.items():
        table.add_row(str(k), str(v))
    console.print(table)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
