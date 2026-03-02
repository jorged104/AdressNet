"""
Reporte detallado de evaluación del Bi-LSTM-CRF.

Genera:
  1. F1 / Precision / Recall por entidad (STREET, ZONE, NUMBER, etc.)
  2. F1 por tipo de dirección (urban_grid, municipal, descriptive, rural)
  3. 5 ejemplos de error con predicción vs etiqueta real

Uso:
  uv run python -m src.model.eval_report
  uv run python -m src.model.eval_report --checkpoint models/best_model.pt --split test
  uv run python -m src.model.eval_report --n-failures 10
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from seqeval.metrics import classification_report

from src.model.dataset import MAX_CHAR_LEN, PAD_CHAR_ID, UNK_CHAR_ID, build_char_vocab
from src.model.model import BiLSTMCRF
from src.model.train import load_checkpoint
from src.utils.gazetteer import get_geo_feature_vector

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")

ENTITY_COLORS = {
    "STREET":       "cyan",
    "ZONE":         "magenta",
    "NUMBER":       "yellow",
    "MUNICIPALITY": "green",
    "DEPARTMENT":   "blue",
    "NEIGHBORHOOD": "bright_blue",
    "LANDMARK":     "bright_magenta",
    "REFERENCE":    "bright_cyan",
    "ADDRESS_TYPE": "bright_green",
    "REGION":       "bright_yellow",
    "COUNTRY":      "white",
}

app     = typer.Typer(name="eval-report", add_completion=False)
console = Console(highlight=False, width=120, legacy_windows=False)


# ---------------------------------------------------------------------------
# Lectura de CoNLL con metadatos
# ---------------------------------------------------------------------------

def read_conll_with_meta(path: Path) -> list[dict]:
    """
    Lee CoNLL (con comentarios # id: ... | type: ... | dirt: ...) y retorna
    lista de dicts con claves: id, type, dirt, text, tokens, labels.
    """
    sentences: list[dict] = []
    current:   dict       = {}
    tokens:    list[str]  = []
    labels:    list[str]  = []

    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if line.startswith("# id:"):
                m = re.match(r"# id: (\S+) \| type: (\S+) \| dirt: (\S+)", line)
                if m:
                    current["id"]   = m.group(1)
                    current["type"] = m.group(2)
                    current["dirt"] = m.group(3)

            elif line.startswith("# text:"):
                current["text"] = line[len("# text: "):]

            elif line == "":
                if tokens:
                    current["tokens"] = tokens
                    current["labels"] = labels
                    sentences.append(current)
                    current = {}
                    tokens  = []
                    labels  = []

            elif not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) == 2:
                    tokens.append(parts[0])
                    labels.append(parts[1])

    if tokens:
        current["tokens"] = tokens
        current["labels"] = labels
        sentences.append(current)

    return sentences


# ---------------------------------------------------------------------------
# Inferencia en lotes (sin DataLoader, para conservar metadatos)
# ---------------------------------------------------------------------------

def predict_all(
    model:       BiLSTMCRF,
    sentences:   list[dict],
    token_vocab: dict[str, int],
    char_vocab:  dict[str, int],
    id2label:    dict[int, str],
    device:      torch.device,
    batch_size:  int = 32,
) -> list[list[str]]:
    """
    Corre inferencia sobre todas las frases y retorna lista de listas de etiquetas.
    El orden está garantizado (sin shuffle).
    """
    pad_token_id = token_vocab.get("<PAD>", 0)
    unk_token_id = token_vocab.get("<UNK>", 1)

    all_preds: list[list[str]] = []
    model.eval()

    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i : i + batch_size]
        B = len(batch_sents)

        # --- Codificar tokens, caracteres y features del gazetteer ---
        token_seqs: list[list[int]]         = []
        char_seqs:  list[list[list[int]]]   = []
        geo_seqs:   list[list[list[float]]] = []

        for sent in batch_sents:
            tids:     list[int]         = []
            cids_all: list[list[int]]   = []
            gids_all: list[list[float]] = []
            for tok in sent["tokens"]:
                tok_l = tok.lower()
                tids.append(token_vocab.get(tok_l, unk_token_id))
                cids = [char_vocab.get(ch, UNK_CHAR_ID) for ch in tok[:MAX_CHAR_LEN]]
                cids_all.append(cids)
                gids_all.append(get_geo_feature_vector(tok))
            token_seqs.append(tids)
            char_seqs.append(cids_all)
            geo_seqs.append(gids_all)

        # --- Padding dinámico ---
        max_seq  = max(len(t) for t in token_seqs)
        max_char = max(
            max((len(c) for c in cids_all), default=1)
            for cids_all in char_seqs
        )

        token_tensor = torch.full((B, max_seq), pad_token_id, dtype=torch.long)
        char_tensor  = torch.full((B, max_seq, max_char), PAD_CHAR_ID, dtype=torch.long)
        geo_tensor   = torch.zeros((B, max_seq, 3), dtype=torch.float)

        for b, (tids, cids_all, gids_all) in enumerate(zip(token_seqs, char_seqs, geo_seqs)):
            token_tensor[b, : len(tids)] = torch.tensor(tids, dtype=torch.long)
            for s, cids in enumerate(cids_all):
                if cids:
                    char_tensor[b, s, : len(cids)] = torch.tensor(cids, dtype=torch.long)
            for s, geo in enumerate(gids_all):
                geo_tensor[b, s] = torch.tensor(geo, dtype=torch.float)

        mask = token_tensor != pad_token_id

        token_tensor = token_tensor.to(device)
        char_tensor  = char_tensor.to(device)
        geo_tensor   = geo_tensor.to(device)
        mask         = mask.to(device)

        with torch.no_grad():
            pred_ids: list[list[int]] = model(token_tensor, char_tensor, mask, geo_feats=geo_tensor)

        for b_idx, pids in enumerate(pred_ids):
            seq_len = int(mask[b_idx].sum().item())
            preds = [id2label.get(p, "O") for p in pids[:seq_len]]
            preds = ["O" if lbl == "PAD" else lbl for lbl in preds]
            all_preds.append(preds)

    return all_preds


# ---------------------------------------------------------------------------
# Reporte 1: métricas por entidad
# ---------------------------------------------------------------------------

def report_per_entity(
    all_true:  list[list[str]],
    all_pred:  list[list[str]],
) -> None:
    console.rule("[bold blue]1. Métricas por entidad[/bold blue]")

    report: dict = classification_report(
        all_true, all_pred,
        zero_division=0,
        output_dict=True,
        scheme=None,
        mode=None,
    )

    table = Table(
        title="Resultados por entidad (entity-level seqeval)",
        box=box.ROUNDED,
        header_style="bold white on dark_blue",
        show_lines=True,
    )
    table.add_column("Entidad",   style="bold", min_width=20)
    table.add_column("Precision", justify="right", min_width=10)
    table.add_column("Recall",    justify="right", min_width=10)
    table.add_column("F1",        justify="right", min_width=10)
    table.add_column("Support",   justify="right", min_width=10)
    table.add_column("F1 bar",    min_width=20)

    # Entidades individuales (excluir promedios agregados)
    aggregates = {"micro avg", "macro avg", "weighted avg"}
    entity_rows: list[tuple[str, dict]] = [
        (k, v) for k, v in report.items()
        if k not in aggregates and isinstance(v, dict)
    ]
    entity_rows.sort(key=lambda x: x[1].get("f1-score", 0), reverse=True)

    for entity, m in entity_rows:
        f1   = m.get("f1-score", 0.0)
        prec = m.get("precision", 0.0)
        rec  = m.get("recall", 0.0)
        sup  = int(m.get("support", 0))

        color = ENTITY_COLORS.get(entity, "white")
        bar_len = int(f1 * 20)
        bar = f"[{color}]{'█' * bar_len}{'░' * (20 - bar_len)}[/{color}]"

        table.add_row(
            f"[{color}]{entity}[/{color}]",
            f"{prec:.4f}",
            f"{rec:.4f}",
            f"[bold {'green' if f1 >= 0.85 else 'yellow' if f1 >= 0.70 else 'red'}]{f1:.4f}[/bold {'green' if f1 >= 0.85 else 'yellow' if f1 >= 0.70 else 'red'}]",
            str(sup),
            bar,
        )

    # Separador + promedios
    table.add_section()
    for avg_key in ("macro avg", "weighted avg"):
        if avg_key in report:
            m    = report[avg_key]
            f1   = m.get("f1-score", 0.0)
            prec = m.get("precision", 0.0)
            rec  = m.get("recall", 0.0)
            sup  = int(m.get("support", 0))
            bar_len = int(f1 * 20)
            bar = f"[white]{'█' * bar_len}{'░' * (20 - bar_len)}[/white]"
            table.add_row(
                f"[dim]{avg_key}[/dim]",
                f"{prec:.4f}",
                f"{rec:.4f}",
                f"[bold]{f1:.4f}[/bold]",
                str(sup),
                bar,
            )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Reporte 2: F1 por tipo de dirección
# ---------------------------------------------------------------------------

def report_per_type(
    sentences: list[dict],
    all_pred:  list[list[str]],
) -> None:
    console.rule("[bold blue]2. F1 por tipo de dirección[/bold blue]")

    # Agrupar por tipo
    groups: dict[str, tuple[list[list[str]], list[list[str]]]] = {}
    for sent, pred in zip(sentences, all_pred):
        addr_type = sent.get("type", "unknown")
        true      = sent["labels"]
        if addr_type not in groups:
            groups[addr_type] = ([], [])
        groups[addr_type][0].append(true)
        groups[addr_type][1].append(pred)

    table = Table(
        title="F1 por tipo de dirección",
        box=box.ROUNDED,
        header_style="bold white on dark_blue",
        show_lines=True,
    )
    table.add_column("Tipo",       style="bold cyan", min_width=20)
    table.add_column("Muestras",   justify="right",   min_width=10)
    table.add_column("Precision",  justify="right",   min_width=10)
    table.add_column("Recall",     justify="right",   min_width=10)
    table.add_column("F1",         justify="right",   min_width=10)
    table.add_column("F1 bar",     min_width=20)

    type_order = ["urban_grid", "municipal", "descriptive", "rural"]
    seen_types = list(groups.keys())
    ordered    = [t for t in type_order if t in seen_types]
    ordered   += [t for t in seen_types if t not in type_order]

    for addr_type in ordered:
        trues, preds = groups[addr_type]
        rep   = classification_report(
            trues, preds, zero_division=0, output_dict=True
        )
        wavg  = rep.get("weighted avg", {})
        f1    = wavg.get("f1-score", 0.0)
        prec  = wavg.get("precision", 0.0)
        rec   = wavg.get("recall", 0.0)
        n     = len(trues)

        bar_len = int(f1 * 20)
        color   = "green" if f1 >= 0.85 else "yellow" if f1 >= 0.70 else "red"
        bar     = f"[{color}]{'█' * bar_len}{'░' * (20 - bar_len)}[/{color}]"

        table.add_row(
            addr_type,
            str(n),
            f"{prec:.4f}",
            f"{rec:.4f}",
            f"[bold {color}]{f1:.4f}[/bold {color}]",
            bar,
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Reporte 3: ejemplos de error
# ---------------------------------------------------------------------------

def _label_color(label: str, is_error: bool) -> str:
    if is_error:
        return "bold red"
    if label == "O":
        return "dim"
    entity = label[2:] if label.startswith(("B-", "I-")) else label
    return ENTITY_COLORS.get(entity, "white")


def report_failures(
    sentences:  list[dict],
    all_pred:   list[list[str]],
    n_failures: int = 5,
) -> None:
    console.rule("[bold blue]3. Ejemplos de error[/bold blue]")

    failures: list[dict] = []
    for sent, pred in zip(sentences, all_pred):
        true = sent["labels"]
        # Asegurar misma longitud
        min_len = min(len(true), len(pred))
        errors  = sum(1 for t, p in zip(true[:min_len], pred[:min_len]) if t != p)
        if errors > 0:
            failures.append({**sent, "pred": pred, "n_errors": errors})

    console.print()
    console.print(
        f"  Oraciones con al menos un error: "
        f"[bold red]{len(failures)}[/bold red] / [cyan]{len(sentences)}[/cyan]  "
        f"([green]{100 * (len(sentences) - len(failures)) / len(sentences):.1f}%[/green] perfectas)\n"
    )

    # Ordenar por número de errores (los peores primero)
    failures.sort(key=lambda x: x["n_errors"], reverse=True)
    shown = failures[:n_failures]

    for idx, sent in enumerate(shown, 1):
        true = sent["labels"]
        pred = sent["pred"]
        toks = sent["tokens"]
        addr_type  = sent.get("type", "?")
        dirt_level = sent.get("dirt", "?")
        raw_text   = sent.get("text", " ".join(toks))
        n_err      = sent["n_errors"]

        panel_title = (
            f"[bold]Ejemplo {idx}[/bold]  "
            f"[cyan]{addr_type}[/cyan] / [yellow]{dirt_level}[/yellow]  "
            f"[red]({n_err} error{'es' if n_err > 1 else ''})[/red]"
        )

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
            padding=(0, 1),
            expand=False,
        )
        table.add_column("Token",    style="white",  min_width=18)
        table.add_column("Real",     min_width=22)
        table.add_column("Predicho", min_width=22)
        table.add_column("",        min_width=3)

        min_len = min(len(toks), len(true), len(pred))
        for tok, t_lbl, p_lbl in zip(toks[:min_len], true[:min_len], pred[:min_len]):
            is_err    = t_lbl != p_lbl
            err_mark  = "[bold red]✗[/bold red]" if is_err else "[green]✓[/green]"
            t_style   = _label_color(t_lbl, False)
            p_style   = _label_color(p_lbl, is_err)

            table.add_row(
                tok,
                f"[{t_style}]{t_lbl}[/{t_style}]",
                f"[{p_style}]{p_lbl}[/{p_style}]",
                err_mark,
            )

        console.print(
            Panel(
                table,
                title=panel_title,
                subtitle=f"[dim]\"{raw_text}\"[/dim]",
                border_style="red" if n_err > 3 else "yellow",
                expand=False,
                padding=(0, 1),
            )
        )


# ---------------------------------------------------------------------------
# Comando principal
# ---------------------------------------------------------------------------

@app.command()
def run(
    checkpoint: str = typer.Option(
        "models/best_model.pt", "--checkpoint", help="Ruta al checkpoint .pt"
    ),
    split: str = typer.Option(
        "test", "--split", help="Split a evaluar: train | val | test"
    ),
    n_failures: int = typer.Option(
        5, "--n-failures", help="Nº de ejemplos de error a mostrar"
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", help="Tamaño de batch para inferencia"
    ),
) -> None:
    """Genera el reporte detallado de evaluación del modelo entrenado."""

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint)

    if not ckpt_path.exists():
        console.print(f"[red]Checkpoint no encontrado: {ckpt_path}[/red]")
        raise typer.Exit(1)

    conll_path = PROCESSED_DIR / f"{split}.conll"
    if not conll_path.exists():
        console.print(f"[red]CoNLL no encontrado: {conll_path}[/red]")
        raise typer.Exit(1)

    # --- Encabezado ---
    console.rule(f"[bold magenta]Reporte de Evaluación — {split.upper()}[/bold magenta]")
    console.print(f"  Checkpoint : [cyan]{ckpt_path}[/cyan]")
    console.print(f"  Split      : [cyan]{split}[/cyan]")
    console.print(f"  Device     : [cyan]{device}[/cyan]\n")

    # --- Cargar modelo ---
    model, cfg, saved_metrics = load_checkpoint(ckpt_path, device)
    console.print(
        f"  Modelo cargado — epoch [cyan]{saved_metrics.get('epoch', '?')}[/cyan]  "
        f"val_f1=[bold green]{saved_metrics.get('f1', '?')}[/bold green]\n"
    )

    # --- Cargar vocabularios ---
    token_vocab: dict[str, int] = json.loads(
        (PROCESSED_DIR / "vocab.json").read_text(encoding="utf-8")
    )
    label_vocab: dict[str, int] = json.loads(
        (PROCESSED_DIR / "label2id.json").read_text(encoding="utf-8")
    )
    id2label: dict[int, str] = {v: k for k, v in label_vocab.items()}
    char_vocab = build_char_vocab(token_vocab)

    # --- Leer CoNLL con metadatos ---
    sentences = read_conll_with_meta(conll_path)
    console.print(f"  Oraciones cargadas: [cyan]{len(sentences)}[/cyan]\n")

    # --- Inferencia ---
    console.print("  [dim]Corriendo inferencia...[/dim]")
    all_pred = predict_all(
        model, sentences, token_vocab, char_vocab, id2label, device, batch_size
    )
    all_true = [sent["labels"] for sent in sentences]

    # --- Reportes ---
    report_per_entity(all_true, all_pred)
    report_per_type(sentences, all_pred)
    report_failures(sentences, all_pred, n_failures)

    console.rule("[bold green]Fin del reporte[/bold green]")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
