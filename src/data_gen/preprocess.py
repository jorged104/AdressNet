"""
Pipeline de preprocesamiento: JSONL → CoNLL + vocabularios.

Lee todos los archivos de data/raw/, deduplica, valida, divide en
train/val/test de forma estratificada y escribe los artefactos que
el entrenador del modelo Bi-LSTM consumirá directamente.

Artefactos de salida (data/processed/):
  train.conll       — secuencias de entrenamiento
  val.conll         — secuencias de validación
  test.conll        — secuencias de evaluación final (no tocar durante el tuning)
  label2id.json     — {etiqueta: id_entero}
  id2label.json     — {id_entero: etiqueta}  (inverso)
  vocab.json        — {token_lowercase: id_entero}  (PAD=0, UNK=1, resto por freq)
  stats.json        — metadata del split (conteos, distribuciones)

Uso:
  uv run python -m src.data_gen.preprocess run
  uv run python -m src.data_gen.preprocess run --train-ratio 0.8 --val-ratio 0.1 --seed 42
  uv run python -m src.data_gen.preprocess verify
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.schemas import AddressSample, AddressType

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DATA_RAW_DIR  = Path("data/raw")
DATA_PROC_DIR = Path("data/processed")

# Tokens especiales reservados para el modelo
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Frecuencia mínima para entrar al vocabulario (tokens más raros → <UNK>)
MIN_TOKEN_FREQ = 2

app     = typer.Typer(name="preprocess", help="Pipeline JSONL → CoNLL.", add_completion=False)
console = Console(highlight=False, legacy_windows=False)


# ---------------------------------------------------------------------------
# Carga y deduplicación
# ---------------------------------------------------------------------------

def load_all_samples() -> list[AddressSample]:
    """
    Lee todos los archivos JSONL de data/raw/, omite la línea de metadata
    de batch (_batch=True) y deduplica por ID de muestra.
    """
    files = sorted(DATA_RAW_DIR.glob("*.jsonl"))
    if not files:
        console.print("[red]No se encontraron archivos en data/raw/[/red]")
        raise typer.Exit(1)

    seen_ids:  set[str]         = set()
    samples:   list[AddressSample] = []
    errors:    int               = 0
    skipped:   int               = 0

    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                obj = json.loads(raw_line)

                # Omitir la línea de metadata del batch
                if line_num == 0 and obj.get("_batch"):
                    continue

                # Deduplicar por ID
                sample_id = obj.get("id", "")
                if sample_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(sample_id)

                try:
                    samples.append(AddressSample.model_validate(obj))
                except ValidationError:
                    errors += 1

    console.print(
        f"Cargados [cyan]{len(samples)}[/cyan] samples  |  "
        f"[dim]Duplicados omitidos: {skipped}  |  Errores de schema: {errors}[/dim]"
    )
    return samples


# ---------------------------------------------------------------------------
# Validación BIO
# ---------------------------------------------------------------------------

def _check_bio(sample: AddressSample) -> list[str]:
    """
    Retorna lista de advertencias BIO para una muestra.
    Regla: I-X solo puede seguir a B-X o I-X del mismo tipo X.
    """
    warnings: list[str] = []
    prev_label = "O"

    for tl in sample.tokens:
        lbl = tl.label
        if lbl.startswith("I-"):
            entity = lbl[2:]
            expected_prev = {f"B-{entity}", f"I-{entity}"}
            if prev_label not in expected_prev:
                warnings.append(
                    f"'{tl.token}':{lbl} sin B- previo (anterior={prev_label})"
                )
        prev_label = lbl

    return warnings


# ---------------------------------------------------------------------------
# Corrección BIO
# ---------------------------------------------------------------------------

def _fix_bio(sample: AddressSample) -> tuple[AddressSample, int]:
    """
    Corrige errores BIO en una muestra mutando su lista de tokens:
      - I-X que abre una entidad (sin B-X/I-X previo del mismo tipo) → B-X
      - I-X que sigue a una entidad de tipo distinto (B-Y o I-Y)      → B-X

    Retorna la muestra corregida y el número de tokens modificados.

    Ejemplo:
      [I-STREET, I-STREET, O, I-ZONE]
       ↓ fix ↓
      [B-STREET, I-STREET, O, B-ZONE]   (2 correcciones)
    """
    fixed = 0
    prev_label = "O"
    new_tokens = []

    for tl in sample.tokens:
        lbl = tl.label
        if lbl.startswith("I-"):
            entity = lbl[2:]
            if prev_label not in {f"B-{entity}", f"I-{entity}"}:
                # I- huérfano → promover a B-
                lbl = f"B-{entity}"
                fixed += 1
        new_tokens.append(tl.model_copy(update={"label": lbl}))
        prev_label = lbl

    corrected = sample.model_copy(update={"tokens": new_tokens})
    return corrected, fixed


# ---------------------------------------------------------------------------
# Conversión a CoNLL
# ---------------------------------------------------------------------------

def sample_to_conll(sample: AddressSample) -> str:
    """
    Convierte una muestra al formato CoNLL con cabecera de comentario.

    Formato de salida:
        # id: <uuid> | type: <addr_type> | dirt: <level>
        # text: <raw_text>
        token1 \\t LABEL1
        token2 \\t LABEL2
        <línea en blanco>
    """
    header = (
        f"# id: {sample.id} | "
        f"type: {sample.address_type.value} | "
        f"dirt: {sample.dirt_level.value}\n"
        f"# text: {sample.raw_text}"
    )
    token_lines = "\n".join(f"{tl.token}\t{tl.label}" for tl in sample.tokens)
    return f"{header}\n{token_lines}\n"


# ---------------------------------------------------------------------------
# Split estratificado
# ---------------------------------------------------------------------------

def stratified_split(
    samples:     list[AddressSample],
    train_ratio: float,
    val_ratio:   float,
    seed:        int,
) -> tuple[list[AddressSample], list[AddressSample], list[AddressSample]]:
    """
    Divide el dataset manteniendo la proporción de cada address_type
    en los tres splits. Estratifica por tipo (no por tipo+nivel, para
    garantizar muestras suficientes en val/test con datasets pequeños).
    """
    by_type: dict[AddressType, list[AddressSample]] = defaultdict(list)
    for s in samples:
        by_type[s.address_type].append(s)

    rng   = random.Random(seed)
    train: list[AddressSample] = []
    val:   list[AddressSample] = []
    test:  list[AddressSample] = []

    for bucket in by_type.values():
        rng.shuffle(bucket)
        n       = len(bucket)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    # Re-mezclar cada split para que no estén agrupados por tipo
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# Construcción de vocabularios
# ---------------------------------------------------------------------------

def build_label_vocab(samples: list[AddressSample]) -> dict[str, int]:
    """
    Construye label2id con orden determinista:
      0 → PAD  (padding de secuencias)
      1 → O    (token fuera de entidad — más frecuente)
      2..N → etiquetas BIO ordenadas alfabéticamente
    """
    all_labels: set[str] = set()
    for s in samples:
        for tl in s.tokens:
            all_labels.add(tl.label)

    # O siempre en el slot 1; PAD en el 0
    sorted_labels = sorted(all_labels - {"O"})
    ordered = ["PAD", "O"] + sorted_labels
    return {lbl: idx for idx, lbl in enumerate(ordered)}


def build_token_vocab(
    samples:  list[AddressSample],
    min_freq: int = MIN_TOKEN_FREQ,
) -> dict[str, int]:
    """
    Construye vocab lowercase, frecuencia mínima MIN_TOKEN_FREQ.
      0 → <PAD>
      1 → <UNK>
      2..N → tokens por frecuencia descendente
    """
    freq: Counter[str] = Counter()
    for s in samples:
        for tl in s.tokens:
            freq[tl.token.lower()] += 1

    vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in freq.most_common():
        if count < min_freq:
            break
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


# ---------------------------------------------------------------------------
# Escritura de artefactos
# ---------------------------------------------------------------------------

def write_conll(samples: list[AddressSample], path: Path) -> int:
    """Escribe el archivo CoNLL; retorna el número de tokens escritos."""
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    token_count = 0
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(sample_to_conll(s))
            f.write("\n")   # blank line entre oraciones
            token_count += len(s.tokens)
    return token_count


def _build_stats(
    train: list[AddressSample],
    val:   list[AddressSample],
    test:  list[AddressSample],
    label2id: dict[str, int],
    vocab:    dict[str, int],
) -> dict:
    """Construye el diccionario stats.json."""
    def split_info(samples: list[AddressSample]) -> dict:
        type_counts:  Counter[str] = Counter(s.address_type.value  for s in samples)
        level_counts: Counter[str] = Counter(s.dirt_level.value    for s in samples)
        label_counts: Counter[str] = Counter(
            tl.label for s in samples for tl in s.tokens
        )
        return {
            "samples":      len(samples),
            "tokens":       sum(len(s.tokens) for s in samples),
            "by_type":      dict(type_counts),
            "by_dirt":      dict(level_counts),
            "label_counts": dict(label_counts.most_common()),
        }

    return {
        "train":      split_info(train),
        "val":        split_info(val),
        "test":       split_info(test),
        "vocab_size": len(vocab),
        "num_labels": len(label2id),
        "labels":     list(label2id.keys()),
    }


# ---------------------------------------------------------------------------
# Comandos CLI
# ---------------------------------------------------------------------------

@app.command()
def run(
    train_ratio: float = typer.Option(0.8,  "--train-ratio", help="Proporción para train."),
    val_ratio:   float = typer.Option(0.1,  "--val-ratio",   help="Proporción para val."),
    seed:        int   = typer.Option(42,   "--seed",        help="Semilla aleatoria."),
    min_freq:    int   = typer.Option(MIN_TOKEN_FREQ, "--min-freq",
                                      help="Frecuencia mínima para incluir token en vocab."),
    strict_bio:  bool  = typer.Option(False, "--strict-bio",
                                      help="Abortar si hay errores BIO (por defecto solo advierte)."),
) -> None:
    """Ejecuta el pipeline completo JSONL -> CoNLL + vocabularios."""
    if train_ratio + val_ratio >= 1.0:
        console.print("[red]train_ratio + val_ratio debe ser < 1.0[/red]")
        raise typer.Exit(1)

    console.rule("[bold blue]Preprocesamiento JSONL -> CoNLL[/bold blue]")

    # 1. Carga -----------------------------------------------------------
    with console.status("Cargando muestras..."):
        samples = load_all_samples()

    # 2. Validación BIO --------------------------------------------------
    console.print("\n[bold]Validando consistencia BIO...[/bold]")
    bio_warnings: list[str] = []
    bad_samples: list[str]  = []
    for s in samples:
        w = _check_bio(s)
        if w:
            bio_warnings.extend(w)
            bad_samples.append(s.id)

    if bio_warnings:
        console.print(
            f"  [yellow]Advertencias BIO: {len(bio_warnings)} en "
            f"{len(bad_samples)} muestras[/yellow]"
        )
        if strict_bio:
            console.print("[red]--strict-bio activo, abortando.[/red]")
            raise typer.Exit(1)

        # Corrección automática: I-X inicial → B-X
        console.print("  Aplicando corrección BIO (I-X inicial -> B-X)...")
        total_fixes = 0
        fixed_samples: list[AddressSample] = []
        for s in samples:
            corrected, n_fixed = _fix_bio(s)
            fixed_samples.append(corrected)
            total_fixes += n_fixed
        samples = fixed_samples
        console.print(f"  [green]Corregidos {total_fixes} tokens en {len(bad_samples)} muestras.[/green]")
    else:
        console.print("  [green]Sin errores BIO.[/green]")

    # 3. Split -----------------------------------------------------------
    console.print("\n[bold]Dividiendo dataset...[/bold]")
    train, val, test = stratified_split(samples, train_ratio, val_ratio, seed)
    console.print(
        f"  Train: [cyan]{len(train)}[/cyan]  |  "
        f"Val: [cyan]{len(val)}[/cyan]  |  "
        f"Test: [cyan]{len(test)}[/cyan]"
    )

    # 4. Vocabularios (solo con train para evitar data leakage) ----------
    console.print("\n[bold]Construyendo vocabularios...[/bold]")
    label2id = build_label_vocab(samples)    # etiquetas: todos los splits
    id2label  = {str(v): k for k, v in label2id.items()}
    vocab     = build_token_vocab(train, min_freq=min_freq)
    console.print(
        f"  Etiquetas: [cyan]{len(label2id)}[/cyan]  |  "
        f"Vocab: [cyan]{len(vocab)}[/cyan] tokens (min_freq={min_freq})"
    )

    # 5. Escritura -------------------------------------------------------
    console.print("\n[bold]Escribiendo artefactos...[/bold]")
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)

    train_tokens = write_conll(train, DATA_PROC_DIR / "train.conll")
    val_tokens   = write_conll(val,   DATA_PROC_DIR / "val.conll")
    test_tokens  = write_conll(test,  DATA_PROC_DIR / "test.conll")

    (DATA_PROC_DIR / "label2id.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA_PROC_DIR / "id2label.json").write_text(
        json.dumps(id2label, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA_PROC_DIR / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats = _build_stats(train, val, test, label2id, vocab)
    (DATA_PROC_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for name, path, n_tok in [
        ("train.conll", DATA_PROC_DIR / "train.conll", train_tokens),
        ("val.conll",   DATA_PROC_DIR / "val.conll",   val_tokens),
        ("test.conll",  DATA_PROC_DIR / "test.conll",  test_tokens),
    ]:
        console.print(f"  [green]OK[/green] {name}  ({n_tok} tokens)")
    console.print(f"  [green]OK[/green] label2id.json  id2label.json  vocab.json  stats.json")

    # 6. Resumen ---------------------------------------------------------
    _print_summary(train, val, test, label2id, vocab)


def _print_summary(
    train:    list[AddressSample],
    val:      list[AddressSample],
    test:     list[AddressSample],
    label2id: dict[str, int],
    vocab:    dict[str, int],
) -> None:
    """Imprime tablas de resumen del pipeline."""
    console.rule("[bold blue]Resumen[/bold blue]")

    # --- Tabla de splits ---
    split_table = Table(title="Splits", show_lines=True)
    split_table.add_column("Split",   style="cyan")
    split_table.add_column("Muestras", justify="right")
    split_table.add_column("Tokens",   justify="right")
    for name, bucket in [("train", train), ("val", val), ("test", test)]:
        split_table.add_row(
            name,
            str(len(bucket)),
            str(sum(len(s.tokens) for s in bucket)),
        )

    # --- Tabla de distribución por tipo ---
    type_table = Table(title="Tipos por split", show_lines=True)
    type_table.add_column("Tipo", style="green")
    type_table.add_column("Train", justify="right")
    type_table.add_column("Val",   justify="right")
    type_table.add_column("Test",  justify="right")

    all_types = sorted({s.address_type.value for s in train + val + test})
    for t in all_types:
        type_table.add_row(
            t,
            str(sum(1 for s in train if s.address_type.value == t)),
            str(sum(1 for s in val   if s.address_type.value == t)),
            str(sum(1 for s in test  if s.address_type.value == t)),
        )

    # --- Tabla de etiquetas ---
    lbl_table = Table(title=f"Etiquetas ({len(label2id)})", show_lines=False)
    lbl_table.add_column("ID",      justify="right", style="dim")
    lbl_table.add_column("Etiqueta", style="yellow")
    for lbl, idx in sorted(label2id.items(), key=lambda x: x[1]):
        lbl_table.add_row(str(idx), lbl)

    console.print(Columns([split_table, type_table, lbl_table]))
    console.print(
        Panel(
            f"Vocab: [cyan]{len(vocab)}[/cyan] tokens  |  "
            f"Etiquetas: [cyan]{len(label2id)}[/cyan]  |  "
            f"Artefactos en: [cyan]data/processed/[/cyan]",
            border_style="green",
        )
    )


@app.command()
def verify() -> None:
    """Verifica los archivos CoNLL y muestra distribución de etiquetas."""
    files = {
        "train": DATA_PROC_DIR / "train.conll",
        "val":   DATA_PROC_DIR / "val.conll",
        "test":  DATA_PROC_DIR / "test.conll",
    }

    missing = [n for n, p in files.items() if not p.exists()]
    if missing:
        console.print(f"[red]Faltan archivos: {missing}. Ejecuta 'run' primero.[/red]")
        raise typer.Exit(1)

    console.rule("[bold blue]Verificación CoNLL[/bold blue]")

    for split_name, path in files.items():
        label_counts: Counter[str] = Counter()
        n_sentences = 0
        n_tokens    = 0
        bio_errors  = 0
        prev_lbl    = "O"

        with path.open(encoding="utf-8") as f:
            in_sentence = False
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("#"):
                    continue
                if line == "":
                    if in_sentence:
                        n_sentences += 1
                        prev_lbl = "O"
                    in_sentence = False
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                _, lbl = parts
                in_sentence = True
                n_tokens += 1
                label_counts[lbl] += 1

                # BIO check
                if lbl.startswith("I-"):
                    entity = lbl[2:]
                    if prev_lbl not in {f"B-{entity}", f"I-{entity}"}:
                        bio_errors += 1
                prev_lbl = lbl

        table = Table(title=f"{split_name} ({n_sentences} oraciones, {n_tokens} tokens)", show_lines=False)
        table.add_column("Etiqueta", style="yellow")
        table.add_column("Count",    justify="right")
        table.add_column("%",        justify="right", style="dim")
        for lbl, count in label_counts.most_common():
            pct = f"{count / n_tokens * 100:.1f}"
            table.add_row(lbl, str(count), pct)
        console.print(table)

        status = "[green]OK[/green]" if bio_errors == 0 else f"[red]{bio_errors} errores BIO[/red]"
        console.print(f"  Integridad BIO: {status}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
