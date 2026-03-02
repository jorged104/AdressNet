"""
Generador de dataset sintético de direcciones guatemaltecas.

Proveedores soportados (configurar via .env):
  OpenAI   → LLM_PROVIDER=openai   | OPENAI_API_KEY=sk-...
  Ollama   → LLM_PROVIDER=ollama   | OLLAMA_BASE_URL=http://localhost:11434/v1

Uso:
    uv run python -m src.data_gen.generator generate --help
    uv run python -m src.data_gen.generator generate --per-type 40 --dirt-level medium
    uv run python -m src.data_gen.generator preview --address-type urban_grid
    uv run python -m src.data_gen.generator preview --address-type rural --dirt-level dirty
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from src.utils.prompts import build_prompt
from src.utils.schemas import AddressSample, AddressType, DirtLevel, GenerationBatch

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración de proveedor (leída desde .env)
# ---------------------------------------------------------------------------

_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}

MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(_PROVIDER, "gpt-4o-mini"))

BATCH_SIZE = 20
MAX_CONCURRENT = 5      # Semáforo: máximo de llamadas simultáneas al LLM
DATA_RAW_DIR = Path("data/raw")


def _build_client() -> AsyncOpenAI:
    """Construye el cliente async según el proveedor configurado en .env."""
    if _PROVIDER == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return AsyncOpenAI(base_url=base_url, api_key="ollama")
    return AsyncOpenAI()


app = typer.Typer(
    name="generate-data",
    help="Generador de dataset sintético de direcciones guatemaltecas.",
    add_completion=False,
)
console = Console(highlight=False, legacy_windows=False)


# ---------------------------------------------------------------------------
# Núcleo async: llamada al LLM + parsing
# ---------------------------------------------------------------------------

async def _call_llm_async(
    client: AsyncOpenAI,
    address_type: AddressType,
    dirt_level: DirtLevel,
    batch_size: int,
) -> list[AddressSample]:
    """
    Llama al LLM y parsea el resultado como lista de AddressSample.
    Reintenta hasta 3 veces si la respuesta no es JSON válido.
    """
    system_prompt, user_prompt = build_prompt(address_type, dirt_level, batch_size)

    last_error: Exception | None = None
    for attempt in range(3):
        response = await client.chat.completions.create(
            model=MODEL,
            # max_tokens=8192,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_text = (response.choices[0].message.content or "").strip()

        # Limpiar bloques markdown que el modelo a veces agrega
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            last_error = exc
            console.print(
                f"[yellow]Intento {attempt + 1}: JSON inválido, reintentando...[/yellow]"
            )
            continue

        samples: list[AddressSample] = []
        parse_errors = 0
        for item in data:
            if "id" not in item or not item["id"]:
                item["id"] = str(uuid.uuid4())
            try:
                samples.append(AddressSample.model_validate(item))
            except ValidationError:
                parse_errors += 1

        if parse_errors > 0:
            console.print(
                f"[dim]{parse_errors} muestras descartadas de {len(data)} en este batch.[/dim]"
            )

        return samples

    raise RuntimeError(
        f"No se pudo parsear la respuesta tras 3 intentos. Último error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Guardado en disco (sync — I/O rápido, no bloquea el event loop de forma notable)
# ---------------------------------------------------------------------------

def _save_batch(batch: GenerationBatch) -> Path:
    """Guarda un batch como JSONL en data/raw/."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    filename = (
        f"{batch.address_type.value}"
        f"_{batch.dirt_level.value}"
        f"_{timestamp}"
        f"_{batch.batch_id[:8]}.jsonl"
    )
    out_path = DATA_RAW_DIR / filename

    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "_batch": True,
                    "batch_id": batch.batch_id,
                    "model": batch.model,
                    "address_type": batch.address_type.value,
                    "dirt_level": batch.dirt_level.value,
                    "sample_count": len(batch.samples),
                    "generated_at": datetime.now().isoformat(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for sample in batch.samples:
            f.write(sample.model_dump_json() + "\n")

    return out_path


# ---------------------------------------------------------------------------
# Orquestación async con semáforo
# ---------------------------------------------------------------------------

async def _run_generate(
    types_to_run: list[AddressType],
    batches_per_type: int,
    dirt_level: DirtLevel,
) -> int:
    """
    Lanza todos los batches en paralelo con un semáforo de MAX_CONCURRENT
    llamadas simultáneas. Devuelve el total de muestras generadas.
    """
    client = _build_client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    total_batches = len(types_to_run) * batches_per_type

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"0/{total_batches} batches completados", total=total_batches
        )
        completed = 0

        async def _bounded(addr_type: AddressType) -> int:
            nonlocal completed
            async with sem:
                try:
                    samples = await _call_llm_async(
                        client, addr_type, dirt_level, BATCH_SIZE
                    )
                except Exception as exc:
                    progress.console.print(
                        f"[red]Error en batch {addr_type.value}: {exc}[/red]"
                    )
                    completed += 1
                    progress.update(
                        task,
                        advance=1,
                        description=f"{completed}/{total_batches} batches completados",
                    )
                    return 0

                batch = GenerationBatch(
                    batch_id=str(uuid.uuid4()),
                    model=MODEL,
                    address_type=addr_type,
                    dirt_level=dirt_level,
                    samples=samples,
                )
                out_path = _save_batch(batch)
                n = len(samples)

                completed += 1
                progress.console.print(
                    f"  [green]OK[/green] {addr_type.value}/{dirt_level.value}"
                    f" - {n} muestras -> [dim]{out_path.name}[/dim]"
                )
                progress.update(
                    task,
                    advance=1,
                    description=f"{completed}/{total_batches} batches completados",
                )
                return n

        # Lanza TODOS los batches a la vez; el semáforo limita la concurrencia
        all_tasks = [
            _bounded(addr_type)
            for addr_type in types_to_run
            for _ in range(batches_per_type)
        ]
        results = await asyncio.gather(*all_tasks)

    return sum(results)


# ---------------------------------------------------------------------------
# Comandos CLI
# ---------------------------------------------------------------------------

@app.command()
def generate(
    per_type: int = typer.Option(
        40,
        "--per-type",
        help="Número de muestras a generar por tipo de dirección.",
        min=BATCH_SIZE,
    ),
    dirt_level: DirtLevel = typer.Option(
        DirtLevel.MEDIUM,
        "--dirt-level",
        help="Nivel de 'suciedad' de las muestras generadas.",
    ),
    address_types: list[AddressType] = typer.Option(
        None,
        "--address-type",
        help="Tipos a generar (repite el flag para varios). Por defecto: todos.",
    ),
    concurrency: int = typer.Option(
        MAX_CONCURRENT,
        "--concurrency",
        help="Máximo de llamadas simultáneas al LLM.",
        min=1,
        max=20,
    ),
) -> None:
    """Genera el dataset sintético en paralelo y lo guarda en data/raw/."""
    global MAX_CONCURRENT
    MAX_CONCURRENT = concurrency

    types_to_run: list[AddressType] = address_types or list(AddressType)
    batches_per_type = max(1, per_type // BATCH_SIZE)
    total_batches = len(types_to_run) * batches_per_type

    console.rule("[bold blue]Generador de direcciones guatemaltecas[/bold blue]")
    console.print(
        f"Proveedor: [magenta]{_PROVIDER}[/magenta]  |  "
        f"Modelo: [magenta]{MODEL}[/magenta]  |  "
        f"Tipos: [cyan]{', '.join(t.value for t in types_to_run)}[/cyan]  |  "
        f"Nivel: [cyan]{dirt_level.value}[/cyan]  |  "
        f"Muestras objetivo: [cyan]{per_type * len(types_to_run)}[/cyan]  |  "
        f"Batches: [cyan]{total_batches}[/cyan]  |  "
        f"Concurrencia: [cyan]{concurrency}[/cyan]"
    )
    console.print()

    total = asyncio.run(_run_generate(types_to_run, batches_per_type, dirt_level))

    console.rule()
    console.print(
        f"[bold green]Dataset generado:[/bold green] "
        f"{total} muestras en [cyan]data/raw/[/cyan]"
    )


@app.command()
def preview(
    address_type: AddressType = typer.Option(
        AddressType.URBAN_GRID,
        "--address-type",
        help="Tipo de dirección a previsualizar.",
    ),
    dirt_level: DirtLevel = typer.Option(
        DirtLevel.MEDIUM,
        "--dirt-level",
        help="Nivel de suciedad.",
    ),
    n: int = typer.Option(5, "--n", help="Número de ejemplos a mostrar."),
) -> None:
    """Genera N ejemplos y los muestra en pantalla sin guardar en disco."""
    console.rule(
        f"[bold blue]Preview: {address_type.value} / {dirt_level.value}[/bold blue]"
    )
    console.print(
        f"Proveedor: [magenta]{_PROVIDER}[/magenta]  |  Modelo: [magenta]{MODEL}[/magenta]"
    )

    async def _run() -> list[AddressSample]:
        client = _build_client()
        return await _call_llm_async(client, address_type, dirt_level, max(n, 5))

    with console.status("Llamando al LLM..."):
        samples = asyncio.run(_run())

    samples = samples[:n]

    for i, sample in enumerate(samples, 1):
        console.print(
            Panel(
                f"[bold]{sample.raw_text}[/bold]",
                title=f"[cyan]#{i} - {sample.address_type.value} / {sample.dirt_level.value}[/cyan]",
                border_style="blue",
            )
        )

        for v_idx, variant in enumerate(sample.variants, 1):
            console.print(f"  [dim]Variante {v_idx}:[/dim] {variant}")

        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
        table.add_column("Token", style="cyan", no_wrap=True)
        table.add_column("Etiqueta", style="green")

        for tl in sample.tokens:
            style = "dim" if tl.label == "O" else ""
            table.add_row(tl.token, tl.label, style=style)

        console.print(table)

        if sample.metadata:
            console.print(
                "  [dim]Metadata:[/dim] "
                + "  ".join(f"[yellow]{k}[/yellow]={v}" for k, v in sample.metadata.items())
            )

        console.print()

    console.print("[bold]JSON del primer ejemplo:[/bold]")
    console.print(
        Syntax(
            samples[0].model_dump_json(indent=2),
            "json",
            theme="monokai",
            word_wrap=True,
        )
    )


@app.command()
def stats() -> None:
    """Muestra estadísticas del dataset generado en data/raw/."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_RAW_DIR.glob("*.jsonl"))

    if not files:
        console.print("[yellow]No hay archivos en data/raw/ aún.[/yellow]")
        raise typer.Exit()

    table = Table(title="Dataset en data/raw/", show_lines=True)
    table.add_column("Archivo", style="cyan")
    table.add_column("Tipo", style="green")
    table.add_column("Nivel", style="yellow")
    table.add_column("Muestras", justify="right")

    total = 0
    for f in files:
        count = 0
        addr_type = dirt = "?"
        with f.open(encoding="utf-8") as fh:
            for line_num, line in enumerate(fh):
                obj = json.loads(line)
                if line_num == 0 and obj.get("_batch"):
                    addr_type = obj.get("address_type", "?")
                    dirt = obj.get("dirt_level", "?")
                else:
                    count += 1
        table.add_row(f.name, addr_type, dirt, str(count))
        total += count

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {total} muestras en {len(files)} archivos")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
