"""
FastAPI para el modelo de NER de direcciones guatemaltecas.

Endpoints:
  GET  /             — Info de la API
  GET  /health       — Estado del modelo
  POST /parse        — Parsear una dirección
  POST /parse/batch  — Parsear hasta 50 direcciones en un request

Uso:
  uv run api-server                        # Puerto 8000 por defecto
  uv run api-server --port 9000 --reload
  uv run uvicorn src.api.app:app --reload  # Modo desarrollo
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.predictor import Predictor

# ---------------------------------------------------------------------------
# Schemas Pydantic de request / response
# ---------------------------------------------------------------------------

class ParseRequest(BaseModel):
    address: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=["3a Avenida 4-56 Zona 1, Guatemala"],
        description="Dirección guatemalteca en texto libre.",
    )


class BatchRequest(BaseModel):
    addresses: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        examples=[["3a Avenida 4-56 Zona 1", "Aldea San Juan, Jalapa"]],
        description="Lista de hasta 50 direcciones.",
    )


class GazetteerInfo(BaseModel):
    is_departamento: bool
    is_municipio:    bool
    is_aldea:        bool


class TokenResponse(BaseModel):
    token:     str
    label:     str
    gazetteer: GazetteerInfo


class EntityResponse(BaseModel):
    text:        str
    label:       str
    start_token: int
    end_token:   int


class ParseResponse(BaseModel):
    address:    str
    tokens:     list[TokenResponse]
    entities:   list[EntityResponse]
    structured: dict[str, str] = Field(
        description="Tipo de entidad → texto extraído (STREET, ZONE, MUNICIPALITY, etc.)"
    )


class HealthResponse(BaseModel):
    status:        str
    model:         str
    use_gazetteer: bool
    val_f1:        float | None
    device:        str
    vocab_size:    int | None
    num_labels:    int | None


# ---------------------------------------------------------------------------
# Estado global del predictor (singleton)
# ---------------------------------------------------------------------------

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar; libera recursos al apagar."""
    global _predictor
    try:
        _predictor = Predictor()
    except FileNotFoundError as exc:
        # Arrancar de todas formas; los endpoints retornarán 503
        print(f"[WARN] No se pudo cargar el modelo: {exc}")
        _predictor = None
    yield
    _predictor = None


def _get_predictor() -> Predictor:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Verifica que models/best_model.pt existe.",
        )
    return _predictor


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GT Address Parser",
    description=(
        "API de NER para parsing de direcciones guatemaltecas. "
        "Modelo: Bi-LSTM + CharCNN + CRF con features del gazetteer INE."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@app.get("/", summary="Info de la API")
def root() -> dict:
    """Información básica y links útiles."""
    return {
        "name":    "GT Address Parser",
        "version": "0.1.0",
        "docs":    "/docs",
        "health":  "/health",
        "parse":   "/parse",
    }


@app.get("/health", response_model=HealthResponse, summary="Estado del modelo")
def health() -> HealthResponse:
    """Verifica que el modelo está cargado y listo."""
    pred = _get_predictor()
    info = pred.info
    return HealthResponse(
        status        = "ok",
        model         = info["model"],
        use_gazetteer = info["use_gazetteer"],
        val_f1        = info.get("val_f1"),
        device        = info["device"],
        vocab_size    = info.get("vocab_size"),
        num_labels    = info.get("num_labels"),
    )


@app.post("/parse", response_model=ParseResponse, summary="Parsear una dirección")
def parse(req: ParseRequest) -> ParseResponse:
    """
    Aplica NER a una dirección guatemalteca.

    Retorna los tokens con sus etiquetas BIO, las entidades extraídas
    y un diccionario estructurado con los campos de la dirección.

    **Entidades posibles:** STREET, NUMBER, ZONE, NEIGHBORHOOD, MUNICIPALITY,
    DEPARTMENT, LANDMARK, DIRECTION, DISTANCE, ADDRESS_TYPE.
    """
    pred   = _get_predictor()
    result = pred.predict(req.address)
    return _to_response(result)


@app.post(
    "/parse/batch",
    response_model=list[ParseResponse],
    summary="Parsear múltiples direcciones",
)
def parse_batch(req: BatchRequest) -> list[ParseResponse]:
    """
    Parsea una lista de hasta 50 direcciones en un único forward pass.
    Más eficiente que llamar a `/parse` múltiples veces.
    """
    pred    = _get_predictor()
    results = pred.predict_batch(req.addresses)
    return [_to_response(r) for r in results]


# ---------------------------------------------------------------------------
# Helpers de conversión resultado → schema
# ---------------------------------------------------------------------------

def _to_response(result) -> ParseResponse:
    tokens = [
        TokenResponse(
            token=t.token,
            label=t.label,
            gazetteer=GazetteerInfo(
                is_departamento=t.geo[0] > 0,
                is_municipio=t.geo[1] > 0,
                is_aldea=t.geo[2] > 0,
            ),
        )
        for t in result.tokens
    ]
    entities = [
        EntityResponse(
            text=e["text"],
            label=e["label"],
            start_token=e["start_token"],
            end_token=e["end_token"],
        )
        for e in result.entities
    ]
    return ParseResponse(
        address=result.address,
        tokens=tokens,
        entities=entities,
        structured=result.structured,
    )


# ---------------------------------------------------------------------------
# Entrypoint CLI (script `api-server`)
# ---------------------------------------------------------------------------

_cli = typer.Typer(name="api-server", add_completion=False)


@_cli.command()
def serve(
    host:   str  = typer.Option("0.0.0.0", "--host",   help="Host donde escucha el servidor."),
    port:   int  = typer.Option(8000,      "--port",   help="Puerto TCP."),
    reload: bool = typer.Option(False,     "--reload", help="Hot-reload (solo desarrollo)."),
    workers: int = typer.Option(1,         "--workers", help="Número de workers Uvicorn."),
) -> None:
    """Arranca el servidor FastAPI con Uvicorn."""
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level="info",
    )


if __name__ == "__main__":
    _cli()
