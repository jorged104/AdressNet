"""Pydantic schemas compartidos en todo el proyecto."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AddressType(str, Enum):
    URBAN_GRID = "urban_grid"       # Ciudad de Guatemala — cuadrícula de calles y zonas
    MUNICIPAL = "municipal"          # Municipios del interior — barrios, cantones
    DESCRIPTIVE = "descriptive"     # Referencias a landmarks
    RURAL = "rural"                 # Aldeas, caseríos, kilómetros


class DirtLevel(str, Enum):
    CLEAN = "clean"     # Sin errores, bien formateada
    MEDIUM = "medium"   # Abreviaciones, capitalización irregular
    DIRTY = "dirty"     # Typos, mezcla de idiomas, componentes faltantes


# ---------------------------------------------------------------------------
# Etiquetas NER (esquema BIO)
# ---------------------------------------------------------------------------

NERTag = Literal[
    "O",
    "B-STREET", "I-STREET",
    "B-NUMBER", "I-NUMBER",
    "B-ZONE", "I-ZONE",
    "B-NEIGHBORHOOD", "I-NEIGHBORHOOD",
    "B-MUNICIPALITY", "I-MUNICIPALITY",
    "B-DEPARTMENT", "I-DEPARTMENT",
    "B-LANDMARK", "I-LANDMARK",
    "B-DIRECTION", "I-DIRECTION",
    "B-DISTANCE", "I-DISTANCE",
    "B-ADDRESS_TYPE", "I-ADDRESS_TYPE",
]


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------

class TokenLabel(BaseModel):
    """Par token → etiqueta BIO."""
    token: str
    label: NERTag


class AddressSample(BaseModel):
    """Una dirección con sus variantes y etiquetas."""
    id: str = Field(description="UUID único del ejemplo")
    address_type: AddressType
    dirt_level: DirtLevel
    raw_text: str = Field(description="Texto original limpio")
    variants: list[str] = Field(
        description="1-3 variantes con typos / abreviaciones / orden variable",
        min_length=1,
        max_length=3,
    )
    tokens: list[TokenLabel] = Field(
        description="Tokenización BIO del raw_text"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Municipio, departamento, zona, u otros datos estructurados extra",
    )


class GenerationBatch(BaseModel):
    """Resultado de una llamada al LLM: N muestras."""
    batch_id: str
    model: str
    address_type: AddressType
    dirt_level: DirtLevel
    samples: list[AddressSample]
