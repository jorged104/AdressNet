# 🇬🇹 GT Address Parser

> Modelo NER para parsear direcciones guatemaltecas — construido desde cero con Bi-LSTM + CRF y datos sintéticos.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model F1](https://img.shields.io/badge/F1-0.8656-orange.svg)]()
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://docs.astral.sh/uv/)

---

## La historia detrás de esto

Las direcciones guatemaltecas son un caos — y lo digo en el mejor sentido.

Una misma dirección en la Ciudad de Guatemala puede escribirse como `"4a Calle 5-15, Zona 10"`, `"4ta. Cll. 5-15 z.10"`, o simplemente `"frente a Portales, zona 10"` y las tres significan exactamente lo mismo. Añadí municipios del interior con direcciones descriptivas como `"100m al norte del parque central, Cobán"`, áreas rurales con `"Aldea El Rancho, km 84 ruta al Atlántico"`, y tenés uno de los sistemas de direcciones más diversos lingüísticamente de Centroamérica.

Este es el **v2** de un proyecto que empecé hace unos años usando una RNN simple. Esa versión fallaba — no porque la arquitectura estuviera mal, sino porque los datos de entrenamiento eran pocos y poco diversos. El modelo memorizaba en lugar de aprender.

Esta vez hice las cosas diferente:

- **Generación de datos sintéticos** usando un LLM para crear miles de ejemplos etiquetados y diversos — cubriendo cuadrículas urbanas, municipios, referencias descriptivas y direcciones rurales
- **Bi-LSTM + CharCNN + CRF** entrenado desde cero, sin transfer learning, sin el peso de un modelo BERT de 400MB
- **Gazetteer del INE** (Instituto Nacional de Estadística de Guatemala) integrado como features explícitas para resolver ambigüedades como *"Jalapa"*, que es municipio Y departamento al mismo tiempo
- **~3M parámetros** que corre en <5ms en CPU — viable para producción real

---

## Qué hace

Dada una dirección guatemalteca en texto libre, GT Address Parser extrae sus componentes estructurados:

```
POST /parse
{ "address": "3a Avenida 4-56 Zona 1, Guatemala" }
```

```json
{
  "structured": {
    "STREET":       "3a Avenida",
    "NUMBER":       "4-56",
    "ZONE":         "Zona 1",
    "MUNICIPALITY": "Guatemala"
  },
  "geolocation": {
    "lat": 14.6349,
    "lon": -90.5069,
    "precision": "exact",
    "source": "nominatim",
    "name": "3a Avenida 4-56 Zona 1, Guatemala"
  }
}
```

Maneja typos, abreviaciones, componentes faltantes y orden variable — porque las direcciones del mundo real no siguen ninguna regla.

---

## Entidades extraídas

| Etiqueta | Descripción | Ejemplo |
|---|---|---|
| `STREET` | Calle o avenida | `3a Avenida`, `Calle Real` |
| `NUMBER` | Número de casa o local | `4-56`, `No. 12` |
| `ZONE` | Zona urbana (Ciudad de Guatemala) | `Zona 10`, `Z-1` |
| `NEIGHBORHOOD` | Colonia, barrio, cantón, aldea | `Col. El Mirador`, `Aldea San Juan` |
| `MUNICIPALITY` | Municipio | `Mixco`, `Jalapa` |
| `DEPARTMENT` | Departamento | `Guatemala`, `Jalapa` |
| `LANDMARK` | Punto de referencia | `frente al mercado` |
| `DIRECTION` | Cardinal o dirección relativa | `norte`, `a la derecha` |
| `DISTANCE` | Distancia o kilómetro | `km 15`, `200 metros` |
| `ADDRESS_TYPE` | Indicador de tipo | `Municipio de`, `Departamento de` |

El esquema de etiquetado es **BIO** (Beginning-Inside-Outside).

---

## Arquitectura del modelo

```
Token de entrada
    │
    ├─► Word Embedding (300-dim)
    │
    ├─► CharCNN
    │     └─ Conv1D ×3 (kernels 2,3,4 · 30 filtros c/u) → 90-dim
    │
    └─► Features del Gazetteer INE
          └─ [is_departamento, is_municipio, is_aldea] → 3-dim
    │
    ▼
Concat (393-dim)
    │
    ▼
Bi-LSTM (2 capas · 256 hidden/dir → 512 output)
    │
    ▼
Linear (512 → 22 etiquetas)
    │
    ▼
CRF (decodificación Viterbi)
```

El CharCNN captura patrones morfológicos (abreviaciones, formatos de número como `5-23`). Las features del gazetteer le dan al modelo conocimiento geográfico explícito — resolviendo la ambigüedad clásica donde `"Jalapa"` puede ser municipio o departamento según el contexto.

**Parámetros totales:** ~3.25M  
**Tamaño del checkpoint:** 13 MB

---

## Resultados

| Entidad | F1 |
|---|---|
| ZONE | 0.9878 |
| STREET | 0.9442 |
| NUMBER | 0.8871 |
| DIRECTION | 0.8841 |
| DEPARTMENT | 0.8834 |
| MUNICIPALITY | 0.8722 |
| DISTANCE | 0.8592 |
| ADDRESS_TYPE | 0.7863 |
| NEIGHBORHOOD | 0.7805 |
| LANDMARK | 0.7699 |
| **macro avg** | **0.8656** |

| Tipo de dirección | F1 |
|---|---|
| `urban_grid` | 0.9626 ✅ |
| `descriptive` | 0.9025 ✅ |
| `municipal` | 0.8668 ✅ |
| `rural` | 0.6686 🚧 |

Las direcciones urbanas y descriptivas funcionan muy bien en producción. La cobertura rural está en mejora continua — contribuciones bienvenidas.

---

## Primeros pasos

Requiere **Python ≥ 3.11** y [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yourusername/gt-address-parser
cd gt-address-parser
uv sync
```

### Levantar la API

```bash
uv run uvicorn src.api.app:app --reload
```

Luego abrí `http://localhost:8000/docs` para la documentación interactiva.

### Parsear una dirección

```bash
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{"address": "Km 15.5 carretera a Jalapa, aldea El Terrero"}'
```

### Parseo en batch (hasta 50 direcciones)

```bash
curl -X POST http://localhost:8000/parse/batch \
  -H "Content-Type: application/json" \
  -d '{
    "addresses": [
      "3a Calle 5-23 Zona 10",
      "Cantón Los Pinos, San Marcos",
      "Aldea El Rancho, km 84 ruta al Atlántico"
    ]
  }'
```

---

## Entrenár tu propio modelo

### 1. Configurar el proveedor LLM

```bash
cp .env.example .env
# Agregá tu OPENAI_API_KEY o configurá Ollama
```

### 2. Generar datos sintéticos

```bash
# Generar 500 ejemplos por tipo de dirección
uv run generate-data generate --per-type 500 --dirt-level medium

# Ver ejemplos antes de generar
uv run generate-data preview --address-type urban_grid
```

### 3. Preprocesar

```bash
uv run preprocess run --train-ratio 0.8 --val-ratio 0.1
uv run preprocess verify
```

### 4. Entrenar

```bash
uv run train-ner fit --epochs 30 --batch-size 32 --lr 1e-3

# Con embeddings fastText preentrenados (recomendado)
uv run train-ner fit --fasttext cc.es.300.vec --epochs 50 --patience 10
```

### 5. Evaluar

```bash
uv run eval-report --split test --n-failures 10
```

---

## Estructura del proyecto

```
gt-address-parser/
├── src/
│   ├── api/
│   │   ├── app.py          # Rutas y schemas FastAPI
│   │   └── predictor.py    # Motor de inferencia
│   ├── data_gen/
│   │   ├── generator.py    # Generación sintética con LLM (OpenAI / Ollama)
│   │   └── preprocess.py   # JSONL → CoNLL + vocabularios
│   ├── model/
│   │   ├── model.py        # Arquitectura BiLSTMCRF + CharCNN
│   │   ├── dataset.py      # NERDataset y carga de datos
│   │   ├── train.py        # Loop de entrenamiento + CLI
│   │   └── eval_report.py  # Reporte detallado de evaluación
│   └── utils/
│       ├── gazetteer.py    # Gazetteer INE (22 departamentos + municipios)
│       ├── schemas.py      # Schemas Pydantic
│       └── prompts.py      # Prompts LLM para generación de datos
├── data/
│   ├── raw/                # JSONL generados por LLM (gitignored)
│   └── processed/          # CoNLL + vocabularios (gitignored)
├── models/
│   └── best_model.pt       # Checkpoint entrenado (gitignored)
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

---

## El Gazetteer del INE

`src/utils/gazetteer.py` contiene el listado oficial del INE con los 22 departamentos de Guatemala y todos sus municipios. Para cada token calcula:

```python
from src.utils.gazetteer import get_geo_features

get_geo_features("Jalapa")
# {"is_departamento": True, "is_municipio": True, "is_aldea": False}

get_geo_features("Mixco")
# {"is_departamento": False, "is_municipio": True, "is_aldea": False}
```

El matching es **insensible a mayúsculas y tildes**. El vector `[is_dept, is_mun, is_aldea]` se concatena al embedding de cada token antes del BiLSTM — dándole al modelo una señal explícita para resolver ambigüedades geográficas.

---

## Contribuir

Este proyecto fue construido pensando en Guatemala pero el enfoque generaliza a cualquier país con formatos de dirección no estándar. Las contribuciones son bienvenidas, especialmente:

- Más datos de entrenamiento para direcciones de comunidades rurales e indígenas
- Soporte para direcciones en idiomas mayas (K'iche', Kaqchikel, Mam, Q'eqchi')
- Integración con geocodificación
- Fine-tuning con datasets de direcciones reales

Abrí un issue o mandá un PR.

---

## Licencia

MIT — hacé lo que quieras, solo dale crédito.

---
