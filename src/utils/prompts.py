"""Prompts del sistema para generación de datos sintéticos."""

from src.utils.schemas import AddressType, DirtLevel

# ---------------------------------------------------------------------------
# Contexto base — se incluye en todos los prompts
# ---------------------------------------------------------------------------

_BASE_CONTEXT = """
Eres un experto en geografía guatemalteca y sistemas de direcciones.
Guatemala usa varios sistemas de direcciones simultáneamente:

1. CUADRÍCULA URBANA (Ciudad de Guatemala y cabeceras grandes):
   - Calles van de Este a Oeste (numeradas), Avenidas de Norte a Sur
   - Formato: "[N]a Calle [NUM]-[NUM], Zona [N], [Colonia/Sector]"
   - Ejemplo: "4a Calle 5-15, Zona 10" → calle 4, entre avenida 5 y 6, puerta 15
   - Zonas del 1 al 25 en Ciudad de Guatemala; municipios usan "Zona 1" genérico
   - Variantes: "4 calle", "4ta calle", "cuarta calle", "C4"

2. DIRECCIONES MUNICIPALES (interior del país):
   - Barrios, colonias, cantones, sectores, lotificaciones
   - Pueden incluir número de casa o sin número
   - Ejemplo: "Barrio El Centro, 3a Calle, Casa 12, Jalapa"
   - Cantones en municipios rurales: "Cantón Las Flores, Aldea San José"

3. DESCRIPCIÓN POR LANDMARKS (muy común en todo el país):
   - Distancia + dirección + punto de referencia
   - Puntos cardinales: norte/sur/este/oeste o n/s/e/o
   - Unidades: metros (m), cuadras, kilómetros (km), pasos
   - Ejemplo: "100m al norte del parque central, frente a la iglesia"
   - Variantes: "una cuadra arriba del mercado", "frente al Palacio Municipal"

4. RURAL (aldeas, caseríos, fincas):
   - Referencia a km de carretera + aldea/caserío + municipio
   - Ejemplo: "Aldea San Pedro, km 45 Carretera al Atlántico, Teculután, Zacapa"
   - Caseríos: sub-división de aldeas
   - Fincas: propiedades privadas con nombre propio

DEPARTAMENTOS Y MUNICIPIOS REPRESENTATIVOS:
- Guatemala: Guatemala, Mixco, Villa Nueva, San Miguel Petapa, Chinautla
- Quetzaltenango: Quetzaltenango (Xela), Salcajá, Cantel, Zunil
- Sacatepéquez: Antigua Guatemala, Jocotenango, Ciudad Vieja, Alotenango
- Chiquimula: Chiquimula, Esquipulas, Jocotán, Camotán
- Alta Verapaz: Cobán, Tactic, Panzós, Lanquín
- Petén: Flores, San Benito, Santa Elena, Melchor de Mencos
- San Marcos: San Marcos, Malacatán, Tajumulco, Sipacapa
- Huehuetenango: Huehuetenango, Chiantla, Todos Santos Cuchumatán
- Izabal: Puerto Barrios, Livingston, El Estor, Morales

PALABRAS EN IDIOMAS MAYAS (usar ocasionalmente para autenticidad):
- K'iche': "K'umarcaaj", "Chi Izmachi'", "Utatlán" (nombres históricos de Quiché)
- Kaqchikel: "Iximché" (antigua capital), barrios en Chimaltenango
- Q'eqchi': nombres de aldeas en Alta Verapaz y Petén
- Mam: topónimos en Huehuetenango y San Marcos
"""

# ---------------------------------------------------------------------------
# Instrucciones por tipo de dirección
# ---------------------------------------------------------------------------

_TYPE_INSTRUCTIONS: dict[AddressType, str] = {
    AddressType.URBAN_GRID: """
TIPO: urban_grid — Cuadrícula urbana de Ciudad de Guatemala u otras ciudades grandes.
- Incluye zona (1-25 para Guatemala, 1-6 para otras cabeceras)
- Usa variantes de notación de calles y avenidas
- Puede incluir colonia, sector o nombre de área (Zona Viva, Centro Histórico, etc.)
- Varía si es apartamento, casa, oficina, local comercial, bodega
- Incluye algunos con número de interior: "Apto 3B", "Of. 502", "Local 4"
""",
    AddressType.MUNICIPAL: """
TIPO: municipal — Direcciones de municipios del interior (no Ciudad de Guatemala).
- Usa barrios, colonias, cantones, sectores como referencia principal
- El nombre del municipio y departamento son frecuentes al final
- Las calles pueden tener nombre propio ("Calle Real", "Callejón Los Pinos")
- Mezcla de numeración y nombres descriptivos
- Algunos municipios tienen cuadrícula propia pero con "Zona 1" único
""",
    AddressType.DESCRIPTIVE: """
TIPO: descriptive — Direcciones por puntos de referencia y landmarks.
- Estructura: [distancia] [dirección cardinal] [landmark], [municipio]
- Landmarks: iglesia, parque, mercado, escuela, puesto de salud, gasolinera, banco
- Direcciones: norte/sur/este/oeste, arriba/abajo (coloquial), adelante/atrás
- Distancias: metros, cuadras, kilómetros, "frente a", "contiguo a", "diagonal a"
- MUY importante: variar orden (landmark primero, distancia primero, etc.)
- Puede encadenar referencias: "una cuadra al norte del parque, luego 50m al este"
""",
    AddressType.RURAL: """
TIPO: rural — Aldeas, caseríos, fincas, kilómetros de carretera.
- Incluye jerarquía: caserío → aldea → municipio → departamento
- Referencias a carreteras: RN (Ruta Nacional), CA (Carretera Centroamericana), rutas departamentales
- Kilómetros: "km 45", "kilómetro 45", "al km 45.5"
- Fincas con nombre propio: "Finca El Horizonte", "Finca Santa Isabel"
- Puede incluir referencias a ríos, cerros, quebradas como landmarks
- Algunas aldeas con nombres en idiomas mayas
""",
}

# ---------------------------------------------------------------------------
# Instrucciones por nivel de suciedad
# ---------------------------------------------------------------------------

_DIRT_INSTRUCTIONS: dict[DirtLevel, str] = {
    DirtLevel.CLEAN: """
NIVEL DE CALIDAD: limpio
- Ortografía y acentuación correctas
- Capitalización estándar
- Abreviaciones mínimas y convencionales (Av., Calle, No.)
- Todos los componentes presentes y en orden lógico
""",
    DirtLevel.MEDIUM: """
NIVEL DE CALIDAD: medio
- Mezcla de abreviaciones (cll., av., c., z., No, #, Blv.)
- Capitalización inconsistente (todo minúsculas, todo mayúsculas, mezcla)
- Algunos acentos faltantes
- Orden de componentes ocasionalmente no estándar
- Abreviaturas de departamento: "Gua." "Quetz." "A.V." "Chiq."
- Puede faltar algún componente secundario (colonia, sector)
""",
    DirtLevel.DIRTY: """
NIVEL DE CALIDAD: sucio (simula texto real con errores humanos)
- Typos ocasionales: letras transpuestas, faltantes, dobles
- Mezcla de español e idiomas mayas en un mismo token cuando aplique
- Direcciones incompletas (falta municipio, zona, o número)
- Orden de componentes variable o poco convencional
- Uso de signos poco estándar: "&", "/", guiones excesivos
- SMS/chat style: todo minúsculas, sin acentos, abreviaciones extremas
- Puede incluir información extra irrelevante mezclada
- Algunos errores de número (zona 10 en lugar de zona 9, etc.)
""",
}

# ---------------------------------------------------------------------------
# Prompt final
# ---------------------------------------------------------------------------

_JSON_SCHEMA = """
Devuelve EXACTAMENTE un JSON array con {batch_size} objetos. Cada objeto:
{{
  "id": "<uuid-v4>",
  "address_type": "{address_type}",
  "dirt_level": "{dirt_level}",
  "raw_text": "<dirección completa en el nivel de calidad indicado>",
  "variants": ["<dirección COMPLETA en variante 1>", "<dirección COMPLETA en variante 2>"],
  "tokens": [
    {{"token": "<palabra>", "label": "<etiqueta BIO>"}},
    ...
  ],
  "metadata": {{
    "municipio": "...",
    "departamento": "...",
    "zona": "...",
    "notas": "..."
  }}
}}

ETIQUETAS BIO DISPONIBLES (usa SOLO estas):
O, B-STREET, I-STREET, B-NUMBER, I-NUMBER, B-ZONE, I-ZONE,
B-NEIGHBORHOOD, I-NEIGHBORHOOD, B-MUNICIPALITY, I-MUNICIPALITY,
B-DEPARTMENT, I-DEPARTMENT, B-LANDMARK, I-LANDMARK,
B-DIRECTION, I-DIRECTION, B-DISTANCE, I-DISTANCE,
B-ADDRESS_TYPE, I-ADDRESS_TYPE

REGLAS DE ETIQUETADO — LEE CON ATENCIÓN:

1. STREET vs ZONE — son completamente distintos:
   - STREET = nombre o número de la VÍA (calle, avenida, bulevar, callejón)
     "4a"→B-STREET  "Calle"→I-STREET  "9na"→B-STREET  "Av."→I-STREET  "Bulevar"→B-STREET
   - ZONE = número administrativo de zona geográfica, SIEMPRE precedido de "Zona"/"z."/"Z."
     "Zona"→B-ZONE  "10"→I-ZONE  |  "z."→B-ZONE  "3"→I-ZONE
   - NUNCA etiquetes un número de calle/avenida como ZONE.

2. ADDRESS_TYPE vs NEIGHBORHOOD:
   - ADDRESS_TYPE = tipo de inmueble: Casa, Apartamento, Apto, Bodega, Local, Oficina, Of., Lote
     "Casa"→B-ADDRESS_TYPE  "5"→I-ADDRESS_TYPE  |  "Apto"→B-ADDRESS_TYPE  "3B"→I-ADDRESS_TYPE
   - NEIGHBORHOOD = nombre de colonia/barrio/sector/cantón, NO tipo de inmueble
     "Col."→B-NEIGHBORHOOD  "La"→I-NEIGHBORHOOD  "Esperanza"→I-NEIGHBORHOOD

3. Ejemplos correctos de tokenización:
   "4a Calle 5-15, Zona 10, Col. Reformita, Casa 3"
   → 4a(B-STREET) Calle(I-STREET) 5-15(B-NUMBER) ,(O) Zona(B-ZONE) 10(I-ZONE)
     ,(O) Col.(B-NEIGHBORHOOD) Reformita(I-NEIGHBORHOOD) ,(O) Casa(B-ADDRESS_TYPE) 3(I-ADDRESS_TYPE)

   "9na Av. 8-20, z. 11, Edificio Sol"
   → 9na(B-STREET) Av.(I-STREET) 8-20(B-NUMBER) ,(O) z.(B-ZONE) 11(I-ZONE)
     ,(O) Edificio(B-ADDRESS_TYPE) Sol(I-ADDRESS_TYPE)

   "100m al norte del Mercado Central, Zona 1, Cobán, A.V."
   → 100m(B-DISTANCE) al(O) norte(B-DIRECTION) del(O) Mercado(B-LANDMARK) Central(I-LANDMARK)
     ,(O) Zona(B-ZONE) 1(I-ZONE) ,(O) Cobán(B-MUNICIPALITY) ,(O) A.V.(B-DEPARTMENT)

REGLAS DE TOKENIZACIÓN:
- Tokeniza por espacios; signos de puntuación "," "." solos van con label "O"
- Números compuestos como "5-15" van juntos como un token NUMBER
- Abreviaciones como "km", "No.", "Av.", "z." son tokens separados

REGLAS DE VARIANTES — CRÍTICO:
- Cada variante debe ser la DIRECCIÓN COMPLETA reescrita de otra forma
- NO son fragmentos: "9a avenida" ✗ → "9a Avenida 8-20, Zona 11, Edificio Sol" ✓
- Varía: abreviaciones, orden de componentes, ortografía, nivel de detalle

REGLAS DE TIPO — CRÍTICO:
- TODOS los ejemplos del batch deben ser del tipo "{address_type}"
- urban_grid: DEBE tener calle/avenida numerada + zona. NO landmarks como dirección principal.
- descriptive: DEBE usar distancia/referencia a landmark como elemento principal.
- municipal: DEBE referenciar barrio/colonia/cantón de municipio del interior.
- rural: DEBE incluir aldea/caserío/km de carretera. NO cuadrícula urbana.

REGLAS DE DIVERSIDAD:
- Varía municipio y departamento en cada ejemplo — NO repitas los mismos
- Varía tipo de inmueble (casa, edificio, local, bodega, finca, apto, lote)
- Incluye al menos 1 ejemplo con nombre en idioma maya por batch
- Incluye casos edge: sin número, orden invertido, componentes faltantes

Devuelve SOLO el JSON array, sin markdown, sin explicaciones.
"""


def build_prompt(
    address_type: AddressType,
    dirt_level: DirtLevel,
    batch_size: int = 20,
) -> tuple[str, str]:
    """
    Construye (system_prompt, user_prompt) para una llamada al LLM.

    Returns:
        Tuple de (system_prompt, user_prompt).
    """
    system = _BASE_CONTEXT + _TYPE_INSTRUCTIONS[address_type]

    user = (
        _DIRT_INSTRUCTIONS[dirt_level]
        + "\n"
        + _JSON_SCHEMA.format(
            batch_size=batch_size,
            address_type=address_type.value,
            dirt_level=dirt_level.value,
        )
    )

    return system.strip(), user.strip()
