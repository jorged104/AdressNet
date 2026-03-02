"""
Gazetteer de lugares de Guatemala para features externas al modelo.

Fuente: listado oficial de departamentos y municipios del INE/IGN (dominio público).
URL original del PDF INE: https://www.ine.gob.gt/sistema/uploads/2014/02/26/L2S0g1gBEIDSAMoWvXFOqkRBRaJolIk7.pdf
(usada como referencia; datos hardcodeados porque el PDF no siempre está disponible)

Uso:
    from src.utils.gazetteer import get_geo_features
    feats = get_geo_features("jalapa")
    # → {"is_departamento": True, "is_municipio": True, "is_aldea": False}
"""

from __future__ import annotations

import unicodedata


# ---------------------------------------------------------------------------
# Normalización: quita tildes y pasa a minúsculas para matching robusto
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normaliza: minúsculas + sin tildes."""
    s = s.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# ---------------------------------------------------------------------------
# 22 Departamentos de Guatemala (nombres canónicos)
# ---------------------------------------------------------------------------

_DEPARTAMENTOS_RAW: list[str] = [
    "Guatemala",
    "El Progreso",
    "Sacatepéquez",
    "Chimaltenango",
    "Escuintla",
    "Santa Rosa",
    "Sololá",
    "Totonicapán",
    "Quetzaltenango",
    "Suchitepéquez",
    "Retalhuleu",
    "San Marcos",
    "Huehuetenango",
    "Quiché",
    "Baja Verapaz",
    "Alta Verapaz",
    "Petén",
    "Izabal",
    "Zacapa",
    "Chiquimula",
    "Jalapa",
    "Jutiapa",
]

# Set normalizado para búsqueda O(1)
DEPARTAMENTOS: set[str] = {_norm(d) for d in _DEPARTAMENTOS_RAW}


# ---------------------------------------------------------------------------
# Municipios por departamento
# ---------------------------------------------------------------------------

MUNICIPIOS_POR_DEPARTAMENTO: dict[str, list[str]] = {
    "Guatemala": [
        "Guatemala", "Santa Catarina Pinula", "San José Pinula", "San José del Golfo",
        "Palencia", "Chinautla", "San Pedro Ayampuc", "Mixco", "San Pedro Sacatepéquez",
        "San Juan Sacatepéquez", "San Raymundo", "Chuarrancho", "Fraijanes",
        "Amatitlán", "Villa Nueva", "Villa Canales", "San Miguel Petapa",
    ],
    "El Progreso": [
        "Guastatoya", "Morazán", "San Agustín Acasaguastlán", "San Cristóbal Acasaguastlán",
        "El Jícaro", "Sansare", "Sanarate", "San Antonio La Paz",
    ],
    "Sacatepéquez": [
        "Antigua Guatemala", "Jocotenango", "Pastores", "Sumpango",
        "Santo Domingo Xenacoj", "Santiago Sacatepéquez", "San Bartolomé Milpas Altas",
        "San Lucas Sacatepéquez", "Santa Lucía Milpas Altas", "Magdalena Milpas Altas",
        "Santa María de Jesús", "Ciudad Vieja", "San Miguel Dueñas", "Alotenango",
        "San Antonio Aguas Calientes", "Santa Catarina Barahona",
    ],
    "Chimaltenango": [
        "Chimaltenango", "San José Poaquil", "San Martín Jilotepeque", "Comalapa",
        "Santa Apolonia", "Tecpán Guatemala", "Patzún", "Pochuta", "Patzicía",
        "Santa Cruz Balanyá", "Acatenango", "Yepocapa", "San Andrés Itzapa",
        "Parramos", "Zaragoza", "El Tejar",
    ],
    "Escuintla": [
        "Escuintla", "Santa Lucía Cotzumalguapa", "La Democracia", "Siquinalá",
        "Masagua", "Tiquisate", "La Gomera", "Guanagazapa", "San José", "Iztapa",
        "Palín", "San Vicente Pacaya", "Nueva Concepción",
    ],
    "Santa Rosa": [
        "Cuilapa", "Barberena", "Santa Rosa de Lima", "Casillas",
        "San Rafael Las Flores", "Oratorio", "San Juan Tecuaco", "Chiquimulilla",
        "Taxisco", "Santa María Ixhuatán", "Guazacapán", "Santa Cruz Naranjo",
        "Pueblo Nuevo Viñas", "Nueva Santa Rosa",
    ],
    "Sololá": [
        "Sololá", "San José Chacayá", "Santa María Visitación", "Santa Lucía Utatlán",
        "Nahualá", "Santa Catarina Ixtahuacán", "Santa Clara La Laguna", "Concepción",
        "San Andrés Semetabaj", "Panajachel", "Santa Catarina Palopó",
        "San Antonio Palopó", "San Lucas Tolimán", "Santa Cruz La Laguna",
        "San Pablo La Laguna", "San Marcos La Laguna", "San Juan La Laguna",
        "San Pedro La Laguna", "Santiago Atitlán",
    ],
    "Totonicapán": [
        "Totonicapán", "San Cristóbal Totonicapán", "San Francisco El Alto",
        "San Andrés Xecul", "Momostenango", "Santa María Chiquimula",
        "Santa Lucía La Reforma", "San Bartolo",
    ],
    "Quetzaltenango": [
        "Quetzaltenango", "Salcajá", "Olintepeque", "San Carlos Sija", "Sibilia",
        "Cabricán", "Cajolá", "San Miguel Sigüilá", "San Juan Ostuncalco", "San Mateo",
        "Concepción Chiquirichapa", "San Martín Sacatepéquez", "Almolonga", "Cantel",
        "Huitán", "Zunil", "Colomba Costa Cuca", "San Francisco La Unión", "El Palmar",
        "Coatepeque", "Génova", "Flores Costa Cuca", "La Esperanza",
        "Palestina de Los Altos",
    ],
    "Suchitepéquez": [
        "Mazatenango", "Cuyotenango", "San Francisco Zapotitlán", "San Bernardino",
        "San José El Ídolo", "Santo Domingo Suchitepéquez", "San Lorenzo", "Samayac",
        "San Pablo Jocopilas", "San Antonio Suchitepéquez", "San Miguel Panán",
        "San Gabriel", "Chicacao", "Patulul", "Santa Bárbara", "San Juan Bautista",
        "Santo Tomás La Unión", "Zunilito", "Pueblo Nuevo", "Río Bravo",
        "San José La Máquina",
    ],
    "Retalhuleu": [
        "Retalhuleu", "San Sebastián", "Santa Cruz Muluá", "San Martín Zapotitlán",
        "San Felipe", "San Andrés Villa Seca", "Champerico", "Nuevo San Carlos",
        "El Asintal",
    ],
    "San Marcos": [
        "San Marcos", "San Pedro Sacatepéquez", "San Antonio Sacatepéquez",
        "Comitancillo", "San Miguel Ixtahuacán", "Concepción Tutuapa", "Tacaná",
        "Sibinal", "Tajumulco", "Tejutla", "San Rafael Pie de la Cuesta",
        "Nuevo Progreso", "El Tumbador", "El Rodeo", "Malacatán", "Catarina",
        "Ayutla", "Ocós", "San Pablo", "El Quetzal", "La Reforma", "Pajapita",
        "Ixchiguán", "San José Ojetenam", "San Cristóbal Cucho", "Sipacapa",
        "Esquipulas Palo Gordo", "Río Blanco", "San Lorenzo",
    ],
    "Huehuetenango": [
        "Huehuetenango", "Chiantla", "Malacatancito", "Cuilco", "Nentón",
        "San Pedro Necta", "Jacaltenango", "Soloma", "Ixtahuacán", "Santa Bárbara",
        "La Libertad", "La Democracia", "San Miguel Acatán",
        "San Rafael La Independencia", "Todos Santos Cuchumatán", "San Juan Atitán",
        "Santa Eulalia", "San Mateo Ixtatán", "Colotenango",
        "San Sebastián Huehuetenango", "Tectitán", "Concepción Huista", "San Juan Ixcoy",
        "San Antonio Huista", "San Sebastián Coatán", "Cocolí", "Santa Cruz Barillas",
        "Aguacatán", "San Rafael Petzal", "San Gaspar Ixchil",
        "Santiago Chimaltenango", "Santa Ana Huista",
    ],
    "Quiché": [
        "Santa Cruz del Quiché", "Chiché", "Chinique", "Zacualpa", "Chajul",
        "Santo Tomás Chichicastenango", "Patzité", "San Antonio Ilotenango",
        "San Pedro Jocopilas", "Cunén", "San Juan Cotzal", "Joyabaj", "Nebaj",
        "San Andrés Sajcabajá", "Uspantán", "Sacapulas", "San Bartolomé Jocotenango",
        "Canillá", "Chicamán", "Playa Grande Ixcán", "Pachalum",
    ],
    "Baja Verapaz": [
        "Salamá", "San Miguel Chicaj", "Rabinal", "Cubulco", "Granados",
        "El Chol", "San Jerónimo", "Purulhá",
    ],
    "Alta Verapaz": [
        "Cobán", "Santa Cruz Verapaz", "San Cristóbal Verapaz", "Tactic", "Tamahú",
        "San Miguel Tucurú", "Panzós", "Senahú", "San Pedro Carchá",
        "San Juan Chamelco", "Lanquín", "Cahabón", "Chisec", "Chahal",
        "Fray Bartolomé de las Casas", "Santa Catarina La Tinta", "Raxruhá",
    ],
    "Petén": [
        "Flores", "San José", "San Benito", "San Andrés", "La Libertad",
        "San Francisco", "Santa Ana", "Dolores", "San Luis", "Sayaxché",
        "Melchor de Mencos", "Poptún",
    ],
    "Izabal": [
        "Puerto Barrios", "Livingston", "El Estor", "Morales", "Los Amates",
    ],
    "Zacapa": [
        "Zacapa", "Estanzuela", "Río Hondo", "Gualán", "Teculután", "Usumatlán",
        "Cabañas", "San Diego", "La Unión", "Huité", "San Jorge",
    ],
    "Chiquimula": [
        "Chiquimula", "San José La Arada", "San Juan La Ermita", "Jocotán", "Camotán",
        "Olopa", "Esquipulas", "Concepción Las Minas", "Quezaltepeque", "San Jacinto",
        "Ipala",
    ],
    "Jalapa": [
        "Jalapa", "San Pedro Pinula", "San Luis Jilotepeque", "San Manuel Chaparrón",
        "San Carlos Alzatate", "Monjas", "Mataquescuintla",
    ],
    "Jutiapa": [
        "Jutiapa", "El Progreso", "Santa Catarina Mita", "Agua Blanca",
        "Asunción Mita", "Yupiltepeque", "Atescatempa", "Jerez", "El Adelanto",
        "Zapotitlán", "Comapa", "Jalpatagua", "Conguaco", "Moyuta", "Pasaco",
        "San José Acatempa", "Quesada",
    ],
}

# Set normalizado de todos los municipios para búsqueda O(1)
MUNICIPIOS: set[str] = {
    _norm(mun)
    for municipios in MUNICIPIOS_POR_DEPARTAMENTO.values()
    for mun in municipios
}

# ---------------------------------------------------------------------------
# Aldeas y caseríos (subconjunto representativo)
# Nota: el listado completo del INE tiene miles de entradas. Esta tabla
# puede ampliarse importando el CSV oficial cuando esté disponible.
# Se incluyen aquí las aldeas más frecuentes en direcciones rurales.
# ---------------------------------------------------------------------------

_ALDEAS_RAW: list[str] = [
    # Aldeas frecuentes en correspondencia y referencias rurales
    "Xela", "Almolonga", "Cantel", "Zunil", "Salcajá",
    "Chicacao", "Patulul", "Samayac",
    "Chichi", "Chichicastenango",
    "Livingston", "Panzós",
    "Nebaj", "Chajul", "Cotzal",
    "Todos Santos", "Jacaltenango",
    "Cobán", "Tactic", "Cahabón", "Lanquín",
    "Sayaxché", "Poptún", "Flores",
    "Esquipulas", "Ipala", "Jocotán",
    "Mataquescuintla", "Monjas",
    "Tiquisate", "Palín", "Amatitlán",
    "Rabinal", "Salamá", "Cubulco",
    # Caseríos y aldeas genéricas comunes en texto libre
    "aldea", "caserio", "caserío", "paraje", "finca",
    "comunidad", "colonia", "barrio", "canton", "cantón",
]

ALDEAS: set[str] = {_norm(a) for a in _ALDEAS_RAW}


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

def get_geo_features(token: str) -> dict[str, bool]:
    """
    Retorna features geográficas para un token dado.

    Args:
        token: Token tal como aparece en el texto (mayúsculas/minúsculas,
               con o sin tildes).

    Returns:
        dict con tres claves booleanas:
          - is_departamento: True si el token coincide con uno de los
                             22 departamentos de Guatemala.
          - is_municipio:    True si el token coincide con algún municipio.
          - is_aldea:        True si el token coincide con una aldea/caserío
                             conocido (lista extensible).

    Nota:
        El matching es insensible a mayúsculas y tildes. Para tokens que
        forman parte de un nombre compuesto (p.ej. "San" en "San Marcos")
        solo se retorna True si el token completo coincide; el modelo Bi-LSTM
        aproviga el contexto para resolverlo.

    Ejemplo:
        >>> get_geo_features("Jalapa")
        {"is_departamento": True, "is_municipio": True, "is_aldea": False}
        >>> get_geo_features("Mixco")
        {"is_departamento": False, "is_municipio": True, "is_aldea": False}
        >>> get_geo_features("finca")
        {"is_departamento": False, "is_municipio": False, "is_aldea": True}
    """
    key = _norm(token)
    return {
        "is_departamento": key in DEPARTAMENTOS,
        "is_municipio":    key in MUNICIPIOS,
        "is_aldea":        key in ALDEAS,
    }


def get_geo_feature_vector(token: str) -> list[float]:
    """
    Versión que retorna directamente [is_dept, is_mun, is_aldea] como floats.
    Conveniente para construir tensores en PyTorch.
    """
    feats = get_geo_features(token)
    return [
        float(feats["is_departamento"]),
        float(feats["is_municipio"]),
        float(feats["is_aldea"]),
    ]
