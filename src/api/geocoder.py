"""Utilidades de geocodificación para direcciones guatemaltecas."""

from __future__ import annotations

import json
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import Request, urlopen

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


@lru_cache(maxsize=1024)
def geocode_exact_address(address: str) -> dict[str, str | float] | None:
    """
    Intenta geocodificar una dirección exacta usando OpenStreetMap Nominatim.

    Retorna lat/lon cuando encuentra un resultado suficientemente específico.
    """
    query = address.strip()
    if not query:
        return None

    params = urlencode(
        {
            "q": f"{query}, Guatemala",
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "countrycodes": "gt",
        }
    )
    url = f"{NOMINATIM_URL}?{params}"

    request = Request(
        url,
        headers={
            "User-Agent": "gt-address-parser/0.1 (+https://github.com/)"
        },
    )

    try:
        with urlopen(request, timeout=4.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    if not payload:
        return None

    first = payload[0]
    try:
        lat = float(first["lat"])
        lon = float(first["lon"])
    except (KeyError, TypeError, ValueError):
        return None

    return {
        "lat": lat,
        "lon": lon,
        "precision": "exact",
        "source": "nominatim",
        "name": str(first.get("display_name", address)),
    }
