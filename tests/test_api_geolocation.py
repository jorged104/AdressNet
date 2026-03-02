"""Tests para geolocalización expuesta por la API."""

from src.api.app import _infer_geolocation


def test_infer_geolocation_exact(monkeypatch):
    monkeypatch.setattr(
        "src.api.app.geocode_exact_address",
        lambda _: {
            "lat": 14.6,
            "lon": -90.5,
            "precision": "exact",
            "source": "nominatim",
            "name": "Dirección exacta",
        },
    )

    geo = _infer_geolocation("3a Avenida 4-56 Zona 1, Guatemala", {"DEPARTMENT": "Guatemala"})
    assert geo is not None
    assert geo["precision"] == "exact"


def test_infer_geolocation_department_fallback(monkeypatch):
    monkeypatch.setattr("src.api.app.geocode_exact_address", lambda _: None)

    geo = _infer_geolocation("sin match exacto", {"DEPARTMENT": "Guatemala"})
    assert geo is not None
    assert geo["precision"] == "department_fallback"


def test_infer_geolocation_returns_none_for_unknown_values(monkeypatch):
    monkeypatch.setattr("src.api.app.geocode_exact_address", lambda _: None)

    geo = _infer_geolocation("direccion inventada", {"MUNICIPALITY": "Municipio Inventado"})
    assert geo is None
