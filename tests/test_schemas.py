"""Tests básicos de validación de schemas."""

import pytest
from pydantic import ValidationError

from src.utils.schemas import AddressSample, AddressType, DirtLevel, TokenLabel


def _make_sample(**overrides) -> dict:
    base = {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "address_type": "urban_grid",
        "dirt_level": "clean",
        "raw_text": "4a Calle 5-15, Zona 10",
        "variants": ["4 calle 5-15 zona 10"],
        "tokens": [
            {"token": "4a", "label": "B-STREET"},
            {"token": "Calle", "label": "I-STREET"},
            {"token": "5-15", "label": "B-NUMBER"},
            {"token": ",", "label": "O"},
            {"token": "Zona", "label": "B-ZONE"},
            {"token": "10", "label": "I-ZONE"},
        ],
        "metadata": {"municipio": "Guatemala", "departamento": "Guatemala"},
    }
    base.update(overrides)
    return base


def test_valid_sample():
    sample = AddressSample.model_validate(_make_sample())
    assert sample.address_type == AddressType.URBAN_GRID
    assert len(sample.tokens) == 6


def test_invalid_label_rejected():
    data = _make_sample()
    data["tokens"][0]["label"] = "B-INVALID"
    with pytest.raises(ValidationError):
        AddressSample.model_validate(data)


def test_empty_variants_rejected():
    data = _make_sample(variants=[])
    with pytest.raises(ValidationError):
        AddressSample.model_validate(data)


def test_too_many_variants_rejected():
    data = _make_sample(variants=["v1", "v2", "v3", "v4"])
    with pytest.raises(ValidationError):
        AddressSample.model_validate(data)


def test_all_address_types():
    for addr_type in AddressType:
        data = _make_sample(address_type=addr_type.value)
        sample = AddressSample.model_validate(data)
        assert sample.address_type == addr_type


def test_all_dirt_levels():
    for level in DirtLevel:
        data = _make_sample(dirt_level=level.value)
        sample = AddressSample.model_validate(data)
        assert sample.dirt_level == level
