#!/usr/bin/env python3
"""
Tests y diagnóstico para train_distilbert.py.

Estos tests requieren que los datos de entrenamiento existan.
"""

import os
import pytest
from pathlib import Path

# Rutas
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "splits"))
DATA_EXISTS = (
    (DATA_DIR / "train.csv").exists() and
    (DATA_DIR / "val.csv").exists() and
    (DATA_DIR / "test.csv").exists()
)

# Saltar todos los tests si no hay datos
pytestmark = pytest.mark.skipif(not DATA_EXISTS, reason=f"Datos de entrenamiento no encontrados en {DATA_DIR}")


class TestDataExists:
    """Test básico de existencia de datos."""
    
    def test_train_csv_exists(self):
        """Verificar que train.csv existe."""
        assert (DATA_DIR / "train.csv").exists()
    
    def test_val_csv_exists(self):
        """Verificar que val.csv existe."""
        assert (DATA_DIR / "val.csv").exists()
    
    def test_test_csv_exists(self):
        """Verificar que test.csv existe."""
        assert (DATA_DIR / "test.csv").exists()
