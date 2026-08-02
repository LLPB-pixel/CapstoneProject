#!/usr/bin/env python3
"""
Tests de la Capa 2 - DeBERTa-v3-base (modelo guard).

Estos tests requieren que el modelo esté entrenado.
Si no hay modelo, los tests se saltan automáticamente.
"""

import os
import sys
import pytest

# Rutas
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "data", "models", "deberta")
_MODEL_EXISTS = os.path.isdir(_MODEL_DIR)

# Saltar todos los tests si no hay modelo
pytestmark = pytest.mark.skipif(not _MODEL_EXISTS, reason=f"Modelo DeBERTa no encontrado en {_MODEL_DIR}")


class TestModelExists:
    """Test básico de existencia del modelo."""
    
    def test_model_directory_exists(self):
        """Verificar que el directorio del modelo existe."""
        assert _MODEL_EXISTS, f"Modelo no encontrado en {_MODEL_DIR}"
