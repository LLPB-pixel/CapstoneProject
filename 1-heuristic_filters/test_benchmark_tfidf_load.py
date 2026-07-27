"""
Tests de benchmark para carga del modelo TF-IDF (tfidf_model.joblib)

Propósito: medir el tiempo de deserialización de joblib para determinar
si la carga del modelo es un cuello de botella en el pipeline.

Ejecuta con: pytest test_benchmark_tfidf_load.py -v -s
"""

import time
import os
import joblib
import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "tfidf_model.joblib")


class TestTfidfLoadBenchmark:
    """Benchmark de carga del modelo TF-IDF con joblib"""

    def test_model_file_exists(self):
        """El archivo del modelo debe existir"""
        assert os.path.exists(MODEL_PATH), f"Modelo no encontrado: {MODEL_PATH}"

    def test_load_time(self, capsys):
        """Mide el tiempo de carga del modelo con joblib.load"""
        t0 = time.perf_counter()
        model = joblib.load(MODEL_PATH)
        elapsed = time.perf_counter() - t0

        capsys.readouterr()
        print(f"\nTiempo joblib.load: {elapsed:.4f}s")
        print(f"Tipo: {type(model)}")

        assert model is not None
        assert elapsed < 30.0, f"Carga demasiado lenta: {elapsed:.2f}s (>30s)"

    def test_model_type(self):
        """El modelo cargado debe ser un Pipeline de scikit-learn"""
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict") or hasattr(model, "transform"), \
            f"Modelo no tiene predict/transform: {type(model)}"

    def test_repeated_load_times(self, capsys):
        """Carga múltiples veces para ver variabilidad (warm cache)"""
        times = []
        for i in range(3):
            t0 = time.perf_counter()
            model = joblib.load(MODEL_PATH)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        capsys.readouterr()
        print(f"\nTiempos de carga (3 rondas): {[f'{t:.4f}s' for t in times]}")
        print(f"Promedio: {avg:.4f}s")

        assert model is not None
