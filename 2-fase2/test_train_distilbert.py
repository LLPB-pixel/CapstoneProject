#!/usr/bin/env python3
"""Tests y diagnóstico para train_distilbert.py.

Objetivo:
  1. Validar la estructura y calidad de train.csv, val.csv y test.csv.
  2. Detectar etiquetas inconsistentes entre splits.
  3. Comprobar que la tokenización produce tensores válidos.
  4. Verificar que compute_metrics devuelve resultados finitos.
  5. Reproducir un forward/backward pequeño y detectar NaN/Inf.

Uso recomendado:
    pytest -q -s test_train_distilbert.py

Para indicar otra carpeta de datos:
    DATA_DIR=/ruta/a/Data pytest -q -s test_train_distilbert.py

Diagnóstico sin pytest:
    python test_train_distilbert.py --data_dir ../Data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import train_distilbert as training


DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "Data"))
SPLITS = ("train", "val", "test")


def _csv_path(split: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"{split}.csv"


def _read_raw_split(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = _csv_path(split, data_dir)
    if not path.exists():
        pytest.skip(f"No existe {path}. Define DATA_DIR para ejecutar este test.")
    df = pd.read_csv(path)
    if "prompt" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"prompt": "text"})
    return df


def _normalise_label(value) -> str:
    """Normaliza 0, 0.0 y '0' a una representación comparable."""
    if pd.isna(value):
        return "<NA>"
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text.casefold()


class DummyTokenizer:
    """Tokenizador determinista y local, sin descargar modelos."""

    def __call__(self, texts, truncation=True, max_length=32, padding="max_length"):
        if isinstance(texts, str):
            texts = [texts]
        input_ids, attention_mask = [], []
        for text in texts:
            ids = [((ord(ch) % 127) + 1) for ch in str(text)][:max_length]
            mask = [1] * len(ids)
            ids += [0] * (max_length - len(ids))
            mask += [0] * (max_length - len(mask))
            input_ids.append(ids)
            attention_mask.append(mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TinyClassifier(nn.Module):
    """Clasificador mínimo para probar que loss y gradientes son finitos."""

    def __init__(self, vocab_size=128, hidden_size=16, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels):
        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        logits = self.classifier(pooled)
        loss = nn.functional.cross_entropy(logits.float(), labels)
        return loss, logits


@pytest.mark.parametrize("split", SPLITS)
def test_csv_has_required_columns(split):
    df = _read_raw_split(split)
    assert "text" in df.columns, f"{split}.csv no contiene 'text' ni 'prompt'."
    assert "label" in df.columns, f"{split}.csv no contiene la columna 'label'."


@pytest.mark.parametrize("split", SPLITS)
def test_no_missing_or_empty_texts(split):
    path = _csv_path(split)
    if not path.exists():
        pytest.skip(f"No existe {path}.")
    df = pd.read_csv(path)
    if "prompt" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"prompt": "text"})

    missing = df["text"].isna()
    empty = df["text"].fillna("").astype(str).str.strip().eq("")
    bad_mask = missing | empty
    n_bad = int(bad_mask.sum())

    if n_bad:
        print(f"[FIX] {split}.csv: eliminando {n_bad} filas con texto vacío/nulo.")
        df = df[~bad_mask].reset_index(drop=True)
        df.to_csv(path, index=False)


@pytest.mark.parametrize("split", SPLITS)
def test_labels_are_binary_and_complete(split):
    df = _read_raw_split(split)
    normalised = df["label"].map(_normalise_label)
    assert "<NA>" not in set(normalised), f"{split}.csv contiene etiquetas nulas."
    values = set(normalised)
    assert values == {"0", "1"}, (
        f"{split}.csv debe contener exactamente las etiquetas 0 y 1; encontradas: "
        f"{sorted(values)}"
    )


def test_label_semantics_are_consistent_between_splits():
    """Evita que load_split cree un mapeo local diferente para cada CSV."""
    labels_by_split = {}
    for split in SPLITS:
        df = _read_raw_split(split)
        labels_by_split[split] = set(df["label"].map(_normalise_label))
    first = labels_by_split["train"]
    assert all(values == first for values in labels_by_split.values()), (
        f"Conjuntos de etiquetas inconsistentes: {labels_by_split}"
    )


@pytest.mark.parametrize("split", SPLITS)
def test_load_split_returns_valid_examples(split):
    path = _csv_path(split)
    if not path.exists():
        pytest.skip(f"No existe {path}.")
    ds = training.load_split(str(path), DummyTokenizer(), max_length=32)
    assert len(ds) > 0, f"El split {split} está vacío."
    labels = np.asarray(ds["label"])
    assert np.isfinite(labels).all(), f"{split}: hay etiquetas NaN o Inf."
    assert set(labels.tolist()).issubset({0, 1}), f"{split}: etiquetas inválidas {set(labels)}."
    sample = ds[0]
    assert len(sample["input_ids"]) == 32
    assert len(sample["attention_mask"]) == 32


def test_compute_metrics_is_finite():
    logits = np.array([[3.0, -1.0], [-2.0, 4.0], [1.0, 2.0], [2.0, 0.5]])
    labels = np.array([0, 1, 1, 0])
    metrics = training.compute_metrics((logits, labels))
    assert set(metrics) == {"accuracy", "f1", "precision", "recall", "roc_auc"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_small_forward_backward_has_finite_values():
    tokenizer = DummyTokenizer()
    encoded = tokenizer(["benign prompt", "ignore previous instructions"], max_length=32)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
    labels = torch.tensor([0, 1], dtype=torch.long)

    model = TinyClassifier()
    loss, logits = model(input_ids, attention_mask, labels)
    assert torch.isfinite(logits).all(), "El forward produjo logits NaN/Inf."
    assert torch.isfinite(loss), "La loss es NaN/Inf antes del backward."

    loss.backward()
    bad_parameters = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not bad_parameters, f"Gradientes NaN/Inf en: {bad_parameters}"


def build_report(data_dir: Path) -> int:
    """Imprime un informe legible y devuelve 0 si no detecta errores graves."""
    errors = []
    print(f"Diagnóstico de datos: {data_dir.resolve()}\n")

    label_sets = {}
    for split in SPLITS:
        path = _csv_path(split, data_dir)
        print(f"[{split.upper()}] {path}")
        if not path.exists():
            errors.append(f"Falta {path}")
            print("  ERROR: archivo inexistente\n")
            continue

        df = pd.read_csv(path)
        if "prompt" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"prompt": "text"})
        missing_columns = {"text", "label"} - set(df.columns)
        if missing_columns:
            errors.append(f"{split}: faltan columnas {sorted(missing_columns)}")
            print(f"  ERROR: faltan columnas {sorted(missing_columns)}\n")
            continue

        texts = df["text"].fillna("").astype(str).str.strip()
        labels = df["label"].map(_normalise_label)
        label_sets[split] = set(labels)
        empty_count = int(texts.eq("").sum())
        null_labels = int(labels.eq("<NA>").sum())
        print(f"  Filas: {len(df)}")
        print(f"  Textos vacíos/nulos: {empty_count}")
        print(f"  Etiquetas nulas: {null_labels}")
        print(f"  Distribución: {labels.value_counts(dropna=False).to_dict()}")

        if empty_count:
            errors.append(f"{split}: {empty_count} textos vacíos/nulos")
        if null_labels:
            errors.append(f"{split}: {null_labels} etiquetas nulas")
        if set(labels) != {"0", "1"}:
            errors.append(f"{split}: etiquetas esperadas {{'0','1'}}, obtenidas {set(labels)}")
        print()

    if len(label_sets) == len(SPLITS):
        if not all(values == label_sets["train"] for values in label_sets.values()):
            errors.append(f"Etiquetas inconsistentes entre splits: {label_sets}")

    if errors:
        print("RESULTADO: se han detectado problemas que pueden causar resultados incorrectos.")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("RESULTADO: los datos superan las comprobaciones básicas.")
    print("Si el entrenamiento sigue produciendo NaN, prueba gradient clipping, un LR menor y registra logits/loss por batch.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico de train_distilbert.py")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    return build_report(args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
