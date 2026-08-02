#!/usr/bin/env python3
"""
Benchmark de la Capa 1 (Heuristic Filter) del pipeline de deteccion de injection.

Evalua:
  1. Accuracy, Precision, Recall, F1-Score
  2. Latencia (media, p50, p95, p99, total)
  3. Matriz de confusion
  4. Rendimiento por componente
"""

import argparse
import os
import sys
import time
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from prompt_guard.layers.layer1_heuristic.filter import HeuristicFilter


@dataclass
class BenchmarkResult:
    config_name: str
    total: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    latencies_ms: List[float]
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_time_s: float
    suspicious_rate: float


def load_data(data_path: str, n_samples: int, seed: int = 42) -> Tuple[List[str], List[int]]:
    """Carga y balancea el dataset para benchmark."""
    df = pd.read_csv(data_path)
    df["label_int"] = (df["label"] == "bad").astype(int)

    n_per_class = n_samples // 2
    good = df[df["label_int"] == 0].sample(n=n_per_class, random_state=seed)
    bad = df[df["label_int"] == 1].sample(n=n_per_class, random_state=seed)

    sampled = pd.concat([good, bad]).sample(frac=1, random_state=seed).reset_index(drop=True)
    texts = sampled["text"].astype(str).tolist()
    labels = sampled["label_int"].tolist()
    return texts, labels


def evaluate(
    filter_obj: HeuristicFilter,
    texts: List[str],
    labels: List[int],
    config_name: str,
) -> BenchmarkResult:
    """Ejecuta el filtro en todos los textos y calcula metricas."""
    tp = fp = tn = fn = 0
    latencies = []

    for text, label in zip(texts, labels):
        t0 = time.perf_counter()
        result = filter_obj.analyze(text)
        t1 = time.perf_counter()
        latencies_ms = (t1 - t0) * 1000
        latencies.append(latencies_ms)

        predicted = 1 if result.is_suspicious else 0
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    total = len(texts)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    lat_arr = np.array(latencies)

    return BenchmarkResult(
        config_name=config_name,
        total=total,
        tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        latencies_ms=latencies,
        avg_latency_ms=float(lat_arr.mean()),
        p50_latency_ms=float(np.percentile(lat_arr, 50)),
        p95_latency_ms=float(np.percentile(lat_arr, 95)),
        p99_latency_ms=float(np.percentile(lat_arr, 99)),
        total_time_s=float(lat_arr.sum() / 1000),
        suspicious_rate=(tp + fp) / total if total > 0 else 0.0,
    )


def print_result(r: BenchmarkResult) -> None:
    """Imprime los resultados de una configuracion."""
    print(f"\n{'=' * 60}")
    print(f"  {r.config_name}")
    print(f"{'=' * 60}")
    print(f"  Muestras:        {r.total}")
    print(f"  TP={r.tp}  FP={r.fp}  TN={r.tn}  FN={r.fn}")
    print(f"  Precision:       {r.precision:.4f}")
    print(f"  Recall:          {r.recall:.4f}")
    print(f"  F1-Score:        {r.f1:.4f}")
    print(f"  Accuracy:        {r.accuracy:.4f}")
    print(f"  Suspicious rate: {r.suspicious_rate:.2%}")
    print(f"  Latencia avg:    {r.avg_latency_ms:.2f} ms")
    print(f"  Latencia p50:    {r.p50_latency_ms:.2f} ms")
    print(f"  Latencia p95:    {r.p95_latency_ms:.2f} ms")
    print(f"  Latencia p99:    {r.p99_latency_ms:.2f} ms")
    print(f"  Tiempo total:    {r.total_time_s:.2f} s")


def print_confusion_matrix(r: BenchmarkResult) -> None:
    """Imprime matriz de confusion formateada."""
    print(f"\n  Matriz de confusion ({r.config_name}):")
    print(f"  {'':>12} {'Pred Good':>10} {'Pred Bad':>10}")
    print(f"  {'True Good':>12} {r.tn:>10} {r.fp:>10}")
    print(f"  {'True Bad':>12} {r.fn:>10} {r.tp:>10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Layer 1 - Heuristic Filter")
    parser.add_argument("--samples", type=int, default=2000, help="Numero total de muestras")
    parser.add_argument("--data", type=str, default=None, help="Ruta a train.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = args.data or os.path.join(project_dir, "..", "data", "processed", "splits", "train.csv")

    print(f"Dataset: {data_path}")
    print(f"Muestras: {args.samples} (balanced 50/50 good/bad)")
    print(f"Seed: {args.seed}")

    try:
        texts, labels = load_data(data_path, args.samples, args.seed)
        n_bad = sum(labels)
        n_good = len(labels) - n_bad
        print(f"Loaded: {n_good} good + {n_bad} bad = {len(texts)} total\n")
    except FileNotFoundError:
        print(f"ERROR: No se encuentra el dataset en {data_path}")
        print("Por favor, entrena el modelo primero o especifica --data con la ruta correcta")
        sys.exit(1)

    # El HeuristicFilter integra todas las fases (regex, encoding, perplexity, TF-IDF)
    configs = [
        ("HeuristicFilter (todas las fases)", HeuristicFilter()),
    ]

    results = []
    for name, filt in configs:
        print(f"Evaluando: {name} ...", flush=True)
        try:
            r = evaluate(filt, texts, labels, name)
            results.append(r)
            print_result(r)
            print_confusion_matrix(r)
        except Exception as e:
            print(f"ERROR evaluando {name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  RESUMEN COMPARATIVO")
    print(f"{'=' * 60}")
    print(f"  {'Config':<30} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Acc':>7} {'Avg ms':>8} {'p95 ms':>8}")
    print(f"  {'-' * 30} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 8}")
    for r in results:
        print(f"  {r.config_name:<30} {r.precision:>7.4f} {r.recall:>7.4f} {r.f1:>7.4f} {r.accuracy:>7.4f} {r.avg_latency_ms:>8.2f} {r.p95_latency_ms:>8.2f}")
    print()


if __name__ == "__main__":
    main()
