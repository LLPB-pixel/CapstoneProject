"""
Calibracion del umbral de perplejidad para el filtro heuristico.

Carga prompts buenos y maliciosos del dataset, calcula su perplejidad
con GPT-2, y determina el umbral a partir del percentil 97.5 de los
prompts buenos.

Uso:
    python calibrate_perplexity.py [--data_dir ../Data] [--n_samples 1000] [--output perplexity_threshold.json]
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np

# Permitir importar PerplexityScorer desde la nueva estructura
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from prompt_guard.layers.layer1_heuristic.filter import PerplexityScorer


def load_prompts(data_dir: str, n_samples: int = 1000, seed: int = 42):
    """
    Carga prompts del dataset train.csv y devuelve dos listas muestreadas:
    good_prompts y bad_prompts.
    """
    csv.field_size_limit(2147483647)

    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"No se encontro {train_path}")

    good = []
    bad = []

    with open(train_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            if not text:
                continue
            if label == "good":
                good.append(text)
            elif label == "bad":
                bad.append(text)

    print(f"Dataset cargado: {len(good)} good, {len(bad)} bad")

    rng = random.Random(seed)
    if len(good) > n_samples:
        good = rng.sample(good, n_samples)
    if len(bad) > n_samples:
        bad = rng.sample(bad, n_samples)

    print(f"Samples seleccionados: {len(good)} good, {len(bad)} bad")
    return good, bad


def compute_perplexities_batch(scorer: PerplexityScorer, prompts: list) -> np.ndarray:
    """
    Calcula perplexity de una lista de prompts usando el PerplexityScorer.
    Procesa de uno en uno (el scorer ya interna GPT-2).
    """
    ppl = []
    for i, text in enumerate(prompts):
        try:
            score = scorer.score(text)
            ppl.append(score)
        except Exception:
            ppl.append(0.0)
        if (i + 1) % 200 == 0:
            print(f"  Procesados {i + 1}/{len(prompts)}...")
    return np.array(ppl)


def main():
    parser = argparse.ArgumentParser(description="Calibrar umbral de perplejidad")
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "..", "Data"))
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "perplexity_threshold.json"))
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print(f"Cargando dataset desde {data_dir}...")
    good_prompts, bad_prompts = load_prompts(data_dir, n_samples=args.n_samples)

    print("\nCargando modelo GPT-2...")
    scorer = PerplexityScorer(model_name="gpt2")

    print("\nCalculando perplejidad de prompts buenos...")
    good_ppl = compute_perplexities_batch(scorer, good_prompts)

    print("\nCalculando perplejidad de prompts maliciosos...")
    bad_ppl = compute_perplexities_batch(scorer, bad_prompts)

    # Calcular estadisticas
    threshold = float(np.percentile(good_ppl, 97.5))

    stats = {
        "threshold": round(threshold, 4),
        "p97_5_good": round(threshold, 4),
        "p99_good": round(float(np.percentile(good_ppl, 99)), 4),
        "mean_good": round(float(np.mean(good_ppl)), 4),
        "median_good": round(float(np.median(good_ppl)), 4),
        "std_good": round(float(np.std(good_ppl)), 4),
        "min_good": round(float(np.min(good_ppl)), 4),
        "max_good": round(float(np.max(good_ppl)), 4),
        "mean_bad": round(float(np.mean(bad_ppl)), 4),
        "median_bad": round(float(np.median(bad_ppl)), 4),
        "std_bad": round(float(np.std(bad_ppl)), 4),
        "min_bad": round(float(np.min(bad_ppl)), 4),
        "max_bad": round(float(np.max(bad_ppl)), 4),
        "n_good": len(good_ppl),
        "n_bad": len(bad_ppl),
        "model": "gpt2",
    }

    # Calcular cuantos bad prompts superan el umbral
    bad_above = int(np.sum(bad_ppl > threshold))
    good_above = int(np.sum(good_ppl > threshold))
    stats["bad_above_threshold"] = bad_above
    stats["bad_above_pct"] = round(bad_above / len(bad_ppl) * 100, 2)
    stats["good_above_threshold"] = good_above
    stats["good_above_pct"] = round(good_above / len(good_ppl) * 100, 2)

    # Guardar
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("ESTADISTICAS DE PERPLEJIDAD")
    print(f"{'='*60}")
    print(f"  Prompts buenos:  media={stats['mean_good']:.2f}  mediana={stats['median_good']:.2f}  std={stats['std_good']:.2f}")
    print(f"  Prompts malos:   media={stats['mean_bad']:.2f}  mediana={stats['median_bad']:.2f}  std={stats['std_bad']:.2f}")
    print(f"  Umbral (P97.5 good): {stats['threshold']:.2f}")
    print(f"  Bad prompts > umbral: {bad_above}/{len(bad_ppl)} ({stats['bad_above_pct']}%)")
    print(f"  Good prompts > umbral: {good_above}/{len(good_ppl)} ({stats['good_above_pct']}%)")
    print(f"\nGuardado en: {args.output}")


if __name__ == "__main__":
    main()
