"""
Test de Precision del Pipeline de Deteccion de Prompt Injection
==============================================================

Este script evalua la efectividad del pipeline completo (3 capas) ejecutandolo
N veces sobre un dataset y calculando metricas de clasificacion.

Formato del dataset (CSV con separador ";"):
    - Columna 1: prompt (texto a analizar)
    - Columna 2: label ("good" para benignos, "bad" para inyectados)

Uso:
    python test_pipeline.py --dataset <RUTA_DATASET> --iterations <NUM> --mistral_key <KEY> [--groq_key <KEY>]

Ejemplo:
    python test_pipeline.py --dataset ../Data/def_combined_prompts_dataset.csv --iterations 100 --mistral_key sk-xxx --groq_key gsk_xxx

Metricas calculadas:
    - True Positives (TP):  Prompt malicioso correctamente detectado como malicioso
    - False Positives (FP): Prompt benigno incorrectamente detectado como malicioso
    - True Negatives (TN):  Prompt benigno correctamente detectado como benigno
    - False Negatives (FN): Prompt malicioso no detectado (paso como benigno)
    - Precision: TP / (TP + FP) - Que tan confiables son las detecciones positivas
    - Recall:    TP / (TP + FN) - Que tan bien detecta todos los maliciosos
    - Accuracy:  (TP + TN) / Total - Acertado general
    - F1-Score:  2 * (Precision * Recall) / (Precision + Recall) - Balance Precision/Recall
"""

import os
import sys
import argparse
import time
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))

from pipeline import run_pipeline

LABEL_MAP_REV = {"good": 0, "bad": 1}
LABEL_NAMES = {0: "benign", 1: "injection"}


def main():
    parser = argparse.ArgumentParser(
        description="Evalua la precision del pipeline de deteccion de Prompt Injection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Ruta al archivo CSV del dataset (separador ';')"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Numero de muestras a evaluar del dataset"
    )
    parser.add_argument(
        "--mistral_key",
        type=str,
        required=True,
        help="API Key de Mistral"
    )
    parser.add_argument(
        "--groq_key",
        type=str,
        default=None,
        help="API Key de Groq (fallback)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para el muestreo aleatorio (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TEST DE PRECISION DEL PIPELINE")
    print("=" * 60)
    print(f"Dataset:    {args.dataset}")
    print(f"Iteraciones: {args.iterations}")
    print(f"Seed:       {args.seed}")
    print()

    # Cargar dataset
    print("Cargando dataset...")
    df = pd.read_csv(args.dataset, sep=";", on_bad_lines="skip")
    df = df[df["label"].isin(["good", "bad"])].copy()
    print(f"Dataset limpio: {len(df)} muestras validas")

    # Muestrear
    n_samples = min(args.iterations, len(df))
    if n_samples < args.iterations:
        print(f"ADVERTENCIA: Solo hay {len(df)} muestras, se usaran {n_samples}")
    df_sample = df.sample(n=n_samples, random_state=args.seed).reset_index(drop=True)
    print(f"Muestras a evaluar: {len(df_sample)}")
    print()

    y_true = []
    y_pred = []
    start_time = time.time()

    for idx, row in df_sample.iterrows():
        prompt = row["prompt"]
        true_label = LABEL_MAP_REV[row["label"]]

        print(f"[{idx + 1}/{len(df_sample)}] Evaluando prompt de {LABEL_NAMES[true_label]}...")
        print(f"  Prompt: \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")

        result = run_pipeline(prompt, args.mistral_key, groq_key=args.groq_key)

        pred_label = 1 if result["final_verdict"] == "BLOCKED" else 0

        y_true.append(true_label)
        y_pred.append(pred_label)

        tp_fp = "TP" if true_label == 1 and pred_label == 1 else ""
        fp = "FP" if true_label == 0 and pred_label == 1 else ""
        fn = "FN" if true_label == 1 and pred_label == 0 else ""
        tn = "TN" if true_label == 0 and pred_label == 0 else ""
        tag = tp_fp or fp or fn or tn

        print(f"  Real: {LABEL_NAMES[true_label]} | Pred: {LABEL_NAMES[pred_label]} | {tag}")
        print()

    elapsed = time.time() - start_time

    # Calcular metricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn_count = cm[0][0]
    fp_count = cm[0][1]
    fn_count = cm[1][0]
    tp_count = cm[1][1]

    # Resultados
    print("=" * 60)
    print("RESULTADOS DE EVALUACION")
    print("=" * 60)
    print(f"Tiempo total:      {elapsed:.1f}s")
    print(f"Muestras evaluadas: {len(y_true)}")
    print()
    print("Matriz de confusion:")
    print(f"                   Pred benign  Pred injection")
    print(f"  Real benign      {tn_count:>10}  {fp_count:>13}")
    print(f"  Real injection   {fn_count:>10}  {tp_count:>13}")
    print()
    print(f"Verdaderos positivos (TP): {tp_count}")
    print(f"Falsos positivos (FP):     {fp_count}")
    print(f"Verdaderos negativos (TN): {tn_count}")
    print(f"Falsos negativos (FN):     {fn_count}")
    print()
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["benign (good)", "injection (bad)"]))


if __name__ == "__main__":
    main()
