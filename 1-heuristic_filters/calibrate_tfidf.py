"""
Calibracion del modelo TF-IDF para el filtro heuristico.

Entrena un modelo TF-IDF + Regresion Logistica con el dataset,
evalua metricas y serializa el modelo para uso en runtime.

Uso:
    python calibrate_tfidf.py [--data_dir ../Data] [--output tfidf_model.joblib]
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from heuristic_filter import TFIDFBaseline


def load_dataset(data_dir: str):
    """Carga train.csv y lo divide en train/test (80/20)."""
    csv.field_size_limit(2147483647)

    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"No se encontro {train_path}")

    texts = []
    labels = []

    with open(train_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            if not text:
                continue
            if label == "good":
                texts.append(text)
                labels.append(0)
            elif label == "bad":
                texts.append(text)
                labels.append(1)

    print(f"Dataset cargado: {len(texts)} prompts ({sum(labels)} bad, {len(labels) - sum(labels)} good)")
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Calibrar modelo TF-IDF")
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "..", "Data"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "tfidf_model.joblib"))
    parser.add_argument("--stats_output", default=os.path.join(os.path.dirname(__file__), "tfidf_stats.json"))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print(f"Cargando dataset desde {data_dir}...")
    texts, labels = load_dataset(data_dir)

    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    print(f"Split: {len(X_train)} train, {len(X_test)} test")

    # Entrenar
    print("\nEntrenando TF-IDF + LogisticRegression...")
    model = TFIDFBaseline(max_features=20000, ngram_range=(1, 2), C=1.0)
    model.train(X_train, y_train)
    print("Entrenamiento completado")

    # Evaluar
    print("\nEvaluando en test set...")
    metrics = model.evaluate(X_test, y_test)

    accuracy = metrics["roc_auc"]  # ROC-AUC
    from sklearn.metrics import accuracy_score, f1_score
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n{'='*60}")
    print("METRICAS TF-IDF")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\n{metrics['classification_report']}")

    # Serializar modelo
    import joblib
    joblib.dump(model, args.output)
    print(f"Modelo guardado en: {args.output}")

    # Guardar stats
    stats = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "max_features": model.max_features,
        "ngram_range": list(model.ngram_range),
    }
    with open(args.stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats guardadas en: {args.stats_output}")


if __name__ == "__main__":
    main()
