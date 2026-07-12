"""
Baseline: TF-IDF + Regresión Logística.

Sirve de referencia rápida antes de fine-tunear DistilBERT: si el baseline ya
saca 0.95 F1, es señal de que el dataset tiene "shortcuts" léxicos fáciles
(p.ej. palabras muy marcadas como "ignore", "DAN", "jailbreak") y que el
transformer probablemente aprenderá lo mismo salvo que cuides el dataset.
Es exactamente la comparación que se espera en un writeup serio.

Ejecutar:
    python baseline_tfidf.py --data_dir ./data
"""

import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="../models")
    ap.add_argument("--max_features", type=int, default=20000)
    args = ap.parse_args()

    train_df = pd.read_csv(f"{args.data_dir}/train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/val.csv")
    test_df = pd.read_csv(f"{args.data_dir}/test.csv")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),      # unigramas + bigramas: captura frases tipo "ignore previous"
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X_train, train_df["label"])

    for name, X, y in [("VAL", X_val, val_df["label"]), ("TEST", X_test, test_df["label"])]:
        preds = clf.predict(X)
        probs = clf.predict_proba(X)[:, 1]
        print(f"\n=== {name} ===")
        print(classification_report(y, preds, target_names=["benigno", "malicioso"]))
        print(f"ROC-AUC: {roc_auc_score(y, probs):.4f}")
        print(f"Matriz de confusión:\n{confusion_matrix(y, preds)}")

    # top features más predictivas de "malicioso" - útil para tu sección de análisis
    feature_names = vectorizer.get_feature_names_out()
    top_idx = clf.coef_[0].argsort()[-20:][::-1]
    print("\nTop 20 n-gramas más predictivos de prompt malicioso:")
    for i in top_idx:
        print(f"  {feature_names[i]:30s}  coef={clf.coef_[0][i]:.3f}")

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(vectorizer, f"{args.out_dir}/tfidf_vectorizer.joblib")
    joblib.dump(clf, f"{args.out_dir}/logreg_baseline.joblib")
    print(f"\nModelo guardado en {args.out_dir}/")


if __name__ == "__main__":
    main()
