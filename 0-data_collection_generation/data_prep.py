"""
Prepara el dataset de entrenamiento combinando:
  - Tu CSV de ~10k prompts maliciosos
  - Prompts benignos de fuentes públicas (Alpaca / OASST1), para tener clases balanceadas

IMPORTANTE: sin negativos (benignos) reales el clasificador no aprende la tarea,
aprende a distinguir "estilo de dataset A" vs "estilo de dataset B", lo cual
no generaliza y es un error clásico en estos proyectos.

Ejecutar:
    python data_prep.py --malicious_csv /ruta/a/tu_dataset.csv \
                         --prompt_col prompt --label_col label \
                         --out_dir ./data
"""

import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_malicious(csv_path: str, prompt_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert prompt_col in df.columns, (
        f"Columna '{prompt_col}' no encontrada. Columnas disponibles: {list(df.columns)}"
    )
    df = df[[prompt_col]].rename(columns={prompt_col: "text"})
    df["label"] = 1  # malicioso
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    return df


def load_benign(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Carga prompts benignos de Alpaca (instrucciones genéricas de usuario).
    Alternativas si quieres más diversidad: 'OpenAssistant/oasst1', 'anon8231489123/ShareGPT_Vicuna_unfiltered'
    """
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    df = ds.to_pandas()[["instruction"]].rename(columns={"instruction": "text"})
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    df = df.sample(n=min(n_samples, len(df)), random_state=seed)
    df["label"] = 0  # benigno
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malicious_csv", required=True)
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mal_df = load_malicious(args.malicious_csv, args.prompt_col)
    print(f"Prompts maliciosos cargados: {len(mal_df)}")

    ben_df = load_benign(n_samples=len(mal_df), seed=args.seed)
    print(f"Prompts benignos cargados: {len(ben_df)}")

    full_df = pd.concat([mal_df, ben_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # split estratificado en train/val/test para que las proporciones de clase se mantengan
    train_df, temp_df = train_test_split(
        full_df, test_size=(args.test_size + args.val_size),
        stratify=full_df["label"], random_state=args.seed,
    )
    val_ratio = args.val_size / (args.test_size + args.val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio),
        stratify=temp_df["label"], random_state=args.seed,
    )

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    train_df.to_csv(f"{args.out_dir}/train.csv", index=False)
    val_df.to_csv(f"{args.out_dir}/val.csv", index=False)
    test_df.to_csv(f"{args.out_dir}/test.csv", index=False)

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Guardado en {args.out_dir}/")


if __name__ == "__main__":
    main()
