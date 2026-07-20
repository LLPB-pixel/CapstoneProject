"""
Fine-tuning de DistilBERT para detección de prompts maliciosos.

Requiere GPU para tiempos razonables (Colab con T4 gratuito es suficiente
para 10-20k ejemplos). En CPU funciona pero es lento.

Ejecutar:
    python train_distilbert.py --data_dir ./data --epochs 3 --batch_size 16
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    import wandb  # noqa: F401
except ImportError:
    wandb = None


MODEL_NAME = "distilbert-base-uncased"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


def load_split(path: str, tokenizer, max_length: int = 256) -> Dataset:
    df = pd.read_csv(path)

    if "prompt" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"prompt": "text"})

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"El archivo {path} debe contener columnas 'text' y 'label'.")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["text"] = df["text"].apply(lambda x: x.strip())
    df["text"] = df["text"].replace(r"^\s*$", "", regex=True)

    label_values = sorted(df["label"].dropna().astype(str).unique())
    label_mapping = {label: int(idx) for idx, label in enumerate(label_values)}
    df["label"] = df["label"].fillna(label_values[0] if label_values else "0").astype(str)
    df["label"] = df["label"].map(label_mapping)
    df = df.dropna(subset=["label"])

    ds = Dataset.from_pandas(df)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    return ds.map(tokenize, batched=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(PROJECT_ROOT / "Data"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "models" / "distilbert_sentinel"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--wandb_project", default=os.getenv("WANDB_PROJECT", "capstone-distilbert"))
    ap.add_argument("--wandb_entity", default=os.getenv("WANDB_ENTITY"))
    ap.add_argument("--wandb_run_name", default=os.getenv("WANDB_RUN_NAME"))
    ap.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default=os.getenv("WANDB_MODE", "online"),
    )
    ap.add_argument(
        "--save",
        choices=["yes", "no"],
        default="no",
        help="Guardar el modelo y tokenizador antes de iniciar el entrenamiento.",
    )
    args = ap.parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.out_dir = resolve_path(args.out_dir)

    if args.wandb_mode != "disabled":
        if wandb is None:
            raise RuntimeError("wandb no está instalado. Instala it con: pip install wandb")
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_run_name:
            os.environ["WANDB_RUN_NAME"] = args.wandb_run_name

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_ds = load_split(f"{args.data_dir}/train.csv", tokenizer, args.max_length)
    val_ds = load_split(f"{args.data_dir}/val.csv", tokenizer, args.max_length)
    test_ds = load_split(f"{args.data_dir}/test.csv", tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"] if args.wandb_mode != "disabled" else "none",
        run_name=args.wandb_run_name or f"distilbert-{args.epochs}e",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"distilbert-{args.epochs}e",
            config={
                "model_name": MODEL_NAME,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_length": args.max_length,
            },
        )

    if args.save == "yes":
        print(f"\n[INFO] Guardando modelo y tokenizador antes de cualquier iteración en: {args.out_dir}")
        trainer.save_model(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)

    try:
        trainer.train()
    finally:
        if args.wandb_mode != "disabled":
            wandb.finish()

    print("\n=== Evaluación en TEST (holdout final) ===")
    test_metrics = trainer.evaluate(test_ds)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"\nModelo guardado en {args.out_dir}")


if __name__ == "__main__":
    main()
