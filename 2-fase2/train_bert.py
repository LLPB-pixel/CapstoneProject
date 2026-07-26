#!/usr/bin/env python3
"""
Fine-tuning de DeBERTa-v3-base para detección de prompts maliciosos.

Parámetros basados en el paper InjecGuard (Sec 5.1):
  - Backbone: microsoft/deberta-v3-base
  - Batch size: 32
  - Epochs: 3
  - Learning rate: 2e-5
  - Warmup: 100 steps (absoluto)
  - Max token length: 512
  - Optimizer: Adam (weight_decay=0)
  - Scheduler: linear

Ejecutar (entrenamiento básico):
    python train_distilbert.py --data_dir ../Data --epochs 3 --batch_size 32

Ejecutar con MOF (pipeline completo InjecGuard):
    python train_distilbert.py --data_dir ../Data --mof
"""

import argparse
import json
import os
import random
import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

csv.field_size_limit(sys.maxsize)

try:
    import wandb
except ImportError:
    wandb = None


MODEL_NAME = "microsoft/deberta-v3-base"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(candidate.resolve())


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


def load_split(path: str, tokenizer, max_length: int = 512) -> Dataset:
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
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return ds.map(tokenize, batched=True)


# ---------------------------------------------------------------------------
# MOF: Mitigating Over-defense for Free
# ---------------------------------------------------------------------------

ATTACK_LABEL = 1


def token_recheck(model, tokenizer, batch_size: int = 256) -> list[str]:
    """Paso 2: Token-wised recheck. Identifica tokens sesgados que el modelo
    clasifica como 'attack' cuando son tokens individuales (deberían ser benignos).

    Devuelve la lista de textos de tokens sesgados.
    """
    print("\n[MOF] === Paso 2: Token-wised recheck ===")
    model.eval()
    device = next(model.parameters()).device

    vocab = tokenizer.get_vocab()
    vocab_items = list(vocab.items())
    random.shuffle(vocab_items)

    biased_tokens = []
    total = len(vocab_items)

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = vocab_items[i : i + batch_size]
            token_texts = [text for _, text in batch]
            token_ids = [tid for tid, _ in batch]

            encodings = tokenizer(
                token_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32,
            ).to(device)

            outputs = model(**encodings)
            preds = outputs.logits.argmax(dim=1).cpu().tolist()

            for tid, text, pred in zip(token_ids, token_texts, preds):
                if pred == ATTACK_LABEL:
                    biased_tokens.append(text)

            progress = min(i + batch_size, total)
            print(f"\r[MOF] Recheck: {progress}/{total} tokens ({100*progress//total}%)", end="", flush=True)

    print(f"\n[MOF] Tokens sesgados encontrados: {len(biased_tokens)}")
    return biased_tokens


def generate_mof_samples(
    biased_tokens: list[str],
    num_samples: int = 1000,
    output_path: str | None = None,
) -> list[dict]:
    """Paso 3: Genera muestras benignas que contienen tokens sesgados.
    Usa templates predefinidos + tokens sesgados para generar oraciones naturales.

    Devuelve lista de dicts con keys 'text' y 'label' (label=0, benign).
    """
    print(f"\n[MOF] === Paso 3: Generar {num_samples} muestras benignas ===")

    if output_path and Path(output_path).exists():
        print(f"[MOF] Cargando datos generados previamente de {output_path}")
        df = pd.read_csv(output_path)
        return df.to_dict("records")

    # Templates para 1 token, 2 tokens y 3 tokens
    TEMPLATES_1 = [
        "Please {w1} the instructions carefully.",
        "I need to {w1} this problem quickly.",
        "Can you help me {w1} the document?",
        "She decided to {w1} the old rules.",
        "We should {w1} the guidelines given.",
        "He tried to {w1} the system errors.",
        "The teacher asked us to {w1} the essay.",
        "They will {w1} the meeting schedule.",
        "Let me {w1} this before proceeding.",
        "You must {w1} the safety protocols.",
        "I forgot to {w1} the settings.",
        "Please {w1} your homework by Friday.",
        "We need to {w1} the budget carefully.",
        "Can you {w1} me with this task?",
        "She will {w1} the project tomorrow.",
        "He always {w1} the important details.",
        "They plan to {w1} the new policy.",
        "I want to {w1} this opportunity.",
        "The team will {w1} the proposal.",
        "Please {w1} the terms before signing.",
    ]

    TEMPLATES_2 = [
        "Please {w1} the {w2} in the report.",
        "I need to {w1} both the {w2} and the data.",
        "Can you {w1} the {w2} before submitting?",
        "She decided to {w1} the {w2} issues.",
        "We should {w1} the {w2} guidelines.",
        "He tried to {w1} the {w2} errors.",
        "The teacher asked us to {w1} the {w2} section.",
        "They will {w1} the {w2} schedule.",
        "Let me {w1} the {w2} first.",
        "You must {w1} both {w2} and safety.",
        "I forgot to {w1} the {w2} settings.",
        "Please {w1} your {w2} assignment.",
        "We need to {w1} the {w2} budget.",
        "Can you {w1} the {w2} details?",
        "She will {w1} the {w2} project.",
        "He always checks the {w2} before he {w1} anything.",
        "They plan to {w1} the {w2} plan.",
        "I want to {w1} this {w2} carefully.",
        "The team will {w1} the {w2} review.",
        "Please {w1} both {w2} and the results.",
    ]

    TEMPLATES_3 = [
        "Please {w1} the {w2} and {w3} sections.",
        "I need to {w1} the {w2}, {w3} and data.",
        "Can you {w1} the {w2} with {w3} in mind?",
        "She decided to {w1} the {w2} and {w3} issues.",
        "We should {w1} the {w2}, {w3} guidelines.",
        "He tried to {w1} the {w2} and {w3} errors.",
        "The teacher asked us to {w1} the {w2}, {w3} parts.",
        "They will {w1} the {w2}, {w3} schedule.",
        "Let me {w1} the {w2} and {w3} first.",
        "You must {w1} the {w2}, {w3} and safety.",
        "I forgot to {w1} the {w2}, {w3} settings.",
        "Please {w1} your {w2} and {w3} tasks.",
        "We need to {w1} the {w2}, {w3} budget.",
        "Can you {w1} the {w2} with {w3}?",
        "She will {w1} the {w2} and {w3} project.",
        "He always {w1} the {w2}, {w3} details.",
        "They plan to {w1} the {w2}, {w3} plan.",
        "I want to {w1} the {w2} and {w3} carefully.",
        "The team will {w1} the {w2}, {w3} review.",
        "Please {w1} the {w2}, {w3} and results.",
    ]

    samples = []
    seen = set()
    rng = random.Random(42)

    while len(samples) < num_samples:
        n_tokens = rng.choice([1, 1, 2, 2, 2, 3, 3])
        chosen = rng.sample(biased_tokens, min(n_tokens, len(biased_tokens)))

        if n_tokens == 1:
            tpl = rng.choice(TEMPLATES_1)
            text = tpl.format(w1=chosen[0])
        elif n_tokens == 2:
            tpl = rng.choice(TEMPLATES_2)
            text = tpl.format(w1=chosen[0], w2=chosen[1])
        else:
            tpl = rng.choice(TEMPLATES_3)
            text = tpl.format(w1=chosen[0], w2=chosen[1], w3=chosen[2])

        if text not in seen:
            seen.add(text)
            samples.append({"text": text, "label": 0})

    print(f"[MOF] Total muestras generadas: {len(samples)}")

    if output_path and samples:
        pd.DataFrame(samples).to_csv(output_path, index=False)
        print(f"[MOF] Guardadas en {output_path}")

    return samples


def create_mof_dataset(
    original_df: pd.DataFrame,
    mof_samples: list[dict],
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """Crea dataset augmentado con muestras MOF etiquetadas como benign (0)."""
    mof_df = pd.DataFrame(mof_samples)
    combined = pd.concat([original_df, mof_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    ds = Dataset.from_pandas(combined)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return ds.map(tokenize_fn, batched=True)


def train_model(
    model,
    tokenizer,
    train_ds,
    val_ds,
    test_ds,
    args,
    wandb_run_suffix: str = "",
    output_dir: str | None = None,
) -> tuple:
    """Entrena un modelo y devuelve (trainer, test_metrics)."""
    out = output_dir or args.out_dir
    total_steps = (len(train_ds) // args.batch_size // args.grad_accum) * args.epochs

    training_args = TrainingArguments(
        output_dir=out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        gradient_accumulation_steps=args.grad_accum,
        fp16=False,
        bf16=False,
        report_to=["wandb"] if args.wandb_mode != "disabled" else "none",
        run_name=args.wandb_run_name or f"deberta-v3-{args.epochs}e{wandb_run_suffix}",
        dataloader_num_workers=4,
        seed=42,
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
            name=args.wandb_run_name or f"deberta-v3-{args.epochs}e{wandb_run_suffix}",
            config={
                "model_name": MODEL_NAME,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_length": args.max_length,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "total_steps": total_steps,
            },
        )

    trainer.train()

    print("\n=== Evaluación en TEST (holdout final) ===")
    test_metrics = trainer.evaluate(test_ds)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"\nModelo guardado en {out}")

    return trainer, test_metrics


def run_mof_pipeline(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    val_ds,
    test_ds,
    args,
):
    """Pipeline MOF completo (pasos 2-4 del paper Sec 4.2).

    2. Token-wised recheck → tokens sesgados
    3. Generar datos benignos con tokens sesgados
    4. Reentrenar desde cero
    """
    mof_dir = Path(args.out_dir) / "mof_artifacts"
    mof_dir.mkdir(parents=True, exist_ok=True)

    biased_tokens_path = str(mof_dir / "biased_tokens.json")
    generated_csv_path = str(mof_dir / "generated_mof.csv")

    # Paso 2: Token recheck (con cache)
    if Path(biased_tokens_path).exists():
        print(f"[MOF] Cargando tokens sesgados de {biased_tokens_path}")
        with open(biased_tokens_path) as f:
            biased_tokens = json.load(f)
        print(f"[MOF] Tokens sesgados: {len(biased_tokens)}")
    else:
        biased_tokens = token_recheck(model, tokenizer)
        with open(biased_tokens_path, "w") as f:
            json.dump(biased_tokens, f)
        print(f"[MOF] Tokens sesgados guardados en {biased_tokens_path}")

    if not biased_tokens:
        print("[MOF] No se encontraron tokens sesgados. Saltando generación.")
        return

    # Paso 3: Generar muestras benignas
    mof_samples = generate_mof_samples(
        biased_tokens,
        num_samples=args.mof_num_samples,
        output_path=generated_csv_path,
    )

    if not mof_samples:
        print("[MOF] No se generaron muestras. Saltando reentrenamiento.")
        return

    # Paso 4: Reentrenar desde cero
    print(f"\n[MOF] === Paso 4: Reentrenar desde cero con {len(mof_samples)} muestras MOF ===")

    train_ds_mof = create_mof_dataset(train_df, mof_samples, tokenizer, args.max_length)
    print(f"[MOF] Dataset augmentado: {len(train_ds_mof)} muestras (original: {len(train_df)}, MOF: {len(mof_samples)})")

    del model
    torch.cuda.empty_cache()

    new_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    new_out_dir = str(Path(args.out_dir) / "mof_retrained")

    if args.wandb_mode != "disabled":
        wandb.finish()

    train_model(
        new_model,
        tokenizer,
        train_ds_mof,
        val_ds,
        test_ds,
        args,
        wandb_run_suffix="-mof",
        output_dir=new_out_dir,
    )


def main():
    ap = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3-base para detección de prompts maliciosos")
    ap.add_argument("--data_dir", default=str(PROJECT_ROOT / "Data"))
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "models" / "distilbert_sentinel"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.0)
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

    # MOF args
    ap.add_argument(
        "--mof",
        action="store_true",
        help="Activar pipeline MOF (Mitigating Over-defense for Free).",
    )
    ap.add_argument("--mof_num_samples", type=int, default=1000, help="Número de muestras benignas a generar para MOF.")


    args = ap.parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.out_dir = resolve_path(args.out_dir)

    if args.wandb_mode != "disabled":
        if wandb is None:
            raise RuntimeError("wandb no está instalado. Instálalo con: pip install wandb")
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_run_name:
            os.environ["WANDB_RUN_NAME"] = args.wandb_run_name

    print(f"[INFO] Modelo: {MODEL_NAME}")
    print(f"[INFO] Epochs: {args.epochs}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Grad accum: {args.grad_accum}")
    print(f"[INFO] Effective batch: {args.batch_size * args.grad_accum}")
    print(f"[INFO] Learning rate: {args.lr}")
    print(f"[INFO] Max length: {args.max_length}")
    print(f"[INFO] Warmup steps: {args.warmup_steps}")
    print(f"[INFO] Weight decay: {args.weight_decay}")
    print(f"[INFO] Data dir: {args.data_dir}")
    print(f"[INFO] Output dir: {args.out_dir}")
    print(f"[INFO] MOF: {'ACTIVADO' if args.mof else 'desactivado'}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print(f"\n[INFO] Cargando splits...")
    train_ds = load_split(f"{args.data_dir}/train.csv", tokenizer, args.max_length)
    val_ds = load_split(f"{args.data_dir}/val.csv", tokenizer, args.max_length)
    test_ds = load_split(f"{args.data_dir}/test.csv", tokenizer, args.max_length)
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    if args.save == "yes":
        print(f"\n[INFO] Guardando modelo y tokenizador antes de cualquier iteración en: {args.out_dir}")
        trainer_dummy = Trainer(model=model)
        trainer_dummy.save_model(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)

    # Paso 1: Entrenamiento estándar
    print("\n[INFO] === Paso 1: Entrenamiento estándar ===")
    train_model(model, tokenizer, train_ds, val_ds, test_ds, args)

    if args.wandb_mode != "disabled":
        wandb.finish()

    # MOF: pasos 2-4
    if args.mof:
        # Recargar modelo entrenado para token recheck
        print("\n[MOF] Recargando modelo entrenado para token recheck...")
        trained_model = AutoModelForSequenceClassification.from_pretrained(args.out_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model = trained_model.to(device)

        # Cargar train_df original (sin tokenizar) para crear dataset augmentado
        train_df = pd.read_csv(f"{args.data_dir}/train.csv")
        if "prompt" in train_df.columns and "text" not in train_df.columns:
            train_df = train_df.rename(columns={"prompt": "text"})

        run_mof_pipeline(
            trained_model,
            tokenizer,
            train_df,
            val_ds,
            test_ds,
            args,
        )


if __name__ == "__main__":
    main()
