import os
import sys
import time
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "models", "distilbert_sentinel", "checkpoint-22797")
DATA_PATH = os.path.join(_SCRIPT_DIR, "..", "Data", "def_combined_prompts_dataset.csv")
MAX_LENGTH = 256
BATCH_SIZE = 64
SAMPLE_SIZE = 100

LABEL_MAP = {0: "benign", 1: "injection"}
LABEL_MAP_REV = {"good": 0, "bad": 1}


class PromptDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def main():
    print(f"Cargando modelo desde {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Modelo cargado en {device}")

    print(f"Cargando dataset desde {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep=";", on_bad_lines="skip")
    df = df[df["label"].isin(["good", "bad"])].copy()
    print(f"Dataset limpio: {len(df)} muestras")

    df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
    print(f"Muestra de test: {len(df_sample)} muestras")

    y_true = df_sample["label"].map(LABEL_MAP_REV).values

    dataset = PromptDataset(df_sample["prompt"], tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    print("Evaluando modelo...")
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(loader)} procesado...")

    elapsed = time.time() - t0
    print(f"Evaluación completada en {elapsed:.1f}s")

    y_pred = all_preds

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nMatriz de confusión:")
    print(f"                 Pred benign  Pred injection")
    print(f"  Real benign    {cm[0][0]:>10}  {cm[0][1]:>13}")
    print(f"  Real injection {cm[1][0]:>10}  {cm[1][1]:>13}")
    print("\nReporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=["benign (good)", "injection (bad)"]))


if __name__ == "__main__":
    main()
