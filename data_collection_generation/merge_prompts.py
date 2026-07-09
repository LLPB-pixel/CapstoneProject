"""
Combina dos CSVs de prompts, elimina duplicados, filtra por idioma y guarda el resultado.

Uso:
    python merge_prompts.py archivo1.csv archivo2.csv output.csv
"""

import csv
import re
import sys
import os
import matplotlib.pyplot as plt




def plot_histograms(rows, output_img="prompts_histogram.png"):
    benign_lengths    = [len(r["prompt"]) for r in rows if r["label"].strip().lower() in ('good', 'benign', 'safe', '0')]
    malignant_lengths = [len(r["prompt"]) for r in rows if r["label"].strip().lower() not in ('good', 'benign', 'safe', '0')]

    if not benign_lengths and not malignant_lengths:
        print("Sin datos suficientes para histogramas.")
        return

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(benign_lengths, bins=50, color='green', alpha=0.7)
    plt.title("Tamaño Prompts Benignos")
    plt.xlabel("Longitud (caracteres)")
    plt.ylabel("Frecuencia")

    plt.subplot(1, 2, 2)
    plt.hist(malignant_lengths, bins=50, color='red', alpha=0.7)
    plt.title("Tamaño Prompts Malignos")
    plt.xlabel("Longitud (caracteres)")
    plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"\n📊 Histogramas guardados en '{output_img}'.")


def merge(file1, file2, output):
    print(f"Leyendo {file1}...")
    rows1 = csv.read_csv(file1)
    print(f"Leyendo {file2}...")
    rows2 = csv.read_csv(file2)

    total_raw = len(rows1) + len(rows2)

    # Deduplicar
    seen = {}
    for row in rows1 + rows2:
        key = row["prompt"].lower().strip()
        if key not in seen:
            seen[key] = row
    deduped = list(seen.values())
    duplicates = total_raw - len(deduped)

    # Filtrar y limpiar
    final = deduped
    # Guardar
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["prompt", "label", "source"])
        for row in final:
            writer.writerow([row["prompt"], row["label"], row["source"]])

    # Resumen
    print(f"\n📄 Archivo 1:              {len(rows1):>6} prompts")
    print(f"📄 Archivo 2:              {len(rows2):>6} prompts")
    print(f"🗑️  Duplicados eliminados:  {duplicates:>6}")
    print(f"✅ Resultado final:         {len(final):>6} prompts únicos → {output}")


    plot_histograms(final)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python merge_prompts.py <archivo1.csv> <archivo2.csv> <output.csv>")
        sys.exit(1)

    f1 = "combined_prompts_dataset.csv"
    f2 = "IA_defs.csv"
    out = "def_combined_prompts_dataset.csv"

    for f in (f1, f2):
        if not os.path.exists(f):
            print(f"❌ No se encuentra: {f}")
            sys.exit(1)

    merge(f1, f2, out)