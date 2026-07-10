"""
Combina dos CSVs de prompts, elimina duplicados, filtra por idioma y guarda el resultado.

Uso:
    python merge_prompts.py archivo1.csv archivo2.csv output.csv
"""

import csv
import sys
import os



def merge(file1, file2, output):
    print(f"Leyendo {file1}...")
    with open(file1, 'r', encoding='utf-8') as f:
        rows1 = list(csv.DictReader(f, delimiter=';'))
    print(f"Leyendo {file2}...")
    with open(file2, 'r', encoding='utf-8') as f:
        rows2 = list(csv.DictReader(f, delimiter=';'))

    total_raw = len(rows1) + len(rows2)

    # Deduplicar
    seen = {}
    for row in rows1 + rows2:
        key = row["prompt"].lower().strip()
        if key not in seen:
            seen[key] = row
    deduped = list(seen.values())
    duplicates = total_raw - len(deduped)

    # Guardar
    final = deduped
    with open(output, "w", encoding="utf-8", newline="") as f:
        if rows1:
            fieldnames = list(rows1[0].keys())
        elif rows2:
            fieldnames = list(rows2[0].keys())
        else:
            fieldnames = ["prompt"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(final)

    # Resumen
    print(f"\n📄 Archivo 1:              {len(rows1):>6} prompts")
    print(f"📄 Archivo 2:              {len(rows2):>6} prompts")
    print(f"🗑️  Duplicados eliminados:  {duplicates:>6}")
    print(f"✅ Resultado final:         {len(final):>6} prompts únicos → {output}")


if __name__ == "__main__":
    f1 = "temp_combined_prompts_dataset.csv"
    f2 = "combined_prompts_dataset.csv"
    out = "def_combined_prompts_dataset.csv"

    for f in (f1, f2):
        if not os.path.exists(f):
            print(f"❌ No se encuentra: {f}")
            sys.exit(1)

    merge(f1, f2, out)