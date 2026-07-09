"""
Combina dos CSVs de prompts, elimina duplicados y guarda el resultado.

Uso:
    python merge_prompts.py archivo1.csv archivo2.csv output.csv

Formatos soportados:
  - Separador coma o punto y coma
  - Con o sin cabecera 'prompt' (o 'prompt;label;source')
  - La comparación de duplicados ignora mayúsculas/minúsculas y espacios extra
"""

import csv
import sys
import os


def detect_separator(filepath):
    with open(filepath, encoding="utf-8") as f:
        sample = f.read(4096)
    return ";" if sample.count(";") > sample.count(",") else ","


def read_csv(filepath):
    sep = detect_separator(filepath)
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        # Si no hay cabecera reconocible, tratar primera columna como 'prompt'
        fieldnames = reader.fieldnames or []
        first = fieldnames[0].strip().lower() if fieldnames else ""

        if first not in ("prompt",):
            # Sin cabecera: releer como lista
            f.seek(0)
            raw = csv.reader(f, delimiter=sep)
            rows = []
            for row in raw:
                if row and row[0].strip():
                    rows.append({
                        "prompt": row[0].strip(),
                        "label": row[1].strip() if len(row) > 1 else "good",
                        "source": row[2].strip() if len(row) > 2 else "AI",
                    })
            return rows

        rows = []
        for row in reader:
            prompt = row.get("prompt", "").strip()
            if prompt:
                rows.append({
                    "prompt": prompt,
                    "label": row.get("label", "good").strip() or "good",
                    "source": row.get("source", "AI").strip() or "AI",
                })
        return rows


def merge(file1, file2, output):
    rows1 = read_csv(file1)
    rows2 = read_csv(file2)

    seen = {}  # clave normalizada -> primera fila encontrada

    for row in rows1 + rows2:
        key = row["prompt"].lower().strip()
        if key not in seen:
            seen[key] = row

    merged = list(seen.values())

    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["prompt", "label", "source"])
        for row in merged:
            writer.writerow([row["prompt"], row["label"], row["source"]])

    total1 = len(rows1)
    total2 = len(rows2)
    duplicates = (total1 + total2) - len(merged)

    print(f"📄 Archivo 1: {total1} prompts")
    print(f"📄 Archivo 2: {total2} prompts")
    print(f"🗑️  Duplicados eliminados: {duplicates}")
    print(f"✅ Resultado: {len(merged)} prompts únicos → {output}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python merge_prompts.py <archivo1.csv> <archivo2.csv> <output.csv>")
        sys.exit(1)

    f1, f2, out = sys.argv[1], sys.argv[2], sys.argv[3]

    for f in (f1, f2):
        if not os.path.exists(f):
            print(f"❌ No se encuentra: {f}")
            sys.exit(1)

    merge(f1, f2, out)