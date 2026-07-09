"""
Convierte un CSV de prompts (formato imagen: columna 'prompt')
al formato destino: prompt;label;source

Uso:
    python convert_prompts.py input.csv output.csv

El CSV de entrada puede tener:
  - Cabecera 'prompt' (con o sin comillas)
  - Sin cabecera (una pregunta por fila en la primera columna)
  - Separador coma o punto y coma
"""

import csv
import sys
import os


def detect_separator(filepath):
    with open(filepath, encoding="utf-8") as f:
        sample = f.read(2048)
    semicolons = sample.count(";")
    commas = sample.count(",")
    return ";" if semicolons > commas else ","


def convert(input_path, output_path):
    sep = detect_separator(input_path)

    with open(input_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=sep)
        rows = list(reader)

    if not rows:
        print("El archivo de entrada está vacío.")
        return

    # Detectar si la primera fila es cabecera
    first = rows[0][0].strip().lower()
    has_header = first in ("prompt", '"prompt"', "'prompt'")
    data_rows = rows[1:] if has_header else rows

    with open(output_path, encoding="utf-8", newline="") as f_check:
        pass  # just ensure writable; open below

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["prompt", "label", "source"])
        for row in data_rows:
            if not row:
                continue
            prompt = row[0].strip()
            if prompt:
                writer.writerow([prompt, "good", "AI"])

    total = sum(1 for r in data_rows if r and r[0].strip())
    print(f"Convertidos {total} prompts → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python convertAIprompts.py <input.csv> <output.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"No se encuentra el archivo: {input_file}")
        sys.exit(1)

    # Crear archivo de salida si no existe (para verificar escritura)
    open(output_file, "a").close()

    convert(input_file, output_file)