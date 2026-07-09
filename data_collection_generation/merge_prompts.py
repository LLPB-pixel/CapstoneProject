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

# --- Configuración ---
MAX_PROMPT_LENGTH = 2000

english_words = {"the", "and", "to", "of", "in", "is", "that", "it", "for", "on", "with"}
spanish_words = {"el", "la", "los", "las", "de", "que", "en", "un", "una", "por", "para", "con"}
french_words  = {"le", "la", "les", "de", "et", "dans", "un", "une", "pour", "qui", "que", "sur", "à"}
german_words  = {"der", "die", "das", "und", "in", "zu", "den", "auf", "für", "mit", "von", "ist"}


def remove_emojis(text):
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)


def detect_languages(text):
    words = set(re.findall(r'\b\w+\b', text.lower()))
    langs = []
    if words & english_words: langs.append('English')
    if words & spanish_words: langs.append('Spanish')
    if words & french_words:  langs.append('French')
    if words & german_words:  langs.append('German')
    return langs


def detect_separator(filepath):
    with open(filepath, encoding="utf-8-sig") as f:
        sample = f.read(4096)
    return ";" if sample.count(";") > sample.count(",") else ","


def read_csv(filepath):
    sep = detect_separator(filepath)
    with open(filepath, encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.reader(f, delimiter=sep))

    if not all_rows:
        return []

    first_cell = all_rows[0][0].strip().lower() if all_rows[0] else ""
    has_header = first_cell == "prompt"
    header = all_rows[0] if has_header else None
    data = all_rows[1:] if has_header else all_rows

    if header:
        h = [c.strip().lower() for c in header]
        idx_prompt = h.index("prompt") if "prompt" in h else 0
        idx_label  = h.index("label")  if "label"  in h else None
        idx_source = h.index("source") if "source" in h else None
    else:
        idx_prompt, idx_label, idx_source = 0, 1, 2

    rows = []
    for row in data:
        if not row or not row[idx_prompt].strip():
            continue
        rows.append({
            "prompt": row[idx_prompt].strip(),
            "label":  (row[idx_label].strip()  if idx_label  is not None and len(row) > idx_label  else "good") or "good",
            "source": (row[idx_source].strip() if idx_source is not None and len(row) > idx_source else "AI")   or "AI",
        })
    return rows


def filter_and_clean(rows):
    lang_counts = {'English': 0, 'Spanish': 0, 'French': 0, 'German': 0, 'Multiple': 0, 'Unknown/Other': 0}
    filtered_out = {'too_long': 0, 'no_language': 0}
    kept = []

    for row in rows:
        prompt = row["prompt"]

        if len(prompt) > MAX_PROMPT_LENGTH:
            filtered_out['too_long'] += 1
            continue

        prompt = remove_emojis(prompt)
        langs = detect_languages(prompt)

        if len(langs) == 0:
            lang_counts['Unknown/Other'] += 1
            filtered_out['no_language'] += 1
            continue
        elif len(langs) == 1:
            lang_counts[langs[0]] += 1
        else:
            lang_counts['Multiple'] += 1

        row["prompt"] = prompt
        kept.append(row)

    return kept, lang_counts, filtered_out


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
    rows1 = read_csv(file1)
    print(f"Leyendo {file2}...")
    rows2 = read_csv(file2)

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
    final, lang_counts, filtered_out = filter_and_clean(deduped)

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
    print(f"✂️  Demasiado largos:       {filtered_out['too_long']:>6}")
    print(f"🌐 Sin idioma reconocido:  {filtered_out['no_language']:>6}")
    print(f"✅ Resultado final:         {len(final):>6} prompts únicos → {output}")
    print(f"\n🌍 Distribución de idiomas:")
    for lang, count in lang_counts.items():
        print(f"   {lang:<15}: {count}")

    plot_histograms(final)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python merge_prompts.py <archivo1.csv> <archivo2.csv> <output.csv>")
        sys.exit(1)

    f1 = "combined_prompts_dataset.csv"
    f2 = "IA_defs.csv"
    out = "combined_prompts_dataset.csv"

    for f in (f1, f2):
        if not os.path.exists(f):
            print(f"❌ No se encuentra: {f}")
            sys.exit(1)

    merge(f1, f2, out)